import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
import peft
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import disabled_train
from lavis.models.bevllm_models.bevllm import BEVLLMBase as Blip2Base 

from timm import create_model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

@registry.register_model("vicuna_drive")
class Blip2VicunaDrive(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        encoder_model="",
        encoder_model_ckpt='',
        load_pretrained=True,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llm_model="",
        max_txt_len=128,
        use_extra_prompt=False,
        freeze_decoder_of_bev_encoder=False,
        has_qformer=True,
        has_gru_decoder=False,
        has_lora=True,
        split_section_num_for_bev_encoder=2, # save gpu memory
    ):
        super().__init__()
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM

        self.use_extra_prompt = use_extra_prompt
        self.freeze_decoder_of_bev_encoder = freeze_decoder_of_bev_encoder

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.has_qformer = has_qformer
        self.has_gru_decoder = has_gru_decoder
        self.has_lora = has_lora
        self.split_section_num_for_bev_encoder = split_section_num_for_bev_encoder


        self.bev_encoder = create_model(encoder_model) 
        logging.info(f"number of output features: {self.bev_encoder.embed_dim}") # 256
        
        self.ln_vision = LayerNorm(self.bev_encoder.embed_dim)
        if load_pretrained:
            pretrain_weights = torch.load(encoder_model_ckpt, map_location=torch.device('cpu'))['state_dict']
            self.bev_encoder.load_state_dict(pretrain_weights, strict=False)

        if freeze_vit:
            for name, param in self.bev_encoder.named_parameters():
                if not self.freeze_decoder_of_bev_encoder:
                    if 'decoder' not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
            self.bev_encoder = self.bev_encoder.eval()
            self.bev_encoder.train = disabled_train


        if 'opt' in llm_model:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side='left')
            self.llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)


        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        if self.has_gru_decoder:
            self.waypoints_fc = nn.Sequential(
                        nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.llm_model.config.hidden_size, 64)
            )
            self.waypoints_predictor = nn.GRUCell(input_size=2, hidden_size=64)
            self.waypoints_output = nn.Linear(64, 2)
        else:
            self.waypoints_predictor = nn.Sequential(
                            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.llm_model.config.hidden_size, 10)
            )
        self.end_predictor = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm_model.config.hidden_size, 2)
        )
        

        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                4, self.bev_encoder.embed_dim #256
            )
            self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
            self.Qformer.cls = None  


        if self.has_lora:
            loraconfig = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm_model = get_peft_model(self.llm_model, loraconfig)
            self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len

        self.waypoints_loss = torch.nn.L1Loss()
        self.end_loss = torch.nn.CrossEntropyLoss()

    def concat_text_image_input(self, input_embeds, input_atts, input_embeds, image_nums, end_flag_pos_list, image_atts=None):
        '''
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = input_embeds.size()[0]
        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            if image_atts is None:
                bs, t, n, dim = input_embeds.size()
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        input_embeds[i].view(t*n, -1),
                        input_embeds[i][this_input_ones:]
                    ])
                )
            else:
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        input_embeds[i],
                        input_embeds[i][this_input_ones:]
                    ])
                )
            if image_atts is None:
                bs, t, n, dim = input_embeds.size()
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        torch.ones((image_nums[i]*n), device=input_embeds.device, dtype=torch.long),
                        torch.zeros(((t-image_nums[i])*n), device=input_embeds.device, dtype=torch.long),
                        input_atts[i][this_input_ones:]
                    ])
                )
            else:
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        image_atts[i],
                        input_atts[i][this_input_ones:]
                    ])
                )
            sub_target_index = []
            for j in end_flag_pos_list[i]:
                sub_target_index.append([i, j + this_input_ones])
            wp_target_index.extend(sub_target_index)
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index

    def concat_text_image_input_with_notice(self, input_embeds, input_atts, input_embeds, image_nums,
                                            end_flag_pos_list, notice_frame_id, notice_text, image_atts=None):
        '''
        the function is made for processing data with [inserted] notice text
        notice_frame_id: how many image frames before the notice
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = input_embeds.size()[0]

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            notice_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(input_embeds.device)
        input_notice_atts = text_input_tokens.attention_mask
        notice_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)

        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)

            this_notice_input_ones = input_notice_atts[i].sum()
            if image_atts is None:
                bs, t, n, dim = input_embeds.size()
                if notice_frame_id[i] <= 0: # which means the scenario do not include any notice
                    llm_inputs.append(
                        torch.cat([
                            input_embeds[i][:this_input_ones],
                            input_embeds[i].view(t*n, -1),
                            input_embeds[i][this_input_ones:],
                            notice_embeds[i][:],
                        ])
                    )
                else:
                    llm_inputs.append(
                        torch.cat([
                            input_embeds[i][:this_input_ones],
                            input_embeds[i, :notice_frame_id[i]].view(notice_frame_id[i]*n, -1),
                            notice_embeds[i][:this_notice_input_ones],
                            input_embeds[i, notice_frame_id[i]:].view((t-notice_frame_id[i])*n, -1),
                            input_embeds[i][this_input_ones:],
                            notice_embeds[i][this_notice_input_ones:],
                        ])
                    )
            else:
                pass 
            if image_atts is None:
                bs, t, n, dim = input_embeds.size()
                if notice_frame_id[i] < 0: # which means the scenario do not include any notice
                    llm_attention_mask.append(
                        torch.cat([
                            input_atts[i][:this_input_ones],
                            torch.ones((image_nums[i]*n), device=input_embeds.device, dtype=torch.long),
                            torch.zeros(((t-image_nums[i])*n), device=input_embeds.device, dtype=torch.long),
                            torch.zeros((input_notice_atts.size(1)), device=input_embeds.device, dtype=torch.long),
                            input_atts[i][this_input_ones:]
                        ])
                    )
                else:
                    llm_attention_mask.append(
                        torch.cat([
                            input_atts[i][:this_input_ones],
                            torch.ones((image_nums[i]*n), device=input_embeds.device, dtype=torch.long),
                            input_notice_atts[i][:this_notice_input_ones],
                            torch.zeros(((t-image_nums[i])*n), device=input_embeds.device, dtype=torch.long),
                            input_atts[i][this_input_ones:],
                            input_notice_atts[i][this_notice_input_ones:],
                        ])
                    )
            else:
                pass
            sub_target_index = []
            for j in range(len(end_flag_pos_list[i])):
                if j < notice_frame_id[i] or notice_frame_id[i] < 0: # when notice is '', the input_ones is 1, not ZERO
                    sub_target_index.append([i, end_flag_pos_list[i][j] + this_input_ones])
                else:
                    sub_target_index.append([i, end_flag_pos_list[i][j] + this_input_ones + this_notice_input_ones])
            wp_target_index.extend(sub_target_index)
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index

    def build_gt_waypoints(self, waypoints, valid_frames):
        gt_waypoints = []
        for i in range(waypoints.size(0)):
            gt_waypoints.append(waypoints[i, :valid_frames[i]])
        gt_waypoints = torch.cat(gt_waypoints, dim=0)
        return gt_waypoints

    def build_gt_end_flags(self, valid_frames):
        gt_end_flags = []
        for i in range(len(valid_frames)):
            gt_end_flags.extend([0]*(valid_frames[i]-1))
            gt_end_flags.append(1)
        gt_end_flags = torch.tensor(gt_end_flags, device=valid_frames.device).long()
        return gt_end_flags

    def prompt_wrap(self, input_embeds, text_before_img, text_after_img, valid_frames):
        bs, t, n, dim = input_embeds.size()
        emb_list = []
        end_flag_pos_list = []
        for i in range(bs):
            before_texts = text_before_img[i].split('|')
            after_texts = text_after_img[i].split('|')
            temp_embeds = []
            temp_end_flag_pos_list = []
            for j in range(valid_frames[i]):
                p_before_tokens = self.llm_tokenizer(before_texts[j], return_tensors="pt", add_special_tokens=False).to(input_embeds.device)
                p_after_tokens = self.llm_tokenizer(before_texts[j], return_tensors="pt", add_special_tokens=False).to(input_embeds.device)
                p_before_embed = self.llm_model.get_input_embeddings()(p_before_tokens.input_ids)
                p_after_embed = self.llm_model.get_input_embeddings()(p_after_tokens.input_ids)
                p_embed = torch.cat([p_before_embed, input_embeds[i][j][None], p_after_embed], dim=1)
                temp_embeds.append(p_embed)
                temp_end_flag_pos_list.append(p_embed.size(1)-1)
            end_flag_pos_list.append(temp_end_flag_pos_list)
            emb_list.append(torch.cat(temp_embeds, dim=1)) # 1 * m * d_dim
        emb_lens = [emb.shape[1] for emb in emb_list]
        pad_emb = self.llm_model.get_input_embeddings()(torch.tensor(self.llm_tokenizer.pad_token_id, device=input_embeds.device))
        wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.long, device=input_embeds.device)
        for i, emb in enumerate(emb_list):
            wrapped_embs[i, :emb_lens[i]] = emb
            wrapped_atts[i, :emb_lens[i]] = 1
        return wrapped_embs, wrapped_atts, end_flag_pos_list

    def split_data(self, samples):
        res = []
        splited_samples = {}
        split_size = samples['rgb'].size(0) // self.split_section_num_for_bev_encoder
        for key in ['rgb', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'target_point','measurements']:
            splited_samples[key] = torch.split(samples[key], split_size_or_sections=split_size, dim=0)
        for i in range(self.split_section_num_for_bev_encoder):
            new_samples = {}
            for key in ['rgb', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'target_point', 'measurements']:
                new_samples[key] = splited_samples[key][i]
            res.append(new_samples)
        return res

    def forward(self, samples, inference_mode=False, input_embeds=None):
        if input_embeds is None: # train mode
            device = samples["rgb"].device
            bs = samples['rgb'].size(0)
            t = samples['rgb'].size(1)

            for key in ['rgb', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'target_point', 'measurements']:
                shapz = samples[key].size()
                samples[key] = samples[key].view(bs*t, *shapz[2:])

            if self.freeze_decoder_of_bev_encoder:
                with torch.no_grad():
                    with self.maybe_autocast():
                        input_embeds_full = []
                        splited_samples = self.split_data(samples)
                        for i in range(self.split_section_num_for_bev_encoder):
                            input_embeds = self.bev_encoder(splited_samples[i])
                            input_embeds_full.append(input_embeds)
                            
                        input_embeds = torch.cat(input_embeds_full, dim=0)
            else:
                with self.maybe_autocast():
                    input_embeds = self.bev_encoder(samples)
        else: # inference mode
            device = input_embeds.device
            bs = input_embeds.size(0)
            t = input_embeds.size(1)
            input_embeds = input_embeds.view(bs*t, *input_embeds.size()[2:])

        input_embeds = self.ln_vision(input_embeds)
        
        if self.has_qformer:
            query_tokens = self.query_tokens.expand(input_embeds.shape[0], -1, -1)
            text_Qformer = self.llm_tokenizer(
                [i for i in samples['text_input'] for _ in range(t)],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
            image_atts = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(device)
            
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=input_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        input_embeds = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        input_embeds = input_embeds.view(bs, t, *input_embeds.size()[1:])

        if self.use_extra_prompt:
            text_before_img = samples['text_before_img']
            text_after_img = samples['text_after_img']
            input_embeds, image_atts, end_flag_pos_list = self.prompt_wrap(input_embeds, text_before_img, text_after_img, samples['valid_frames'])
        else:
            image_atts = None
            end_flag_pos_list = []
            n_length = input_embeds.size(2)
            for i in range(bs):
                end_flag_pos_list.append([n_length*(j+1)-1 for j in range(samples['valid_frames'][i])])

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)

     
        llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index = self.concat_text_image_input(inputs_embeds, text_input_tokens.attention_mask,
                                                                                                                   input_embeds, samples['valid_frames'], end_flag_pos_list, image_atts)
        wp_target_index = torch.tensor(wp_target_index, device=device).long()

        with self.maybe_autocast():
            hidden_states = self.llm_model(
                inputs_embeds=llm_inputs,
                attention_mask=llm_attention_mask,
                return_dict=False,
            )


        if self.has_gru_decoder:
            output_wp = []
            _, n_tokens, _ =hidden_states.size()
            x = torch.zeros(size=(bs*n_tokens, 2), dtype=hidden_states.dtype).to(device)
            target_point = samples['target_point'].view(bs, -1, 2).to(device)

            target_point_list = []
            for i in range(bs):
                target_point_list.append(target_point[i, :samples['valid_frames'][i], :])
            target_point = torch.cat(target_point_list, 0)


            target_point_zeros = torch.zeros(size=(bs, n_tokens, 2), dtype=hidden_states.dtype).to(device)
            target_point_zeros[wp_target_index[:,0], wp_target_index[:, 1]] = target_point.to(hidden_states.dtype)
            target_point_zeros = target_point_zeros.view(bs*n_tokens, 2)
            target_point = target_point_zeros

            waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, self.llm_model.config.hidden_size))
            for _ in range(5):
                x_in = x# + target_point
                waypoints_feature = self.waypoints_predictor(x_in, waypoints_feature)
                dx = self.waypoints_output(waypoints_feature)
                x = dx + x
                output_wp.append(x)
            predicted_waypoints = torch.cat(output_wp, dim=1)
            predicted_waypoints = predicted_waypoints.view(bs, n_tokens, 10)

        else:
            predicted_waypoints = self.waypoints_predictor(hidden_states)

        predicted_waypoints = predicted_waypoints[wp_target_index[:,0], wp_target_index[:, 1]]
        predicted_end_prob = self.end_predictor(hidden_states)
        predicted_end_prob = predicted_end_prob[wp_target_index[:,0], wp_target_index[:, 1]]

     

        if inference_mode:
            return predicted_waypoints, predicted_end_prob

        gt_waypoints = self.build_gt_waypoints(samples['local_future_waypoints'], samples['valid_frames'])
        waypoints_loss = self.waypoints_loss(predicted_waypoints, gt_waypoints)

        gt_end_flags = self.build_gt_end_flags(samples['valid_frames'])
        end_loss = self.end_loss(predicted_end_prob, gt_end_flags)

        predicted_end = torch.argmax(predicted_end_prob, dim=1)
        end_acc = (predicted_end == gt_end_flags).float().mean().item()

        #TODO: Implement training logic
        stage2_full = False  
        threshold = 0.2
        target_acc = 0.8

        if waypoints_loss < threshold and end_acc > target_acc:
            stage2_full = True


        total_loss = waypoints_loss + end_loss * 0.2
     
        

            

        return {"loss": total_loss, 'waypoints_loss': waypoints_loss, 'end_loss': end_loss, 'end_acc': end_acc}

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'bev_encoder' in name:
                group_name = "vit_layer_%s" % (group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        optim_params = list(parameter_group_vars.values())
        return optim_params


    @classmethod
    def from_config(cls, cfg):
        encoder_model = cfg.get("encoder_model")
        encoder_model_ckpt = cfg.get("encoder_model_ckpt")
        load_pretrained = cfg.get('load_pretrained', True)
        img_size = cfg.get("image_size")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 64)
        use_extra_prompt = cfg.get("use_extra_prompt", False)
        freeze_decoder_of_bev_encoder = cfg.get("freeze_decoder_of_bev_encoder", False)
        has_gru_decoder = cfg.get("has_gru_decoder", False)
        has_lora = cfg.get('has_lora', True)
        split_section_num_for_bev_encoder = cfg.get('split_section_num_for_bev_encoder', 2)

        model = cls(
            img_size=img_size,
            encoder_model=encoder_model,
            encoder_model_ckpt=encoder_model_ckpt,
            load_pretrained=load_pretrained,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llm_model=llm_model,
            max_txt_len=max_txt_len,
            use_extra_prompt=use_extra_prompt,
            freeze_decoder_of_bev_encoder=freeze_decoder_of_bev_encoder,
            has_gru_decoder=has_gru_decoder,
            has_lora=has_lora,
            split_section_num_for_bev_encoder=split_section_num_for_bev_encoder,
        )


        return model
