"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.bevllm_models.bevllm import (
    BEVLLMBase,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.bevllm_models.bevllm_outputs import BEVLLMOutput, BEVLLMOutputFeatures
from mmdet3d.apis import init_model

@registry.register_model("bevllm_bevdet")
@registry.register_model("bevllm_bevdet_feature_extractor")
class BEVLLMQformerBEVdet(BEVLLMBase):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=128,
    ):
        super().__init__()

        rank = dist.get_rank()
        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 16384, cross_attention_freq
        )
        # resize vocab size with padding to multiple of 
        self.Qformer.resize_token_embeddings(len(self.tokenizer), 8)
        
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, samples):
        
        bev = samples['bev']
        text = samples["text"]
        
        #print(f'bev sample: {bev}')
        bev_embeds = bev
        #print(f'bev embeds: {bev_embeds}')
        bev_embeds_shape = bev_embeds.shape
        #print(f'bev embeds shape: {bev_embeds_shape}')
        bev_embeds = torch.flatten(bev_embeds, start_dim=2)
        #print(f'bev embeds shape after flatten: {bev_embeds.shape}')
        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )
        #print(f'bev atts size: {bev_embeds.size()[:-1]}')
        #print(f' query tokens before expansion {self.query_tokens.shape}')
        # expand query tokens to bev fusion batch size 
        query_tokens = self.query_tokens.expand(bev_embeds.shape[0], -1, -1).to(bev_embeds.device)
        #print(f'query tokens after expansion: {query_tokens.shape}')

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=bev_embeds,
            encoder_attention_mask=bev_atts,
            use_cache=True,
            return_dict=True,
        )
        #print(f'query output: {query_output[0].shape}')
        # normalize the output of linear layer to project to shared embedding space
        bev_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(bev_embeds.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        bev_feats_all = concat_all_gather(
            bev_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        #print(f'bev_feats shape {bev_feats_all.shape}')
        #print(f'text feats all shape {text_feat_all.shape}')
        #print(f'are feature tensors the same? {torch.equal(bev_feats_all, bev_feats)}')
        sim_q2t = torch.matmul(
            bev_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        #print(f'matmul squeezed: {_.squeeze().shape}')
        #print(f'matmul unsqueezed: {_.shape}')
        #print(f' unsqueezed bev: {bev_feats.unsqueeze(1).shape}')
        #print(f' unsqueezed text: {text_feat_all.unsqueeze(-1).shape}')
        #print(sim_q2t.shape)
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        #print(f'sim_i2t before division: {sim_i2t.shape}')
        sim_i2t = sim_i2t / self.temp
        #print(f'sim_i2t after division: {sim_i2t}')
        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), bev_feats_all.permute(0, 2, 1)
        ).squeeze()
        #print(f'sim t2q: {sim_t2q.shape}')
        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        rank = dist.get_rank()
        bs = bev_embeds.size(0)

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            bev_embeds.device
        )
        #print(f'targets: {targets.shape}')
        # adjust targets to float
        #targets = targets.to(torch.float)

        #print(f'targets: {targets.shape}')  
        #print(f'sim_i2t: {sim_i2t.shape}')             
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        bev_embeds_world = all_gather_with_grad(bev_embeds)
        world_size = dist.get_world_size()
        #print(f'shape of similarity {sim_t2i.shape}')
        with torch.no_grad():
            if bs == 1:
                sim_t2i[0] = -10000
                sim_i2t[0] = -10000   

                weights_t2i = F.softmax(sim_t2i, dim=0)
                weights_i2t = F.softmax(sim_i2t, dim=0)           
            else:
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                    
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            #print(f'weight t2i: {weights_t2i}')
            if bs == 1:
                neg_idx = 0 
            else:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(bev_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            if bs == 1:
                neg_idx = 0
            else:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [bev_embeds, image_embeds_neg, bev_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(bev_embeds.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BEVLLMOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        #image = samples.get("image")
        #caption = samples.get("text_input")

        bev = samples.get("bev")
        caption = samples.get("text")
        

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "bev",
            "text",
            "multimodal",
        ], "mode must be one of 'bev', 'text', 'multimodal'"

        # initalize output
        bev_embeds, text_embeds, multimodal_embeds = None, None, None
        bev_features, text_features = None, None

        if mode == "bev":
            assert (
                bev is not None
            ), "Bev is not provided for mode 'bev' or 'multimodal'"
            # return query features

            bev_embeds = bev
            bev_embeds = torch.flatten(bev_embeds, start_dim=2)
            bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(
                bev_embeds.device)
            query_tokens = self.query_tokens.expand(bev_embeds.shape[0], -1, -1).to(bev_embeds.device)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=bev_embeds,
                encoder_attention_mask=bev_atts,
                use_cache=True,
                return_dict=True,
            )
            bev_embeds = query_output.last_hidden_state
            bev_features = F.normalize(self.vision_proj(bev_embeds), dim=-1)

            

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            #print(f' last hidden state text output : {text_embeds.shape}')
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            bev_embeds = self.bev_fusion.test_step(bev)[0]
            bev_embeds = torch.flatten(bev_embeds, start_dim=2)
            bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(
                bev_embeds.device)
            query_tokens = self.query_tokens.expand(bev_embeds.shape[0], -1, -1).to(bev_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=bev_embeds,
                encoder_attention_mask=bev_atts,
                return_dict=True,
            )
            #print(f'multimodal embeds shape: {output.last_hidden_state.shape}')
            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BEVLLMOutputFeatures(
            bev_embeds=bev_embeds,
            bev_embeds_proj=bev_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
