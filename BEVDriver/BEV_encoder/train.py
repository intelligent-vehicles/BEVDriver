#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was originally based on:
- An early version of the PyTorch ImageNet example:
  https://github.com/pytorch/examples/tree/master/imagenet
- NVIDIA CUDA-specific speedups from NVIDIA Apex examples:
  https://github.com/NVIDIA/apex/tree/master/examples/imagenet
- The InterFuser repository:
  https://github.com/opendilab/InterFuser

Modifications copyright 2025 Intelligent Vehicles Lab
https://github.com/intelligent-vehicles/bevdriver

"""
import argparse
import time
import yaml
import os
import sys
import logging
import json
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from render import render

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

timm_path = os.path.abspath(".")
sys.path.insert(0, timm_path)


from timm.data import (
    create_dataset,
    create_carla_dataset,
    create_loader,
    create_carla_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)

from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument(
    "--train-towns",
    type=int,
    nargs="+",
    default=[0],
    help="dataset train towns (default: [0])",
)
parser.add_argument(
    "--val-towns",
    type=int,
    nargs="+",
    default=[1],
    help="dataset validation towns (default: [1])",
)
parser.add_argument(
    "--train-weathers",
    type=int,
    nargs="+",
    default=[0],
    help="dataset train weathers (default: [0])",
)
parser.add_argument(
    "--val-weathers",
    type=int,
    nargs="+",
    default=[1],
    help="dataset validation weathers (default: [1])",
)
parser.add_argument(
    "--saver-decreasing",
    action="store_true",
    default=False,
    help="StarAt with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--with-lidar",
    action="store_true",
    default=False,
    help="load lidar data in the dataset",
)
parser.add_argument(
    "--with-seg",
    action="store_true",
    default=False,
    help="load segmentation data in the dataset",
)
parser.add_argument(
    "--with-depth",
    action="store_true",
    default=False,
    help="load depth data in the dataset",
)
parser.add_argument(
    "--multi-view",
    action="store_true",
    default=False,
    help="load multi-view data in the dataset",
)
parser.add_argument(
    "--multi-view-input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224) for left- right- or rear- view",
)

parser.add_argument(
    "--freeze-num", type=int, default=-1, help="Number of freeze layers in the backbone"
)
parser.add_argument(
    "--backbone-lr", type=float, default=5e-4, help="The learning rate for backbone"
)
parser.add_argument(
    "--with-backbone-lr",
    action="store_true",
    default=False,
    help="The learning rate for backbone is set as backbone-lr",
)

# Dataset / Model parameters
parser.add_argument("data_dir", metavar="DIR", nargs="+", help="path to dataset")
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="newcarla",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--model",
    default="resnet101",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "countception"',
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--validation-batch-size-multiplier",
    type=int,
    default=1,
    metavar="N",
    help="ratio of validation batch size to training batch size (default: 1)",
)
parser.add_argument(
    "--augment-prob",
    type=float,
    default=0.0,
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd"',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)


# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=5e-4, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=5e-6,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode", type=str, default="const", help='Random erase mode (default: "const")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)"
)
parser.add_argument(
    "--smoothed_l1", type=bool, default=False, help="L1 smooth"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.1,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)

# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=2,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=5,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)

parser.add_argument(
    "--local-rank",
    type=int,
    default=0
)

#Interfuser Detection Loss
class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:6]
        target_1 = target[target_1_mask][:][:, 1:6]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 6]
        target_2 = target[target_1_mask][:][:, 6]
        if target_2.numel() == 0:
            loss_3 = 0
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3

#LMDrive Detection Loss
class LAVLoss:
    def __init__(self):
        self.prob_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loc_criterion = nn.L1Loss(reduction='none')
        self.ori_criterion = nn.L1Loss(reduction='none')
        self.box_criterion = nn.L1Loss(reduction='none')
        self.spd_criterion = nn.L1Loss(reduction='none')
        #self.loc_criterion = nn.SmoothL1Loss(reduction='none')
        #self.ori_criterion = nn.SmoothL1Loss(reduction='none')
        #self.box_criterion = nn.SmoothL1Loss(reduction='none')
        #self.spd_criterion = nn.SmoothL1Loss(reduction='none')

    def __call__(self, output, target):
        prob = target[:, : ,0:1]
        prob_mean = prob.mean()
        prob_mean = torch.maximum(prob_mean, torch.ones_like(prob_mean) * 1e-7)
        prob_det = torch.sigmoid(output[:, :, 0] * (1 - 2 * target[:, :, 0]))

        det_loss = (prob_det * self.prob_criterion(output[:, :, 0], target[:, :, 0])).mean() / prob_det.mean()
        loc_loss = (prob * self.loc_criterion(output[:, :, 1:3], target[:, :, 1:3])).mean() / prob_mean
        box_loss = (prob * self.box_criterion(output[:, :, 3:5], target[:, :, 3:5])).mean() / prob_mean
        ori_loss = (prob * self.ori_criterion(output[:, :, 5:7], target[:, :, 5:7])).mean() / prob_mean
        spd_loss = (prob * self.ori_criterion(output[:, :, 7:8], target[:, :, 7:8])).mean() / prob_mean
        

        det_loss = 0.4 * det_loss + 0.2 * loc_loss + 0.2 * box_loss + 0.2 * ori_loss
        return det_loss, spd_loss

#Cross Entropy
class SemanticLoss:
    def __init__(self):
        self.prob_criterion = nn.CrossEntropyLoss()

    def convert_to_tensor(mask):
        pass


    def __call__(self, output, input):
        pass


#Focal Loss
class FocalLoss:
    def __init__(self, alpha=0.9, gamma=5, reduction='mean', ignore_index=-100): 
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def __call__(self, output, target):

        target = target.long()
        ce_loss = F.cross_entropy(output, target, reduction='none', ignore_index=self.ignore_index)
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

#Contrastive Embedding loss
class ContrastiveLoss:
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(self, current_embedding, similar_embedding, distant_embedding):
        current = current_embedding.mean(dim=1)  
        similar = similar_embedding.mean(dim=1)
        distant = distant_embedding.mean(dim=1)

        positive_distance = torch.norm(current - similar, p=2, dim=1)
        negative_distance = torch.norm(current - distant, p=2, dim=1)
        
        positive_loss = positive_distance ** 2
        negative_loss = torch.clamp(self.margin - negative_distance, min=0) ** 2

        loss = (positive_loss + negative_loss).mean()

        return loss


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )
    args.prefetcher = not args.no_prefetcher
    args.distributed = True

    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        freeze_num=args.freeze_num,
    )

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    linear_scaled_lr = (
        args.lr * args.batch_size * torch.distributed.get_world_size() / 512.0
    )
    args.lr = linear_scaled_lr
    if args.with_backbone_lr:
        if args.local_rank == 0:
            _logger.info(
                "CNN backbone and transformer blocks using different learning rates!"
            )
        backbone_linear_scaled_lr = (
            args.backbone_lr
            * args.batch_size
            * torch.distributed.get_world_size()
            / 512.0
        )
        backbone_weights = []
        other_weights = []
        for name, weight in model.named_parameters():
            if "backbone" in name and "lidar" not in name:
                backbone_weights.append(weight)
            else:
                other_weights.append(weight)
        if args.local_rank == 0:
            _logger.info(
                "%d weights in the cnn backbone, %d weights in other modules"
                % (len(backbone_weights), len(other_weights))
            )
        optimizer = create_optimizer_v2(
            [
                {"params": other_weights},
                {"params": backbone_weights, "lr": backbone_linear_scaled_lr},
            ],
            **optimizer_kwargs(cfg=args),
        )
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != "native":
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[args.local_rank], find_unused_parameters=True
            )  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    _logger.info("Creating Datasets")
    # create the train and eval datasets
    if "carla" in args.dataset:
        dataset_train = create_carla_dataset(
            args.dataset,
            root=args.data_dir,
            towns=args.train_towns,
            weathers=args.train_weathers,
            batch_size=args.batch_size,
            with_lidar=args.with_lidar,
            with_seg=True,
            with_depth=args.with_depth,
            multi_view=True,
            augment_prob=args.augment_prob
        )
        dataset_eval = create_carla_dataset(
            args.dataset,
            root=args.data_dir,
            towns=args.val_towns,
            weathers=args.val_weathers,
            batch_size=args.batch_size,
            with_lidar=args.with_lidar,
            with_seg=True,
            with_depth=args.with_depth,
            multi_view=True,
            augment_prob=args.augment_prob
        )
    else:
        dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
        )
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
        )

    collate_fn = None
    mixup_fn = None
    _logger.info("Creating Data Loaders")
    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    loader_train = create_carla_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        multi_view_input_size=args.multi_view_input_size,
        is_training=True,
        scale=args.scale,
        color_jitter=args.color_jitter,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
    )

    loader_eval = create_carla_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        multi_view_input_size=args.multi_view_input_size,
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.smoothing > 0:
        cls_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        cls_loss = nn.CrossEntropyLoss()

    if args.smoothed_l1:
        l1_loss = torch.nn.SmoothL1Loss
    else:
        l1_loss = torch.nn.L1Loss

    train_loss_fns = {
        #"traffic": LAVLoss(),
        "semantic": FocalLoss(),
        "embeddings": ContrastiveLoss(),
        "cls": cls_loss,
        #"semantic": SemanticLoss(), #Alternatively to FocalLoss
        "traffic": MVTL1Loss(1.0, l1_loss=l1_loss),

    }
    validate_loss_fns = {
        #"traffic": LAVLoss(),
        "semantic": FocalLoss(),
        "embeddings": ContrastiveLoss(),
        "cls": cls_loss,
        #"semantic": SemanticLoss(), #Alternatively to FocalLoss
        "traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
    }

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    writer = None
    if args.rank == 0:
        if args.experiment:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                    args.experiment,
                ]
            )
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = get_outdir(args.output if args.output else "./output", exp_name)
        writer = SummaryWriter(log_dir=output_dir)
        decreasing = args.saver_decreasing
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fns,
                args,
                writer,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            eval_metrics = validate(
                epoch,
                model,
                loader_eval,
                validate_loss_fns,
                args,
                writer,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
                )
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def retransform(data):
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    data = data * std_tensor + mean_tensor
    return data


def traffic_iou(output, target):
    ''' Args: 
    Excpecting shapes torch.Size([1,20,20])

    output: Prediction, clip(output[0][0],0,1).view(1,20,20,7)[:,:,:,0]
    target: Ground Truth, target[1][0]
    
    Returns float (IoU value)
    '''

    # convert probabilities to binary values
    prediction = (output > 0.5).bool()
    target = (target > 0.5).bool()

    intersection = torch.sum(prediction & target).float()
    union = torch.sum(prediction | target).float()

    if union == 0.0:
        return 0.0
    
    iou = intersection / union 

    return iou.item()


def semantic_iou(output, target):
    ''' Args: 
    Excpecting shapes torch.Size([1,224,224])

    output: Prediction, torch.argmax(output[2].unsqueeze(0)), shape(1,H,W)
    target: Ground Truth, target[2][0], shape(1,H,W)
    
    Returns array of floats of length num_classes
    '''


    # flatten
    pred = output.view(-1) 
    target = target.view(-1)

    unique_target_values = torch.unique(target)
    num_classes = len(unique_target_values)

    iou_per_class = torch.zeros(num_classes, dtype=torch.float32)
    weighted_classes = torch.zeros(num_classes, dtype=torch.float32)


    #print("\nClass distribution:")
    for c in range(num_classes):
        pred_count = (pred == c).sum().item()
        target_count = (target == c).sum().item()


    # iterate over classes
    for c in range(1, num_classes+1):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = torch.sum(pred_c & target_c).float() 
        union = torch.sum(pred_c | target_c).float()
    
        if union > 0: #skip if union is 0 and class is not evident in either gt or prediction
            iou = intersection / union
            iou_per_class[c-1] = iou
            weighted_classes[c-1] = torch.sum(target_c).float() #store number of pixels in class
            
        
    return iou_per_class, weighted_classes


def get_batch_embeddings(embeddings, similar_window_size=10, distant_window_size=50):
    batch_size = embeddings.shape[0]
    device = embeddings.device

    # Initialize tensors to store selected indices
    similar_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    distant_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # For each frame, find valid similar and distant frames
    for i in range(batch_size):
        similar_range_start = i + 1
        similar_range_end = min(i + similar_window_size + 1, batch_size)
        
        if similar_range_start < similar_range_end:
            similar_indices[i] = torch.randint(
                similar_range_start,
                similar_range_end,
                (1,),
                device=device
            )
            
            distant_candidates = torch.cat([
                torch.arange(0, max(0, i - distant_window_size), device=device),
                torch.arange(min(batch_size, i + distant_window_size + 1), batch_size, device=device)
            ])
            
            if len(distant_candidates) > 0:
                distant_idx = torch.randint(0, len(distant_candidates), (1,), device=device)
                distant_indices[i] = distant_candidates[distant_idx]
                valid_mask[i] = True
    
    # Expand valid_mask to match embeddings dimensions
    valid_mask = valid_mask.unsqueeze(1).unsqueeze(2).expand(-1, embeddings.shape[1], embeddings.shape[2])
    
    # Gather the selected similar and distant embeddings
    similar_embeddings = embeddings[similar_indices]
    distant_embeddings = embeddings[distant_indices]

    return embeddings, similar_embeddings, distant_embeddings, valid_mask



def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fns,
    args,
    writer,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_seg = AverageMeter()
    losses_embeddings = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    traffic_iou_m = AverageMeter()
    semantic_iou_m = AverageMeter() #averaged over all classes (old version)
    semantic_iou_m_valid = AverageMeter() #only classes with valid pixels, unweighted (excluding not present classes)
    semantic_iou_m_weighted = AverageMeter() #weighted by number of pixels in each class

    with open('BEV_encoder/cityscapes_dict.json', 'r') as f:
        cityscapes_dict = json.load(f)

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    _logger.info("Len Loader: {}".format(len(loader)))

    for batch_idx, (input, target) in enumerate(loader):

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if isinstance(input, (tuple, list)):
            batch_size = input[0].size(0)
        elif isinstance(input, dict):
            batch_size = input[list(input.keys())[0]].size(0)
        else:
            batch_size = input.size(0)
        if not args.prefetcher:
            if isinstance(input, (tuple, list)):
                input = [x.cuda() for x in input]
            elif isinstance(input, dict):
                for key in input:
                    input[key] = input[key].cuda()
            else:
                input = input.cuda()
            if isinstance(target, (tuple, list)):
                target = [x.cuda() for x in target]
            elif isinstance(target, dict):
                for key in target:
                    target[key] = target[key].cuda()
            else:
                target = target.cuda()

        with amp_autocast():
            output = model(input)
            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[1])

            loss_traffic_light_state = loss_fns["cls"](
                output[3], target[4]
            )
            valid_mask = target[3]==1
         
            semantic_target = target[2].squeeze(1)
            semantic_target = (semantic_target*255).long()
            semantic_target = semantic_target[valid_mask]
            semantic_target[semantic_target >= 28] = -100


            semantic_pred = torch.argmax(output[6], dim=1)
            semantic_pred = semantic_pred[valid_mask]

            if semantic_target.numel() > 0:
                loss_semantic = loss_fns["semantic"](output[6][valid_mask], semantic_target)

            else:
                loss_semantic = torch.tensor(0.0, requires_grad=True).to(output[6].device)



            current, similar, distant, valid_mask = get_batch_embeddings(
                output[7], 
                similar_window_size=10, 
                distant_window_size=50
            )
            
            loss_embeddings = torch.tensor(0.0, device=current.device, requires_grad=True)

            if valid_mask[:, 0, 0].any():              
                loss_embeddings = loss_fns["embeddings"](
                    current[valid_mask[:, 0, 0]], 
                    similar[valid_mask[:, 0, 0]], 
                    distant[valid_mask[:, 0, 0]]
                )
           

            loss = (
                loss_traffic * 0.3
                + loss_semantic * 0.6
                + loss_traffic_light_state * 0.1
                + loss_embeddings * 0.2
            )

            traffic_pred = torch.clip(output[0][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0]
            traffic_target = target[1][0].view(1, 20, 20, 7)[:, :, :, 0]
            batch_traffic_iou = traffic_iou(traffic_pred, traffic_target)
            traffic_iou_m.update(batch_traffic_iou, batch_size)

            if valid_mask.sum() > 0:  
                batch_semantic_ious, batch_weighted_classes = semantic_iou(semantic_pred, semantic_target)
                semantic_iou_m.update(batch_semantic_ious.mean().item(), batch_size)

                valid_classes = batch_weighted_classes > 0 
                if valid_classes.sum() > 0:  
                    valid_ious = batch_semantic_ious[valid_classes]
                    semantic_iou_m_valid.update(valid_ious.mean().item(), batch_size)
                else:
                    semantic_iou_m_valid.update(0.0, batch_size)  
                
                if batch_weighted_classes.sum() > 0:
                    weighted_mean = (batch_semantic_ious * batch_weighted_classes).sum() / batch_weighted_classes.sum()
                    semantic_iou_m_weighted.update(weighted_mean.item(), batch_size)
                else:
                    semantic_iou_m_weighted.update(0.0, batch_size) 


        if not args.distributed:
            losses_traffic.update(loss_traffic.item(), batch_size)
            losses_m.update(loss.item(), batch_size)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), batch_size)
                reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
                losses_traffic.update(reduced_loss_traffic.item(), batch_size)
        
                reduced_loss_seg = reduce_tensor(
                    loss_semantic.data, args.world_size
                )
                losses_seg.update(reduced_loss_seg.item(), batch_size)
            
                losses_embeddings.update(loss_embeddings.item(), batch_size)
                reduced_losses_embeddings = reduce_tensor(loss_embeddings.data, args.world_size)
                
                reduced_loss_traffic_light_state = reduce_tensor(
                    loss_traffic_light_state.data, args.world_size
                )
                losses_traffic_light_state.update(
                    reduced_loss_traffic_light_state.item(), batch_size
                )
                if writer and args.local_rank == 0:
                    writer.add_scalar("train/loss", reduced_loss.item(), num_updates)
                    writer.add_scalar(
                        "train/loss_traffic", reduced_loss_traffic.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_seg", reduced_loss_seg.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_embeddings", reduced_losses_embeddings.item(), num_updates
                    )
           
                    writer.add_scalar("train/traffic_iou", traffic_iou_m.avg, num_updates)
                    writer.add_scalar("train/semantic_iou", semantic_iou_m.avg, num_updates) 
                    writer.add_scalar("train/semantic_iou_valid", semantic_iou_m_valid.avg, num_updates)
                    writer.add_scalar("train/semantic_iou_weighted", semantic_iou_m_weighted.avg, num_updates) 
                    writer.add_scalar(
                        "train/loss_traffic_light_state",
                        reduced_loss_traffic_light_state.item(),
                        num_updates,
                    )

                    writer.add_image(
                        "train/front_view", retransform(input["rgb"][0]), num_updates
                    )

                    semantic_pred_vis = semantic_pred[0]  
                    colors = torch.zeros((semantic_pred_vis.shape[0], semantic_pred_vis.shape[1], 3), dtype=torch.uint8, device=semantic_pred.device)
                    for class_idx in cityscapes_dict:
                        mask = (semantic_pred_vis == class_idx)
                        colors[mask] = torch.tensor(cityscapes_dict[str(max(0, int(class_idx)))]["color"], dtype=torch.uint8, device=semantic_pred.device)
                    colors = colors.permute(2, 0, 1)

                    semantic_target_vis = semantic_target[0] 
                    colors_gt = torch.zeros((semantic_target_vis.shape[0], semantic_target_vis.shape[1], 3), dtype=torch.uint8, device=semantic_target.device)
                    for class_idx in cityscapes_dict:
                        mask = (semantic_target_vis == class_idx) 
                        colors_gt[mask] = torch.tensor(cityscapes_dict[str(max(0, int(class_idx)))]["color"], dtype=torch.uint8, device=semantic_target.device)
                    colors_gt = colors_gt.permute(2, 0, 1)
                                        
                    assert semantic_pred.ndimension() == 3, f"Expected 3D tensor (C, H, W), but got {semantic_pred.shape}"
                    
                    writer.add_image(
                        "train/seg_image",
                        colors,
                        num_updates,
                        dataformats="CHW"
                    )

                    writer.add_image(
                        "train/gt_seg_view", 
                        colors_gt,
                        num_updates,
                        dataformats="CHW"
                    )


                    writer.add_image(
                        "train/left_view",
                        retransform(input["rgb_left"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/right_view",
                        retransform(input["rgb_right"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/front_center_view",
                        retransform(input["rgb_center"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_traffic",
                        (torch.clip(output[0][0], 0, 1)*255).view(1, 20, 20, 7)[:, :, :, 0],
                        num_updates,
                    )

                    writer.add_image(
                        "train/pred_traffic_render",
                        torch.clip(
                            torch.tensor(
                                render(
                                    output[0][0].view(20, 20, 7).detach().cpu().numpy()
                                )[:100, 40:140]
                            ),
                            0,
                            255,
                        ).view(1, 100, 100),
                        num_updates,
                    )
                    input["lidar"][0] = input["lidar"][0] / torch.max(input["lidar"][0])
                    writer.add_image(
                        "train/lidar", torch.clip(input["lidar"][0], 0, 1), num_updates
                    )
                    writer.add_image(
                        "train/gt_traffic",
                        torch.clip(target[1][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0],
                        num_updates,
                    )
                    writer.add_image(
                        "train/gt_highres_traffic",
                        torch.clip(target[0][0], 0, 1),
                        num_updates,
                    )



            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Loss(traffic): {loss_traffic.val:>9.6f} ({loss_traffic.avg:>6.4f})  "
                    "Loss(semantic): {loss_semantic.val:>9.6f} ({loss_semantic.avg:>6.4f})  "
                    "Loss(embeddings): {loss_embeddings.val:>9.6f} ({loss_embeddings.avg:>6.4f})  "
                    "Traffic IoU: {traffic_iou.val:.3f} ({traffic_iou.avg:.3f})  "
                    "Semantic IoU: {semantic_iou.val:.3f} ({semantic_iou.avg:.3f})  "
                    "Semantic IoU Valid: {semantic_iou_valid.val:.3f} ({semantic_iou_valid.avg:.3f})  "
                    "Semantic IoU Weighted: {semantic_iou_weighted.val:.3f} ({semantic_iou_weighted.avg:.3f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})"
                    "Loss (traffic_light_state): {loss_traffic_light_state.val:.3f} ({loss_traffic_light_state.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        loss_traffic=losses_traffic,
                        loss_semantic=losses_seg,
                        loss_embeddings=losses_embeddings,
                        traffic_iou=traffic_iou_m,
                        semantic_iou=semantic_iou_m,
                        semantic_iou_valid=semantic_iou_m_valid,
                        semantic_iou_weighted=semantic_iou_m_weighted,
                        batch_time=batch_time_m,
                        rate=batch_size * args.world_size / batch_time_m.val,
                        rate_avg=batch_size * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                        loss_traffic_light_state=losses_traffic_light_state,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()


    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    epoch, model, loader, loss_fns, args, writer, amp_autocast=suppress, log_suffix=""
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    l1_errorm = AverageMeter()
    losses_seg = AverageMeter()
    
    losses_embeddings = AverageMeter()
    traffic_iou_m = AverageMeter()
    semantic_iou_m = AverageMeter() 
    semantic_iou_m_valid = AverageMeter() 
    semantic_iou_m_weighted = AverageMeter() 

    with open('cityscapes_dict.json', 'r') as f:
        cityscapes_dict = json.load(f)

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if isinstance(input, (tuple, list)):
                batch_size = input[0].size(0)
            elif isinstance(input, dict):
                batch_size = input[list(input.keys())[0]].size(0)
            else:
                batch_size = input.size(0)
            if isinstance(input, (tuple, list)):
                input = [x.cuda() for x in input]
            elif isinstance(input, dict):
                for key in input:
                    input[key] = input[key].cuda()
            else:
                input = input.cuda()
            if isinstance(target, (tuple, list)):
                target = [x.cuda() for x in target]
            elif isinstance(target, dict):
                for key in target:
                    input[key] = input[key].cuda()
            else:
                target = target.cuda()

            with amp_autocast():
                output = model(input)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[1])
            loss_traffic_light_state = loss_fns["cls"](
                output[3], target[4]
            )

            valid_mask = target[3] == 1  
    
            semantic_target = target[2].squeeze(1)
            
            semantic_target = semantic_target[valid_mask]

            semantic_pred = torch.argmax(output[6], dim=1)
            semantic_pred = semantic_pred[valid_mask]

            if semantic_target.numel() > 0:  
                loss_semantic = loss_fns["semantic"](output[6][valid_mask], semantic_target)
            else:
                loss_semantic = torch.tensor(0.0, requires_grad=True).to(output[6].device)

            current, similar, distant, valid_mask = get_batch_embeddings(
                output[7], 
                similar_window_size=10, 
                distant_window_size=50
            )

            loss_embeddings = torch.tensor(0.0, device=current.device, requires_grad=True)
            
            if valid_mask[:, 0, 0].any():  
                loss_embeddings = loss_fns["embeddings"](
                    current[valid_mask[:, 0, 0]], 
                    similar[valid_mask[:, 0, 0]], 
                    distant[valid_mask[:, 0, 0]]
                )
        
            loss = (
                loss_traffic * 0.3
                + loss_semantic * 0.6
                + loss_traffic_light_state * 0.1
                #+ loss_embeddings * 0.2 
            )

            traffic_pred = torch.clip(output[0][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0]
            traffic_target = target[1][0].view(1, 20, 20, 7)[:, :, :, 0]
            batch_traffic_iou = traffic_iou(traffic_pred, traffic_target)
            traffic_iou_m.update(batch_traffic_iou, batch_size)


            if valid_mask.sum() > 0:  
                batch_semantic_ious, batch_weighted_classes = semantic_iou(semantic_pred, semantic_target)
                semantic_iou_m.update(batch_semantic_ious.mean().item(), batch_size)
                semantic_iou_m_valid.update(batch_semantic_ious[batch_weighted_classes > 0].mean().item(), batch_size)
                if batch_weighted_classes.sum() > 0:
                    semantic_iou_m_weighted.update((batch_semantic_ious * batch_weighted_classes).sum().item() / batch_weighted_classes.sum().item(), batch_size)
                else: 
                    print("No valid classes found. Skipping weighted IoU calculation.")
                    print("Batch weighted classes: ", batch_weighted_classes)
                    print("Batch semantic ious: ", batch_semantic_ious)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
         
                reduced_loss_seg = reduce_tensor(loss_semantic.data, args.world_size)
                reduced_loss_embeddings = reduce_tensor(loss_embeddings.data, args.world_size)
                reduced_loss_traffic_light_state = reduce_tensor(loss_traffic_light_state.data, args.world_size)
            else:
                reduced_loss = loss.data
                reduced_loss_seg = reduce_tensor(loss_semantic.data, args.world_size)
                reduced_loss_embeddings = reduce_tensor(loss_embeddings.data, args.world_size)
                reduced_loss_traffic_light_state = reduce_tensor(loss_traffic_light_state.data, args.world_size)

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), batch_size)
            losses_traffic.update(reduced_loss_traffic.item(), batch_size)
            losses_seg.update(reduced_loss_seg.item(), batch_size)
            losses_embeddings.update(reduced_loss_embeddings.item(), batch_size)
            losses_traffic_light_state.update(reduced_loss_traffic_light_state.item(), batch_size)
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Loss(traffic): {loss_traffic.val:>7.4f} ({loss_traffic.avg:>6.4f})  "
                    "Loss(semantic): {loss_semantic.val:>9.6f} ({loss_semantic.avg:>6.4f})  "
                    "Loss(embeddings): {loss_embeddings.val:>9.6f} ({loss_embeddings.avg:>6.4f})  "
                    "Traffic IoU: {traffic_iou.val:.3f} ({traffic_iou.avg:.3f})  "
                    "Semantic IoU: {semantic_iou.val:.3f} ({semantic_iou.avg:.3f})  "
                    "Semantic IoU Valid: {semantic_iou_valid.val:.3f} ({semantic_iou_valid.avg:.3f})  "
                    "Semantic IoU Weighted: {semantic_iou_weighted.val:.3f} ({semantic_iou_weighted.avg:.3f})  ".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        loss_traffic=losses_traffic,
                        loss_semantic=losses_seg,
                        loss_embeddings=losses_embeddings,
                        traffic_iou=traffic_iou_m,
                        semantic_iou=semantic_iou_m,
                        semantic_iou_valid=semantic_iou_m_valid,
                        semantic_iou_weighted=semantic_iou_m_weighted,
                    )

                )
                if writer:
                    writer.add_image(
                        "val/front_view" ,
                        retransform(input["rgb"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/left_view" ,
                        retransform(input["rgb_left"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/right_view" ,
                        retransform(input["rgb_right"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/front_center_view" ,
                        retransform(input["rgb_center"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/pred_traffic" ,
                        torch.clip(output[0][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0],
                        epoch,
                    )
                    writer.add_image(
                        "val/gt_traffic" ,
                        torch.clip(target[1][0], 0, 1).view(1, 20, 20, 7)[:, :, :, 0],
                        epoch,
                    )
                    writer.add_image(
                        "val/highres_gt_traffic" ,
                        torch.clip(target[0][0], 0, 1),
                        epoch,
                    )

                    semantic_pred_vis = semantic_pred[0] 
                    colors = torch.zeros((semantic_pred_vis.shape[0], semantic_pred_vis.shape[1], 3), dtype=torch.uint8, device=semantic_pred.device)
                    for class_idx in cityscapes_dict:
                        mask = (semantic_pred_vis == class_idx)
                        colors[mask] = torch.tensor(cityscapes_dict[str(max(0, int(class_idx)))]["color"], dtype=torch.uint8, device=semantic_pred.device)
                    colors = colors.permute(2, 0, 1)

                    semantic_target_vis = semantic_target[0]  
                    colors_gt = torch.zeros((semantic_target_vis.shape[0], semantic_target_vis.shape[1], 3), dtype=torch.uint8, device=semantic_target.device)
                    for class_idx in cityscapes_dict:
                        mask = (semantic_target_vis == class_idx) 
                        colors_gt[mask] = torch.tensor(cityscapes_dict[str(max(0, int(class_idx)))]["color"], dtype=torch.uint8, device=semantic_target.device)
                    colors_gt = colors_gt.permute(2, 0, 1)
                    
                    assert semantic_pred.ndimension() == 3, f"Expected 3D tensor (C, H, W), but got {semantic_pred.shape}"
                    
                    writer.add_image(
                        "val/seg_image",
                        colors,
                        epoch,
                        dataformats="CHW"
                    )

                    writer.add_image(
                        "val/gt_seg_view", 
                        colors_gt,
                        epoch,
                        dataformats="CHW"
                    )

                    if valid_mask.sum() > 0:  
                        batch_semantic_ious, batch_weighted_classes = semantic_iou(semantic_pred, semantic_target)
                        semantic_iou_m.update(batch_semantic_ious.mean().item(), batch_size)
                        semantic_iou_m_valid.update(batch_semantic_ious[batch_weighted_classes > 0].mean().item(), batch_size)
                        if batch_weighted_classes.sum() > 0:
                            semantic_iou_m_weighted.update((batch_semantic_ious * batch_weighted_classes).sum().item() / batch_weighted_classes.sum().item(), batch_size)
                        else: 
                            print("No valid classes found. Skipping weighted IoU calculation.")
                            print("Batch weighted classes: ", batch_weighted_classes)
                            print("Batch semantic ious: ", batch_semantic_ious)

                        for class_idx, iou in enumerate(batch_semantic_ious):
                            if class_idx in cityscapes_dict:  
                                writer.add_scalar(
                                    f"val/semantic_iou_class_{cityscapes_dict[str(max(0,int(class_idx)))]['tag']}", 
                                    iou.item(),
                                    epoch,
                                )
                
                    writer.add_image(
                        "val/pred_traffic_render" ,
                        torch.clip(
                            torch.tensor(
                                render(
                                    output[0][0].view(20, 20, 7).detach().cpu().numpy()
                                )[:100, 40:140]
                            ),
                            0,
                            255,
                        ).view(1, 100, 100),
                        epoch,
                    )
                    
                   

        if writer:
            writer.add_scalar("val/loss", losses_m.avg, epoch)
            writer.add_scalar("val/loss_traffic", losses_traffic.avg, epoch)
            writer.add_scalar("val/loss_seg", losses_seg.avg, epoch)
            writer.add_scalar("val/traffic_iou", traffic_iou_m.avg, epoch)
            writer.add_scalar("val/semantic_iou", semantic_iou_m.avg, epoch)
            writer.add_scalar("val/semantic_iou_valid", semantic_iou_m_valid.avg, epoch)
            writer.add_scalar("val/semantic_iou_weighted", semantic_iou_m_weighted.avg, epoch)
            writer.add_scalar("val/loss_traffic_light_state", losses_traffic_light_state.avg, epoch)


    metrics = OrderedDict([("loss", losses_m.avg), ("l1_error", l1_errorm.avg), ("traffic_iou", traffic_iou_m.avg), ("semantic_iou", semantic_iou_m_valid.avg)])

    return metrics


def infer_num_classes(normalized_tensor):
    unique_values = torch.unique(normalized_tensor)
    
    if len(unique_values) > 1:
        step_size = (unique_values[1:] - unique_values[:-1]).mean()

        num_classes = int(torch.round(1/step_size)) + 1

        max_val = unique_values.max()
        expected_max = (num_classes - 1) * step_size

        return num_classes
    else:
        return 1


if __name__ == "__main__":
    main()


