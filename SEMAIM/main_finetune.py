# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import builtins

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from datasets.datasets import ImageListFolder, build_transform
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_vit
from engines.engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    """
    Parse command-line arguments for fine-tuning UM-MAE models on image classification.
    Returns an argparse parser object.
    """
    parser = argparse.ArgumentParser('UM-MAE fine-tuning for image classification', add_help=False)
    # General training parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per-GPU batch size for training.')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs for fine-tuning.')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradients for this many iterations to effectively increase batch size.')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of the ViT-based model variant to use.')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input image size (height and width).')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate for stochastic depth.')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm; None disables clipping.')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (L2 regularization) factor.')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate. If None, determined from base LR and batch size.')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate; actual LR scales with batch size.')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='Layer-wise learning rate decay factor.')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='Lower bound for the learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Number of warmup epochs for LR scheduling.')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor.')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='AutoAugment policy to use.')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing for training.')

    # Random Erase parameters
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Probability of random erasing patches during training.')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Mode for random erase augmentation.')
    parser.add_argument('--recount', type=int, default=1,
                        help='Number of times to apply random erase.')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Whether to not random erase the first clean augmentation split.')

    # Mixup/ Cutmix parameters
    parser.add_argument('--mixup', type=float, default=0,
                        help='Mixup alpha. Enables Mixup if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='CutMix alpha. Enables CutMix if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='Cutmix min/max ratio. Overrides alpha and enables cutmix if set.')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix.')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix are enabled.')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix: "batch", "pair", or "elem".')

    # Finetuning parameters
    parser.add_argument('--finetune', default='',
                        help='Path to a pre-trained checkpoint to finetune from.')
    parser.add_argument('--global_pool', action='store_true',
                        help='Use global average pooling for classification head.')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token for classification instead of global pool.')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='Path to dataset.')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='Number of classes in the dataset.')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory for output files (checkpoints, logs).')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='Directory for tensorboard logs.')
    parser.add_argument('--saveckp_freq', default=20, type=int,
                        help='Frequency (in epochs) to save checkpoints.')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu) for training.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--resume', default='',
                        help='Path to resume checkpoint from.')
    parser.add_argument('--experiment', default='exp', type=str,
                        help='Experiment name, used in logging.')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch for training.')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model without training.')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enable distributed evaluation.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of CPU workers for data loading.')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin memory in DataLoader for performance.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes (GPUs).')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank of this process; used internally by torch.distributed.')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Flag for distributed training on internal cluster.')
    parser.add_argument('--dist_url', default='env://',
                        help='URL to set up distributed training.')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='Backend for distributed training.')

    return parser


def main(args):
    # Initialize distributed mode if applicable
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set random seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True  # Enable cudnn auto-tuner

    # Build transformations for training and validation sets
    transform_train = build_transform(is_train=True, args=args)
    transform_val = build_transform(is_train=False, args=args)

    # Build datasets and data loaders
    dataset_train = ImageListFolder(
        os.path.join(args.data_path, 'train'),
        transform=transform_train,
        ann_file=os.path.join(args.data_path, 'train.txt')
    )
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    dataset_val = ImageListFolder(
        os.path.join(args.data_path, 'train'),
        transform=transform_val,
        ann_file=os.path.join(args.data_path, 'train.txt')
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print("Sampler_val = %s" % str(sampler_val))

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    # Setup Mixup/ CutMix if specified
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes
        )

    # Create model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load a pre-trained checkpoint for fine-tuning if specified
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'state_dict' in checkpoint:
            checkpoint_model = checkpoint['state_dict']
        else:
            checkpoint_model = checkpoint['model']

        state_dict = model.state_dict()
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}

        # Remove head weights if shape mismatch
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Interpolate position embedding if needed
        interpolate_pos_embed(model, checkpoint_model)

        # Load weights
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        print("global_pool = ", args.global_pool)

        # Check missing keys for global pooling variant
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # Initialize the linear head
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model_without_ddp = model

    # Count number of trainable parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # Compute effective batch size (across all GPUs)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # If LR is not provided, compute it from base LR and batch size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Convert model for distributed training if necessary
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Build optimizer with layer-wise lr decay
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # Choose appropriate loss function
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # Resume from a temporary checkpoint if it exists
    ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not found in {}, training from scratch".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler)

    # If eval flag is set, just evaluate and exit
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    # Setup TensorBoard log writer on main process
    if misc.get_rank() == 0 and args.log_dir is not None and not args.eval:
        log_dir = os.path.join(args.log_dir, f"{args.model}.{args.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        # Set epoch for distributed sampler to ensure different shuffling each epoch
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        # Save temporary checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()

        ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
        misc.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        # Evaluate model on validation set
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Pretrained from: {args.finetune}")
        print(f"Accuracy on {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy so far: {max_accuracy:.2f}%')

        if log_writer is not None:
            # Log validation statistics to TensorBoard
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        # Log all stats (train and test) to a log file
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, f"{args.model}.{args.experiment}.log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # Silence printing in all processes except the main one
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # Parse arguments and create output directory if needed
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
