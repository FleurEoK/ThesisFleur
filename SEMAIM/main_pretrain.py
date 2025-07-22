# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import timm.optim.optim_factory as optim_factory
from timm.utils import ModelEma
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, ConcatDataset

from models import models_semaim as models_aim
from engines.engine_pretrain import train_one_epoch
from torchvision.datasets import CIFAR10
# from datasets.datasets import ImagenetLoader
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


class FolderPermutationDataset(Dataset):
    """Dataset that loads all images from a folder containing image permutations"""
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # Get all image files from the folder
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {folder_path}")
            
        print(f"Found {len(self.image_files)} image files in {folder_path}")
        for i, img_path in enumerate(self.image_files):
            print(f"  {i+1}: {os.path.basename(img_path)}")
        
    def _get_image_files(self):
        """Get all image files from the folder"""
        import glob
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.folder_path, ext)))
            # image_files.extend(glob.glob(os.path.join(self.folder_path, ext.upper())))
        
        # Sort files for consistent ordering
        image_files.sort()
        return image_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        # Load the specific image file
        image_path = self.image_files[idx]
        try:
            print(f"Loading image {idx}: {os.path.basename(image_path)}")
            image = Image.open(image_path).convert('RGB')
            print(f"  PIL image size: {image.size}")
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
                print(f"  After transform shape: {image.shape}")
                print(f"  After transform type: {type(image)}")
            
            # Ensure image is a 3D tensor (C, H, W)
            if isinstance(image, torch.Tensor) and len(image.shape) == 3:
                return image
            else:
                raise ValueError(f"Image {image_path} has unexpected shape: {image.shape if hasattr(image, 'shape') else type(image)}")
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise


def get_args_parser():
    parser = argparse.ArgumentParser('SemAIM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150, type=int)  # Reduced epochs for small dataset
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='aim_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,  # Changed to 224 for better compatibility
                        help='images input size')

    parser.add_argument('--query_depth', default=12, type=int,
                        help='decoder depth')
    parser.add_argument('--share_weight', action='store_true',
                        help='Share weight between encoder and decoder')

    parser.add_argument('--prediction_head_type', default='MLP', type=str,
                        help='the type of prediction head: MLP or LINEAR')
    parser.add_argument('--gaussian_kernel_size', default=None, type=int,
                        help='Use gaussian blur to smooth the target image')
    parser.add_argument('--gaussian_sigma', default=None, type=int,
                        help='standard deviation of gaussian blur')
    parser.add_argument('--loss_type', default='L2', type=str,
                        help='Calculate loss between prediction and target per pixel: L1 or L2')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # semaim
    parser.add_argument('--permutation_type', default='raster', type=str,
                        help='Permutation type for autoregression: zigzag, raster, stochastic, center2out, out2center, saliency,'
                        ' attention, attention_guided, saliency_guided, stochastic_center, attention_center')
    parser.add_argument('--use_ema_model', action='store_true', help='Use ema features as targets for computing loss')
    parser.set_defaults(use_ema_model=False)
    parser.add_argument('--predict_feature', default='none', type=str, help='Use features as targets: none, inference, ema, dino, clip')
    parser.add_argument('--attention_type', default='cls', type=str, help='Attention type: gap, cls and self')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.01,  # Reduced weight decay
                        help='weight decay (default: 0.01)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-6, metavar='LR',  # Reduced base learning rate
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',  # Reduced warmup epochs
                        help='epochs to warmup LR')
    parser.add_argument('--not_use_fp16', action='store_true', help='whether to use fp16')
    parser.set_defaults(not_use_fp16=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/cifar_alt', type=str, help='dataset path')
    parser.add_argument('--permutation_folder', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/SemAIM/hutsel/random_cat_images/', type=str, help='path to folder containing image permutations')

    parser.add_argument('--output_dir', default='./pretrain/aim_base',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--experiment', default='exp_folder_cats', type=str, help='experiment name (for log)')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)  # Reduced workers for small dataset
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

# main training function
def main(args):
    # initialize distributed training if enabled
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cpu")

    # fix the random seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Define standard transforms for the permutation images
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset from folder containing permutation images
    dataset_train = FolderPermutationDataset(
        folder_path=args.permutation_folder,
        transform=transform_train
    )
    
    print(f"Loaded {len(dataset_train)} image permutations from {args.permutation_folder}")
    
    # Test the dataset to ensure proper tensor shapes
    print("\n=== Testing dataset ===")
    try:
        test_sample = dataset_train[0]
        print(f"Sample tensor shape: {test_sample.shape}")
        print(f"Sample tensor dtype: {test_sample.dtype}")
        print(f"Sample tensor min/max: {test_sample.min():.3f}/{test_sample.max():.3f}")
        
        # Test batch loading
        print("\n=== Testing batch loading ===")
        test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=False)
        test_batch = next(iter(test_loader))
        print(f"Batch tensor shape: {test_batch.shape}")
        print(f"Expected shape: [batch_size, channels, height, width] = [2, 3, {args.input_size}, {args.input_size}]")
        
        if test_batch.shape != (2, 3, args.input_size, args.input_size):
            raise ValueError(f"Batch shape mismatch! Got {test_batch.shape}, expected (2, 3, {args.input_size}, {args.input_size})")
        
        print("Dataset and batching working correctly!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        raise

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)  # Use RandomSampler for better shuffling

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop last batch for small dataset
        collate_fn=None,  # Use default collate function
    )
    
    # define the model
    out_dim = 512
    model = models_aim.__dict__[args.model](permutation_type=args.permutation_type,attention_type=args.attention_type,
                                             query_depth=args.query_depth, share_weight=args.share_weight,out_dim=out_dim,
                                             prediction_head_type=args.prediction_head_type,
                                             gaussian_kernel_size=args.gaussian_kernel_size,
                                             gaussian_sigma=args.gaussian_sigma,
                                             loss_type=args.loss_type, predict_feature=args.predict_feature,
                                             norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    if misc.is_main_process():
        print("Model = %s" % str(model_without_ddp))

    # define ema (Exponential Moving Average) model
    model_ema = None
    teacher_model = None
    if args.use_ema_model:
        model_ema = ModelEma(model, decay=0.999, device=args.device, resume='')
    elif args.predict_feature == 'dino':
        teacher_model = timm.models.vit_base_patch16_224(num_classes=0)
        state_dict = torch.load('/path_to_dino_model/dino_vitbase16_pretrain.pth')
        msg = teacher_model.load_state_dict(state_dict, strict=False)
        print("loaded dino model with msg:", msg)
        teacher_model.to(device)
        teacher_model.eval()
    elif args.predict_feature == 'clip':
        from models.models_clip import build_model
        state_dict = torch.load('/path_to_clip_model/clip_vitbase16_pretrain.pth', map_location='cpu')
        model_clip = build_model(state_dict)
        msg = model_clip.load_state_dict(state_dict, strict=False)
        print("loaded clip model with msg:", msg)
        model_clip.float()
        teacher_model = model_clip.visual
        teacher_model.to(device)
        teacher_model.eval()

    # calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.cpu], 
        find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # prepare optimizer
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # define the mixed precision scaler
    if args.not_use_fp16:
        loss_scaler = None
    else:
        loss_scaler = NativeScaler()
    
    # load checkpoint if specified
    ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        model_ema_state_dict = model_ema.ema if args.use_ema_model else None
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, model_ema=model_ema_state_dict,
            optimizer=optimizer, loss_scaler=loss_scaler)

    # initialize the log writer
    if misc.get_rank() == 0:
        log_dir = os.path.join(args.log_dir, f"{args.model}.{args.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    print(f"Training on {len(dataset_train)} image permutations from folder")
    start_time = time.time()
    
    # Debug: Test one training step before starting the loop
    print("\n=== Testing one training step ===")
    model.train()
    data_iter = iter(data_loader_train)
    try:
        batch = next(data_iter)
        print(f"Training batch shape: {batch.shape}")
        print(f"Training batch dtype: {batch.dtype}")
        print(f"Training batch device: {batch.device}")
        
        # Move batch to device
        batch = batch.to(device)
        print(f"After moving to device: {batch.shape}")
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            output = model(batch)
            print(f"Model output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Model output tuple length: {len(output)}")
                for i, item in enumerate(output):
                    if hasattr(item, 'shape'):
                        print(f"  Output {i} shape: {item.shape}")
                    else:
                        print(f"  Output {i} type: {type(item)}")
        
        print("✓ Forward pass successful!")
        
    except Exception as e:
        print(f"Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, args.clip_grad,
            log_writer=log_writer,
            args=args, model_ema=model_ema, teacher_model=teacher_model,
        )

        # save a temporary checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()
        if model_ema is not None:
            save_dict['ema_state_dict'] = model_ema.ema.state_dict()

        #save periodic checkpoints
        ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
        misc.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if args.output_dir and ((epoch + 1) % args.saveckp_freq == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir, "{}.{}.{:04d}.pth".format(args.model, args.experiment, epoch+1))
            misc.save_on_master(save_dict, ckpt_path)

        # log stats for the epoch
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir,"{}.{}.log.txt".format(args.model,args.experiment)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # training complete 
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)