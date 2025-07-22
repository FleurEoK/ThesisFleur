# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torchvision
import cv2
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched


def generate_saliency(saliency_model, imgs):
    """
    Generate a saliency map from the given saliency model and input images.

    Returns:
        pred (Tensor): A normalized, downscaled saliency map (B, width*width).
        saliency (Tensor): The original saliency map with shape (B, 1, 224, 224).
    """
    patch_size = 16
    width = imgs.shape[2] // patch_size  # width in terms of patch grids
    with torch.no_grad():
        # The saliency model returns multiple feature maps, we only take the first (d1)
        d1, _, _, _, _, _, _, _ = saliency_model(imgs)
        saliency = d1[:, 0, :, :]  # Extract the first channel as saliency

        # Resize saliency map to match patch grid using bilinear interpolation
        pred = torch.nn.functional.interpolate(
            saliency.unsqueeze(dim=1),
            (width, width),
            mode='bilinear',
            align_corners=True
        )

        # Flatten the resized saliency maps to (B, width*width)
        N, _, _, _ = pred.shape
        pred = pred.reshape(N, -1)

        # Normalize the saliency values between 0 and 1
        mx, mn = torch.max(pred, dim=-1, keepdim=True)[0], torch.min(pred, dim=-1, keepdim=True)[0]
        pred = (pred - mn) / (mx - mn + 1e-5)

    return pred, saliency.unsqueeze(dim=1)


def forward_teacher_features(model, x, model_type):
    """
    Forward pass using a teacher model (DINO or CLIP) to get token embeddings.

    Returns:
        Tensor: Features from the teacher model.
    """
    assert model_type in ['dino', 'clip']
    if model_type == 'dino':
        return forward_features_dino(model, x)
    else:
        return forward_features_clip(model, x)


def forward_features_dino(model, x):
    """
    Forward pass through a DINO model to extract features.

    Returns:
        Tensor: The output tokens (including class token).
    """
    B = x.shape[0]
    # Embed patches
    x = model.patch_embed(x)

    # Add class token
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Add positional embeddings
    x = x + model.pos_embed
    x = model.pos_drop(x)

    # Pass through transformer blocks
    for blk in model.blocks:
        x = blk(x)

    x = model.norm(x)
    # return x[:, 1:, :]
    return x


def forward_features_clip(model, x):
    """
    Forward pass through a CLIP model (ViT-based) to extract features.

    Returns:
        Tensor: The output tokens after the transformer, including class token.
    """
    # Initial convolutional embedding
    x = model.conv1(x)  # [B, width, grid, grid]

    # Flatten spatial dimensions
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
    x = x.permute(0, 2, 1)  # [B, grid**2, width]

    # Add class token
    x = torch.cat(
        [
            model.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ],
        dim=1
    )  # [B, grid**2+1, width]

    # Add positional embeddings
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    # Transformer forward pass
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # Final layer norm
    # x = model.ln_post(x[:, 0, :])
    x = model.ln_post(x)

    # Optional projection if available
    if model.proj is not None:
        x = x @ model.proj

    # return x[:, 1:, :]
    return x


def calculate_similarity(tokens):
    """
    Calculate similarity scores between the class token and all other tokens.

    Returns:
        Tensor: Similarity scores normalized between 0 and 1, shape [B, L-1].
    """
    # L2 normalize token features
    tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)

    # Compute dot product similarity between class token (index 0) and others
    similarity = torch.sum(tokens[:, 0, :].unsqueeze(1) * tokens[:, 1:, :], dim=-1)

    # Normalize similarity values between 0 and 1
    mx, mn = torch.max(similarity, dim=1, keepdim=True)[0], torch.min(similarity, dim=1, keepdim=True)[0]
    similarity = (similarity - mn) / (mx - mn + 1e-6)

    return similarity


def applyColorMap_on_tensor(tensor, images, alpha=0.3, norm=False, inverse=False):
    """
    Apply a color map to a given saliency tensor and blend it with the original images.

    Returns:
        Tensor: The heatmapped images (B, C, H, W).
    """
    heat_map = []
    tensor = tensor.cpu()
    # Apply color map to each tensor in the batch
    for i in range(tensor.shape[0]):
        temp_map = tensor[i]
        if norm:
            temp_map = (temp_map - temp_map.min()) / (temp_map.max() - temp_map.min() + 1e-5)
        if inverse:
            temp_map = 1 - temp_map
        temp_map = np.uint8(255 * temp_map)
        temp_map = cv2.applyColorMap(temp_map, cv2.COLORMAP_JET)  # Apply color map
        heat_map.append(temp_map)

    # Convert to Tensor and blend with original images
    heat_map = torch.Tensor(np.array(heat_map)).cuda().permute(0, 3, 1, 2)
    heat_map = torch.clip(heat_map * alpha + images * (1 - alpha), 0, 255)
    return heat_map


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm=None,
                    log_writer=None,
                    args=None,
                    model_ema=None,
                    teacher_model=None):
    """
    Train the model for one epoch.

    Returns:
        dict: A dictionary of averaged metrics over the epoch.
    """
    # Set model to training mode
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 20
    accum_iter = args.accum_iter

    # Adjust EMA decay if using EMA model
    if args.use_ema_model:
        assert model_ema is not None
        if epoch < 100:
            model_ema.decay = 0.999 + epoch / 100 * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

    # Initialize gradients to zero
    optimizer.zero_grad()

    # Print log directory if log writer is provided
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Iterate over batches (token_order added here)
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step

        # Adjust learning rate per iteration
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # token_orders = token_orders.to(device, non_blocking=True) if token_orders is not None else None


        # Initialize variables for attention maps and enc_tokens
        enc_tokens, attention = None, None
        feature_attention, self_attention = None, None

        # Generate attention tokens if certain conditions are met
        if args.predict_feature == 'none' and 'attention' in args.permutation_type:
            model.eval()
            with torch.no_grad():
                if args.use_ema_model:
                    # Use EMA model for forward pass
                    _, feature_attention, self_attention = model_ema.ema(samples, forward_encoder=True)
                else:
                    # Use teacher model for forward pass
                    enc_tokens, feature_attention, self_attention = model(samples, forward_encoder=True)
            model.train()
            feature_attention, self_attention = feature_attention.detach(), self_attention.detach()
            attention = self_attention if args.attention_type == 'self' else feature_attention

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(loss_scaler is not None):
            # Depending on args, we generate enc_tokens or attention maps using teacher models or EMA
            if args.predict_feature == 'inference':
                model.eval()
                # Forward pass the model to get enc_tokens and attention maps while disabling gradients
                with torch.no_grad():
                    enc_tokens, feature_attention, self_attention = model(samples, forward_encoder=True)
                    enc_tokens = enc_tokens.detach()
                    feature_attention, self_attention = feature_attention.detach(), self_attention.detach()
                model.train()
                # Select the appropriate attention map based on the attention type
                attention = self_attention if args.attention_type == 'self' else feature_attention
            # Use EMA model for forward pass
            elif args.predict_feature == 'ema':
                if enc_tokens is None:
                    with torch.no_grad():
                        enc_tokens, _, _ = model_ema.ema(samples, forward_encoder=True)
                    enc_tokens = enc_tokens.detach()
                attention = self_attention if args.attention_type == 'self' else feature_attention
            # Use teacher model for forward pass
            elif args.predict_feature == 'dino':
                with torch.no_grad():
                    enc_tokens = forward_teacher_features(teacher_model, samples, 'dino')
                enc_tokens = enc_tokens.detach()
                attention = calculate_similarity(enc_tokens)
                feature_attention = attention
            # Use teacher model for forward pass
            elif args.predict_feature == 'clip':
                with torch.no_grad():
                    enc_tokens = forward_teacher_features(teacher_model, samples, 'clip')
                enc_tokens = enc_tokens.detach()
                attention = calculate_similarity(enc_tokens)
                feature_attention = attention

            # Forward pass the model with the appropriate targets (enc_tokens / attention)
            loss, permutation, loss_map = model(samples, enc_tokens, attention)

        loss_value = loss.item()

        # Check for invalid loss values
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Gradient accumulation
        loss /= accum_iter

        # Update gradients
        if loss_scaler is None:
            # Standard FP32 training
            # Backward pass
            loss.backward()
            if (data_iter_step + 1) % accum_iter == 0:
                norm = 0
                # Gradient clipping
                if max_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            else:
                norm = None
        else:
            # FP16 / AMP training with gradient scaling
            # Use the loss_scaler to handle gradient scaling and optional gradient clipping.
            norm = loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                clip_grad=max_norm,
                update_grad=(data_iter_step + 1) % accum_iter == 0
            )
            fp16_scaler = loss_scaler._scaler.get_scale()

        # Reset gradients if we have completed an accumulation cycle
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        # Synchronize CUDA operations
        if torch.cuda.is_available() and torch.device.type == 'cuda':
            torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_value, total_norm=norm)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Reduce loss for logging (in multi-GPU scenario)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # Logging to tensorboard or other loggers
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('loss', loss_value_reduce, it)
            log_writer.add_scalar('lr', lr, it)
            log_writer.add_scalar('grad_norm', norm, it)
            if loss_scaler is not None:
                log_writer.add_scalar('fp16_scaler', fp16_scaler, it)

    # Synchronize metrics across processes (for distributed training)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Return averaged metrics as a dictionary
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
