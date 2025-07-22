# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,              # The model to be trained
    criterion: torch.nn.Module,          # The loss function
    data_loader: Iterable,               # DataLoader providing training samples
    optimizer: torch.optim.Optimizer,    # Optimizer to update model parameters
    device: torch.device,                # Device (GPU/CPU) on which computations will be performed
    epoch: int,                          # Current epoch index
    loss_scaler,                         # A gradient scaling utility for mixed precision training
    max_norm: float = 0,                 # Maximum gradient norm for gradient clipping
    mixup_fn: Optional[Mixup] = None,    # Optional mixup function for data augmentation
    log_writer=None,                     # Optional writer for logging (e.g., TensorBoard)
    args=None                            # Additional arguments (e.g., number of epochs, etc.)
):
    # Set the model to training mode
    model.train(True)

    # Create a metric logger for tracking metrics during training
    metric_logger = misc.MetricLogger(delimiter="  ")
    
    # Add a learning rate meter to track the learning rate changes
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # Format the header to display the current epoch and total epochs
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # Frequency of printing training logs (iterations)
    print_freq = 20

    # Number of iterations to accumulate gradients before updating
    accum_iter = args.accum_iter

    # Initialize gradients to zero before training
    optimizer.zero_grad()

    # If a log writer (e.g., TensorBoard) is provided, print the log directory
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Loop over each batch of data in the provided DataLoader
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Adjust the learning rate per iteration rather than per epoch
        # This is helpful in large batch training strategies.
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move input samples and targets to the designated device
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup augmentation if provided
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Enable mixed precision autocast for forward pass to speed up training and save memory
        with torch.cuda.amp.autocast():
            outputs = model(samples)      # Forward pass: compute model outputs
            loss = criterion(outputs, targets)  # Compute the loss

        # Get the scalar loss value for logging
        loss_value = loss.item()

        # Check for NaN or infinite losses and stop training if found
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Scale down the loss for gradient accumulation
        loss /= accum_iter

        # Use the loss_scaler to handle gradient scaling and optional gradient clipping.
        # The 'update_grad' flag ensures that gradients are actually updated every 'accum_iter' steps.
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )

        # After the chosen accumulation steps, reset gradients
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Wait for all CUDA operations to finish
        torch.cuda.synchronize()

        # Update the metric logger with the current loss
        metric_logger.update(loss=loss_value)

        # Track the minimum and maximum learning rates across parameter groups
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # Update the metric logger with the current (max) learning rate
        metric_logger.update(lr=max_lr)

        # Reduce the loss value across all processes (in distributed training)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # Log the loss and learning rate at a finer scale (e.g., per iteration)
        # The 'epoch_1000x' scaling is used for better alignment in TensorBoard when batch sizes differ.
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # After finishing one epoch, synchronize metrics across processes (for distributed training)
    metric_logger.synchronize_between_processes()

    # Print averaged stats over the epoch
    print("Averaged stats:", metric_logger)

    # Return the averaged metrics as a dictionary
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    # Define the loss criterion for evaluation (CrossEntropyLoss for classification tasks)
    criterion = torch.nn.CrossEntropyLoss()

    # Create a metric logger for evaluation metrics
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch the model to evaluation mode
    model.eval()

    # Iterate over the test/validation DataLoader
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # Move images and targets to the correct device
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute model outputs in mixed precision mode
        with torch.cuda.amp.autocast():
            output = model(images)    # Forward pass
            loss = criterion(output, target)  # Compute loss

        # Compute top-1 and top-5 accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # Update metrics
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()

    # Print the averaged evaluation metrics
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # Return the averaged evaluation metrics as a dictionary
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
