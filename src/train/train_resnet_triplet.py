# src/train/train_resnet_triplet.py
"""
Train patch encoder with triplet loss for writer retrieval.

Key improvements over baseline:
- MPerClassSampler for balanced batch sampling (critical for triplet loss)
- Morphological data augmentation (erosion/dilation)
- Cosine learning rate schedule with warmup
- Gradient clipping for stability
- Support for larger batch sizes
"""
import argparse
from pathlib import Path
import math
import numpy as np
import logging
import sys
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim

from ..features.resnet_patch_extractor import create_resnet_patch_encoder
from .patch_dataset import PatchDataset
from .triplet_loss import BatchHardTripletLoss
from .samplers import MPerClassSampler
from .augmentation import get_train_augmentation
from ..sampling.adaptive_sampler import AdaptiveSamplingConfig
from ..utils.preprocessing import clear_dir


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train patch encoder with triplet loss')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with image_path,writer_id')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Optional root directory for image paths')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (larger is better for triplet mining)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--final-lr', type=float, default=1e-5,
                        help='Final learning rate for cosine schedule')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Triplet loss margin')
    parser.add_argument('--emb-dim', type=int, default=64,
                        help='Embedding dimension (64 matches reference)')
    parser.add_argument('--out-dim', type=int, default=None,
                        help='Alias for --emb-dim for compatibility')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (default: save-dir/logs)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sampler', type=str, default='adaptive',
                        choices=['dense', 'contour', 'adaptive', 'char'])
    parser.add_argument('--m-per-class', type=int, default=8,
                        help='Samples per class in each batch (for MPerClassSampler)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use morphological augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='Disable augmentation')
    parser.add_argument('--grad-clip', type=float, default=2.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--backbone', type=str, default='small',
                        choices=['small', 'small_deep', 'resnet18', 'resnet34'],
                        help='Backbone architecture (default: small)')
    parser.add_argument('--debug', action='store_true', 
                        help='Show debug logs in console (always saved to file)')

    return parser.parse_args()


def cosine_scheduler(base_lr, final_lr, epochs, niter_per_ep, warmup_epochs=0):
    """
    Cosine learning rate schedule with linear warmup.
    
    Returns a list of learning rates for each iteration.
    """
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep

    schedule = []
    for i in range(total_iters):
        if i < warmup_iters:
            # Linear warmup
            lr = base_lr * (i + 1) / warmup_iters
        else:
            # Cosine decay
            progress = (i - warmup_iters) / (total_iters - warmup_iters)
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))
        schedule.append(lr)

    return schedule


def main():
    args = parse_args()

    # Handle --out-dim alias
    if args.out_dim is not None:
        args.emb_dim = args.out_dim

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    clear_dir(save_dir)  # Clear previous checkpoints/logs

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info('=' * 70)
    logger.info('🚀 CARA-WR Training Pipeline Started')
    logger.info('=' * 70)
    logger.info('')
    logger.info('📋 Configuration:')
    for k, v in vars(args).items():
        logger.info(f'   {k:20s}: {v}')
    logger.info('')
    logger.info(f'💻 Device: {device}')
    if torch.cuda.is_available():
        logger.info(f'   GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    logger.info('=' * 70)

    sampler_map = {
        'dense': 'dense',
        'contour': 'contour',
        'char': 'char',
        'adaptive': 'auto',
    }

    sampler_cfg = AdaptiveSamplingConfig(
        patch_size=32,
        dense_stride=24,
        contour_step=12,
        max_patches=1500,
        line_threshold=4,
        mode=sampler_map[args.sampler],
    )

    # Get augmentation transform if enabled
    transform = get_train_augmentation() if args.augment else None
    if args.augment:
        logger.info('✅ Morphological augmentation ENABLED')
    else:
        logger.info('⚠️  Morphological augmentation DISABLED')

    logger.info('')
    logger.info('📂 Loading dataset...')
    start_time = time.time()

    # Each image yields 1 random patch per __getitem__ call
    # With 189 training images and multiple epochs, we get diversity through:
    # 1. Random patch selection per access (different each epoch)
    # 2. Data augmentation (morphological transforms)

    dataset = PatchDataset(
        csv_path=args.csv,
        root_dir=args.root_dir,
        sampler_cfg=sampler_cfg,
        max_side=1600,
        transform=transform,
        min_patches=10,
    )

    load_time = time.time() - start_time
    n_images = len(dataset)
    logger.info(f'   ✓ Loaded {n_images} images in {load_time:.1f}s')

    # Get labels for MPerClassSampler
    all_labels = [writer_id for _, writer_id in dataset.samples]

    # train/val split by writer (not random)
    unique_writers = list(set(all_labels))
    np.random.shuffle(unique_writers)

    val_count = max(1, int(len(unique_writers) * args.val_split))
    val_writers = set(unique_writers[:val_count])
    train_writers = set(unique_writers[val_count:])

    # Map indices to train/val based on writer
    train_indices = []
    val_indices = []
    for i, (_, writer_id) in enumerate(dataset.samples):
        if writer_id in train_writers:
            train_indices.append(i)
        else:
            val_indices.append(i)

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_labels = [all_labels[i] for i in train_indices]

    logger.info('')
    logger.info('📊 Dataset Split:')
    logger.info(f'   Train: {len(train_ds)} images from {len(train_writers)} writers')
    logger.info(f'   Val:   {len(val_ds)} images from {len(val_writers)} writers')
    logger.info(f'   Images per writer: ~{len(train_ds) / len(train_writers):.1f}')

    # Use MPerClassSampler for balanced batches
    train_sampler = MPerClassSampler(
        labels=train_labels,
        m=args.m_per_class,
        batch_size=args.batch_size,
        length_before_new_iter=len(train_ds),  # One pass through all samples
    )
    logger.info(f'   ✓ Using MPerClassSampler (m={args.m_per_class} samples/class)')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f'   Batches per epoch: {len(train_loader)} train, {len(val_loader)} val')

    logger.info('')
    logger.info('🔧 Building model...')
    model = create_resnet_patch_encoder(emb_dim=args.emb_dim, backbone=args.backbone).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'   ✓ {args.backbone} encoder created')
    logger.info(f'   Backbone: {args.backbone}')
    logger.info(f'   Embedding dimension: {args.emb_dim}')
    logger.info(f'   Trainable parameters: {num_params:,}')

    criterion = BatchHardTripletLoss(margin=args.margin)
    logger.info(f'   Loss: BatchHardTripletLoss (margin={args.margin})')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    logger.info(f'   Optimizer: Adam (lr={args.lr:.0e})')

    # Cosine schedule with warmup
    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(
        args.lr, args.final_lr, args.epochs, niter_per_ep, args.warmup_epochs
    )
    logger.info(f'   LR Schedule: Cosine annealing ({args.lr:.0e} → {args.final_lr:.0e})')
    logger.info(f'   Warmup: {args.warmup_epochs} epochs')
    logger.info(f'   Gradient clipping: max_norm={args.grad_clip}')

    best_val_loss = float('inf')
    best_epoch = 0
    global_iter = 0

    logger.info('')
    logger.info('=' * 70)
    logger.info('🏋️  Starting Training')
    logger.info('=' * 70)

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        logger.info('')
        logger.info(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        logger.info(f'📅 Epoch {epoch}/{args.epochs}')
        logger.info(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

        train_loss, global_iter = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            lr_schedule, global_iter, args.grad_clip, epoch
        )

        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        logger.info('')
        logger.info(f'📈 Epoch {epoch} Summary:')
        logger.info(f'   Train Loss: {train_loss:.4f}')
        logger.info(f'   Val Loss:   {val_loss:.4f}')
        logger.info(f'   Learning Rate: {current_lr:.2e}')
        logger.info(f'   Time: {epoch_time:.1f}s')

        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt_path = save_dir / f'best_epoch_{epoch}_loss_{val_loss:.4f}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, ckpt_path)
            logger.info(f'   ✅ NEW BEST! Improved by {improvement:.4f}')
            logger.info(f'   💾 Saved: {ckpt_path.name}')
        else:
            epochs_since_best = epoch - best_epoch
            logger.info(f'   ⏳ No improvement for {epochs_since_best} epoch(s)')

        # Early stopping
        if epoch - best_epoch > args.patience:
            logger.info('')
            logger.info(f'⚠️  Early stopping triggered (no improvement for {args.patience} epochs)')
            break

    total_time = time.time() - training_start
    logger.info('')
    logger.info('=' * 70)
    logger.info('🎉 Training Complete!')
    logger.info('=' * 70)
    logger.info(f'   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})')
    logger.info(f'   Total training time: {total_time / 60:.1f} minutes')
    logger.info(f'   Checkpoints saved to: {save_dir}')
    logger.info('=' * 70)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    lr_schedule, global_iter, grad_clip, epoch):
    """Train for one epoch with LR scheduling and gradient clipping."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    total_batches = len(loader)

    log_interval = max(1, total_batches // 10)  # Log ~10 times per epoch

    for batch_idx, (patches, labels) in enumerate(loader):
        # Update learning rate
        if global_iter < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[global_iter]

        # Handle multi-patch batches: [B, P, C, H, W] -> [B*P, C, H, W]
        # Labels from PatchDataset are already [B, P], just flatten them
        if patches.dim() == 5:
            B, P, C, H, W = patches.shape
            patches = patches.view(B * P, C, H, W)
            labels = labels.view(B * P)  # [B, P] -> [B*P]

        patches = patches.to(device)  # [B*P, 1, H, W] or [B, 1, H, W]
        labels = labels.to(device)  # [B*P] or [B]

        optimizer.zero_grad()
        emb = model(patches)  # [B*P, D] or [B, D]
        loss = criterion(emb, labels)
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = 0.0

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        global_iter += 1

        # Progress logging
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            progress = 100 * (batch_idx + 1) / total_batches
            lr = optimizer.param_groups[0]['lr']
            avg_loss = running_loss / num_batches
            logger.info(f'   [{progress:5.1f}%] Batch {batch_idx + 1:4d}/{total_batches} | '
                        f'Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | '
                        f'LR: {lr:.2e}')

    return running_loss / max(1, num_batches), global_iter


def validate(model, loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    running_loss = 0.0
    num_batches = 0

    logger.info('   🔍 Running validation...')

    with torch.no_grad():
        for patches, labels in loader:
            # Handle multi-patch batches: [B, P, C, H, W] -> [B*P, C, H, W]
            # Labels from PatchDataset are already [B, P], just flatten them
            if patches.dim() == 5:
                B, P, C, H, W = patches.shape
                patches = patches.view(B * P, C, H, W)
                labels = labels.view(B * P)  # [B, P] -> [B*P]

            patches = patches.to(device)
            labels = labels.to(device)
            emb = model(patches)
            loss = criterion(emb, labels)
            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    logger.info(f'   ✓ Validation complete ({num_batches} batches)')
    return avg_loss


if __name__ == '__main__':
    main()
