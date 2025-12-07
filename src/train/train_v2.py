# src/train/train_v2.py
"""
Improved training pipeline for writer retrieval.

Key improvements over train_resnet_triplet.py:
- Multi-patch per image (patches_per_call parameter)
- Multiple loss function options (triplet, semi-hard, ms, circle)
- Pretrained backbone support (ResNet-18/34)
- Dataset-specific configuration support
- Better handling of flattened batch tensors
- End-to-end aggregation training option (future)
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
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from ..features.resnet_patch_extractor import create_resnet_patch_encoder
from .patch_dataset import PatchDataset
from .triplet_loss import get_loss_function
from .samplers import MPerClassSampler
from .augmentation import get_train_augmentation
from ..sampling.adaptive_sampler import AdaptiveSamplingConfig
from ..utils.preprocessing import clear_dir


def setup_logging(save_dir: Path = None, log_level: int = logging.DEBUG):
    """Setup logging configuration for training."""
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if save_dir:
        log_file = save_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train writer retrieval model (v2)')
    
    # Data
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with image_path,writer_id')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Optional root directory for image paths')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides other args)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['small', 'small_deep', 'resnet18', 'resnet34'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone (for resnet18/34)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--freeze-bn', action='store_true', default=False,
                        help='Freeze BatchNorm layers')
    parser.add_argument('--emb-dim', type=int, default=128,
                        help='Embedding dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--final-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--grad-clip', type=float, default=2.0)
    parser.add_argument('--patience', type=int, default=15)
    
    # Loss
    parser.add_argument('--loss', type=str, default='triplet',
                        choices=['triplet', 'semi_hard', 'soft_margin', 'ms', 'circle'],
                        help='Loss function type')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Triplet loss margin')
    
    # Sampling
    parser.add_argument('--m-per-class', type=int, default=4,
                        help='Samples per class in each batch')
    parser.add_argument('--patches-per-call', type=int, default=8,
                        help='Number of patches returned per image access')
    parser.add_argument('--sampler', type=str, default='adaptive',
                        choices=['dense', 'char', 'adaptive'])
    
    # Augmentation
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--strong-augment', action='store_true', default=False)
    
    # Validation
    parser.add_argument('--val-split', type=float, default=0.1)
    
    # Output
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def cosine_scheduler(base_lr, final_lr, epochs, niter_per_ep, warmup_epochs=0):
    """Cosine learning rate schedule with linear warmup."""
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep
    
    schedule = []
    for i in range(total_iters):
        if i < warmup_iters:
            lr = base_lr * (i + 1) / warmup_iters
        else:
            progress = (i - warmup_iters) / (total_iters - warmup_iters)
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))
        schedule.append(lr)
    
    return schedule


def flatten_batch(patches, labels):
    """
    Flatten multi-patch batch to single-patch batch.
    
    Input:
        patches: [B, P, C, H, W] where P = patches_per_call
        labels: [B, P] (repeated writer_id)
    
    Output:
        patches: [B*P, C, H, W]
        labels: [B*P]
    """
    B, P = patches.shape[0], patches.shape[1]
    patches = patches.view(B * P, *patches.shape[2:])
    labels = labels.view(B * P)
    return patches, labels


def train_one_epoch(model, loader, criterion, optimizer, device,
                    lr_schedule, global_iter, grad_clip, epoch, patches_per_call):
    """Train for one epoch with multi-patch support."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    total_batches = len(loader)
    
    log_interval = max(1, total_batches // 5)

    for batch_idx, (patches, labels) in enumerate(loader):
        # Update learning rate
        if global_iter < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[global_iter]
        
        # Handle multi-patch batches
        if patches.dim() == 5:  # [B, P, C, H, W]
            patches, labels = flatten_batch(patches, labels)
        
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        emb = model(patches)
        loss = criterion(emb, labels)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        global_iter += 1

        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            progress = 100 * (batch_idx + 1) / total_batches
            lr = optimizer.param_groups[0]['lr']
            avg_loss = running_loss / num_batches
            logger.info(f'   [{progress:5.1f}%] Batch {batch_idx+1:4d}/{total_batches} | '
                       f'Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | '
                       f'LR: {lr:.2e}')

    return running_loss / max(1, num_batches), global_iter


def validate(model, loader, criterion, device, patches_per_call):
    """Validate model on validation set."""
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for patches, labels in loader:
            if patches.dim() == 5:
                patches, labels = flatten_batch(patches, labels)
            
            patches = patches.to(device)
            labels = labels.to(device)
            emb = model(patches)
            loss = criterion(emb, labels)
            running_loss += loss.item()
            num_batches += 1

    return running_loss / max(1, num_batches)


def main():
    args = parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    clear_dir(save_dir)
    
    global logger
    logger = setup_logging(save_dir)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info('=' * 70)
    logger.info('🚀 CARA-WR Training Pipeline v2')
    logger.info('=' * 70)
    logger.info('')
    logger.info('📋 Configuration:')
    for k, v in vars(args).items():
        logger.info(f'   {k:20s}: {v}')
    logger.info('')
    logger.info(f'💻 Device: {device}')
    if torch.cuda.is_available():
        logger.info(f'   GPU: {torch.cuda.get_device_name(0)}')
    logger.info('=' * 70)

    # Sampler config
    sampler_map = {'dense': 'dense', 'char': 'char', 'adaptive': 'auto'}
    sampler_cfg = AdaptiveSamplingConfig(
        patch_size=32,
        dense_stride=24,
        max_patches=1500,
        mode=sampler_map[args.sampler],
    )

    # Augmentation
    transform = get_train_augmentation(strong=args.strong_augment) if args.augment else None
    
    logger.info('')
    logger.info('📂 Loading dataset...')
    start_time = time.time()
    
    dataset = PatchDataset(
        csv_path=args.csv,
        root_dir=args.root_dir,
        sampler_cfg=sampler_cfg,
        max_side=1600,
        transform=transform,
        min_patches=10,
        patches_per_call=args.patches_per_call,
        cache_patches=False,
    )
    
    load_time = time.time() - start_time
    logger.info(f'   ✓ Loaded {len(dataset)} images in {load_time:.1f}s')
    logger.info(f'   ✓ Patches per call: {args.patches_per_call}')

    # Get labels
    all_labels = [writer_id for _, writer_id in dataset.samples]
    unique_writers = list(set(all_labels))
    np.random.shuffle(unique_writers)
    
    # Train/val split by writer
    if args.val_split > 0:
        val_count = max(1, int(len(unique_writers) * args.val_split))
        val_writers = set(unique_writers[:val_count])
        train_writers = set(unique_writers[val_count:])
    else:
        # No validation split - use all data for training
        val_writers = set()
        train_writers = set(unique_writers)
    
    train_indices = []
    val_indices = []
    for i, (_, writer_id) in enumerate(dataset.samples):
        if writer_id in train_writers:
            train_indices.append(i)
        else:
            val_indices.append(i)
    
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else None
    
    train_labels = [all_labels[i] for i in train_indices]
    
    logger.info('')
    logger.info('📊 Dataset Split:')
    logger.info(f'   Train: {len(train_ds)} images from {len(train_writers)} writers')
    if val_ds:
        logger.info(f'   Val:   {len(val_ds)} images from {len(val_writers)} writers')
    else:
        logger.info(f'   Val:   NONE (all data used for training)')
    
    # MPerClassSampler
    train_sampler = MPerClassSampler(
        labels=train_labels,
        m=args.m_per_class,
        batch_size=args.batch_size,
        length_before_new_iter=len(train_ds) * args.patches_per_call,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    logger.info(f'   Batches per epoch: {len(train_loader)} train')
    if val_loader:
        logger.info(f'                      {len(val_loader)} val')

    # Model
    logger.info('')
    logger.info('🔧 Building model...')
    model = create_resnet_patch_encoder(
        emb_dim=args.emb_dim,
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_bn=args.freeze_bn,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'   ✓ {args.backbone.upper()} encoder created')
    logger.info(f'   Pretrained: {args.pretrained}')
    logger.info(f'   Embedding dimension: {args.emb_dim}')
    logger.info(f'   Trainable parameters: {num_params:,}')
    
    # Loss
    criterion = get_loss_function(args.loss, margin=args.margin)
    logger.info(f'   Loss: {args.loss} (margin={args.margin})')
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f'   Optimizer: AdamW (lr={args.lr:.0e}, wd={args.weight_decay})')
    
    # LR schedule
    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(
        args.lr, args.final_lr, args.epochs, niter_per_ep, args.warmup_epochs
    )

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
        logger.info(f'━━━ Epoch {epoch}/{args.epochs} ━━━')
        
        train_loss, global_iter = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            lr_schedule, global_iter, args.grad_clip, epoch, args.patches_per_call
        )
        
        # Validation
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device, args.patches_per_call)
            metric_loss = val_loss
        else:
            val_loss = None
            metric_loss = train_loss  # Use train loss if no validation
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info('')
        logger.info(f'📈 Epoch {epoch} Summary:')
        logger.info(f'   Train Loss: {train_loss:.4f}')
        if val_loss is not None:
            logger.info(f'   Val Loss:   {val_loss:.4f}')
        logger.info(f'   LR: {current_lr:.2e} | Time: {epoch_time:.1f}s')

        if metric_loss < best_val_loss:
            improvement = best_val_loss - metric_loss
            best_val_loss = metric_loss
            best_epoch = epoch
            
            ckpt_path = save_dir / f'best_epoch_{epoch}_loss_{metric_loss:.4f}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': metric_loss,
                'train_loss': train_loss,
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
            logger.info(f'⚠️  Early stopping (no improvement for {args.patience} epochs)')
            break

    total_time = time.time() - training_start
    logger.info('')
    logger.info('=' * 70)
    logger.info('🎉 Training Complete!')
    logger.info('=' * 70)
    logger.info(f'   Best loss: {best_val_loss:.4f} (epoch {best_epoch})')
    logger.info(f'   Total time: {total_time / 60:.1f} minutes')
    logger.info(f'   Checkpoints: {save_dir}')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
