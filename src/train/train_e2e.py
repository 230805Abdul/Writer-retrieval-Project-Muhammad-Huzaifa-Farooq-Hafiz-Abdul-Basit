# src/train/train_e2e.py
"""
End-to-End Writer Retrieval Training.

CRITICAL DIFFERENCE FROM train_resnet_triplet.py:
- Old: Loss on individual PATCHES → Model learns patch similarity
- New: Loss on AGGREGATED PAGE descriptors → Model learns WRITER similarity

This is the key insight: when we train on patches, we're asking "are these 
patches similar?" But patches from the same writer can look very different 
(letter "a" vs word "the"). This confuses the model.

When we train on aggregated page descriptors, we're asking "do these pages
come from the same writer?" The aggregation captures the overall style
(sharp loops, slant, stroke width) rather than individual letter shapes.

Usage:
    python -m src.train.train_e2e \\
        --csv data/cvl_train.csv \\
        --root-dir data/CVL \\
        --agg-type gem \\
        --patches-per-page 32 \\
        --batch-size 16 \\
        --m-per-class 2

Key hyperparameters:
    - patches_per_page: 32-64 (more = better representation, slower training)
    - batch_size: 16-32 (limited by GPU memory due to page-level batching)
    - m_per_class: 2 (2 pages per writer per batch, standard for triplet)
    - agg_type: 'gem' (best for most cases), 'netvlad' (more parameters)
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
import torch.nn as nn
import torch.optim as optim

from ..features.resnet_patch_extractor import create_resnet_patch_encoder
from ..models.e2e_writer_net import EndToEndWriterNet
from .page_bag_dataset import PageBagDataset, PageBagCollateFn, create_page_dataloader
from .triplet_loss import BatchHardTripletLoss
from .samplers import MPerClassSampler
from .augmentation import get_train_augmentation
from ..sampling.adaptive_sampler import AdaptiveSamplingConfig
from ..utils.preprocessing import clear_dir

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='End-to-End Writer Retrieval Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on CVL with GeM aggregation
    python -m src.train.train_e2e --csv data/cvl_train.csv --agg-type gem
    
    # Train on IAM with NetVLAD
    python -m src.train.train_e2e --csv data/iam_train.csv --agg-type netvlad
    
    # Fine-tune from pretrained patch encoder
    python -m src.train.train_e2e --csv data/cvl_train.csv --pretrained checkpoints/best.pt
        """
    )

    # Data
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with image_path,writer_id')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Root directory for image paths')
    parser.add_argument('--val-csv', type=str, default=None,
                        help='Validation CSV (if not provided, split from train)')

    # Model
    parser.add_argument('--encoder', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture for patch encoder')
    parser.add_argument('--emb-dim', type=int, default=64,
                        help='Patch embedding dimension')
    parser.add_argument('--agg-type', type=str, default='gem',
                        choices=['gem', 'mean', 'sum', 'netvlad', 'netrvlad'],
                        help='Aggregation type (gem recommended)')
    parser.add_argument('--num-clusters', type=int, default=100,
                        help='Number of clusters for NetVLAD')
    parser.add_argument('--gem-p', type=float, default=3.0,
                        help='Initial GeM pooling power')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained patch encoder checkpoint')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (number of pages per batch)')
    parser.add_argument('--patches-per-page', type=int, default=32,
                        help='Patches to sample from each page')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--final-lr', type=float, default=1e-6,
                        help='Final learning rate for cosine schedule')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Triplet loss margin')
    parser.add_argument('--m-per-class', type=int, default=2,
                        help='Pages per writer per batch')

    # Sampling
    parser.add_argument('--sampler', type=str, default='contour',
                        choices=['dense', 'contour', 'adaptive', 'char'],
                        help='Patch sampling strategy')
    parser.add_argument('--patch-size', type=int, default=32,
                        help='Patch size in pixels')

    # Regularization
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--grad-clip', type=float, default=2.0)
    parser.add_argument('--weight-decay', type=float, default=1e-5)

    # Infrastructure
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cache-patches', action='store_true', default=True,
                        help='Cache patches in memory')
    parser.add_argument('--no-cache', dest='cache_patches', action='store_false')

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


def train_one_epoch(model, loader, criterion, optimizer, device,
                    lr_schedule, global_iter, grad_clip, epoch):
    """Train for one epoch with page-level triplet loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (pages, labels) in enumerate(loader):
        # pages: [B, P, 1, H, W] - batch of pages, each with P patches
        # labels: [B] - one writer ID per page

        pages = pages.to(device)
        labels = labels.to(device)

        # Update learning rate
        if global_iter < len(lr_schedule):
            lr = lr_schedule[global_iter]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()

        # Forward pass through E2E model
        # Input: [B, P, 1, H, W]
        # Output: [B, D] - aggregated page descriptors
        page_descriptors = model(pages)

        # Triplet loss on PAGE descriptors (not patches!)
        # This is the key difference from patch-level training
        loss = criterion(page_descriptors, labels)

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_iter += 1

        # Progress logging
        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f'  Batch {batch_idx}/{len(loader)}: '
                         f'loss={loss.item():.4f}, lr={current_lr:.2e}')

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_iter


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate with page-level loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for pages, labels in loader:
        pages = pages.to(device)
        labels = labels.to(device)

        page_descriptors = model(pages)
        loss = criterion(page_descriptors, labels)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    clear_dir(save_dir)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info('=' * 70)
    logger.info('🚀 END-TO-END Writer Retrieval Training')
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

    # Sampler configuration
    sampler_map = {
        'dense': 'dense',
        'contour': 'contour',
        'char': 'char',
        'adaptive': 'auto',
    }

    sampler_cfg = AdaptiveSamplingConfig(
        patch_size=args.patch_size,
        dense_stride=args.patch_size - 8,
        contour_step=12,
        max_patches=2000,
        line_threshold=4,
        mode=sampler_map[args.sampler],
    )

    transform = get_train_augmentation() if args.augment else None
    if args.augment:
        logger.info('✅ Morphological augmentation ENABLED')

    # =========================================================================
    # Build E2E Model
    # =========================================================================
    logger.info('')
    logger.info('🔧 Building End-to-End model...')

    # Create E2E model with aggregation
    # EndToEndWriterNet creates its own encoder internally
    model = EndToEndWriterNet(
        emb_dim=args.emb_dim,
        backbone=args.encoder,
        pretrained=True,
        aggregator=args.agg_type,  # 'gem', 'netvlad', 'netrvlad', 'attention'
        num_clusters=args.num_clusters,
        gem_p=args.gem_p,
        power_norm=True,
        power_alpha=0.4,
    ).to(device)

    # Load pretrained weights if provided (for fine-tuning)
    if args.pretrained:
        logger.info(f'   Loading pretrained encoder from: {args.pretrained}')
        ckpt = torch.load(args.pretrained, map_location='cpu')
        # Handle different checkpoint formats
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        # Load into encoder only
        model.encoder.load_state_dict(state_dict, strict=False)
        logger.info('   ✓ Pretrained encoder weights loaded')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'   ✓ EndToEndWriterNet created')
    logger.info(f'   Aggregation: {args.agg_type}')
    logger.info(f'   Embedding dimension: {args.emb_dim}')
    logger.info(f'   Output dimension: {model.output_dim}')
    logger.info(f'   Trainable parameters: {num_params:,}')

    # =========================================================================
    # Dataset & DataLoader
    # =========================================================================
    logger.info('')
    logger.info('📂 Loading dataset...')
    start_time = time.time()

    # Create full dataset
    full_dataset = PageBagDataset(
        csv_path=args.csv,
        root_dir=args.root_dir,
        sampler_cfg=sampler_cfg,
        patches_per_page=args.patches_per_page,
        cache_patches=args.cache_patches,
        transform=transform,
        debug=args.debug,
    )

    load_time = time.time() - start_time
    logger.info(f'   ✓ Loaded {len(full_dataset)} pages in {load_time:.1f}s')
    logger.info(f'   Writers: {len(full_dataset.writers)}')

    # Split by writer (not random split!)
    all_labels = full_dataset.get_all_labels()
    unique_writers = list(set(all_labels))
    np.random.shuffle(unique_writers)

    val_count = max(1, int(len(unique_writers) * args.val_split))
    val_writers = set(unique_writers[:val_count])
    train_writers = set(unique_writers[val_count:])

    train_indices = []
    val_indices = []
    for i, (_, writer_id) in enumerate(full_dataset.samples):
        if writer_id in train_writers:
            train_indices.append(i)
        else:
            val_indices.append(i)

    # Create subset datasets
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]

    logger.info('')
    logger.info('📊 Dataset Split:')
    logger.info(f'   Train: {len(train_ds)} pages from {len(train_writers)} writers')
    logger.info(f'   Val:   {len(val_ds)} pages from {len(val_writers)} writers')

    # MPerClassSampler for balanced batches
    train_sampler = MPerClassSampler(
        labels=train_labels,
        m=args.m_per_class,
        batch_size=args.batch_size,
        length_before_new_iter=len(train_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=PageBagCollateFn(),
    )

    # Validation loader (no special sampling needed)
    val_sampler = MPerClassSampler(
        labels=val_labels,
        m=args.m_per_class,
        batch_size=args.batch_size,
        length_before_new_iter=len(val_ds),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=PageBagCollateFn(),
    )

    logger.info(f'   ✓ Using MPerClassSampler (m={args.m_per_class} pages/writer)')
    logger.info(f'   Batches per epoch: {len(train_loader)} train, {len(val_loader)} val')
    logger.info(f'   Patches per page: {args.patches_per_page}')
    logger.info(f'   Effective patches per batch: {args.batch_size * args.patches_per_page}')

    # =========================================================================
    # Training Setup
    # =========================================================================
    criterion = BatchHardTripletLoss(margin=args.margin)
    logger.info(f'   Loss: BatchHardTripletLoss (margin={args.margin})')
    logger.info('   📍 KEY: Loss computed on PAGE descriptors, not patches!')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    logger.info(f'   Optimizer: AdamW (lr={args.lr:.0e}, wd={args.weight_decay:.0e})')

    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(
        args.lr, args.final_lr, args.epochs, niter_per_ep, args.warmup_epochs
    )
    logger.info(f'   LR Schedule: Cosine ({args.lr:.0e} → {args.final_lr:.0e})')
    logger.info(f'   Warmup: {args.warmup_epochs} epochs')

    best_val_loss = float('inf')
    best_epoch = 0
    global_iter = 0
    patience_counter = 0

    logger.info('')
    logger.info('=' * 70)
    logger.info('🏋️  Starting End-to-End Training')
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

        # Log GeM p parameter if using GeM
        gem_p_str = ''
        if args.agg_type == 'gem' and hasattr(model, 'aggregator'):
            gem_p = model.aggregator.p.item()
            gem_p_str = f', gem_p={gem_p:.2f}'

        logger.info(f'   Train Loss: {train_loss:.4f}')
        logger.info(f'   Val Loss:   {val_loss:.4f}')
        logger.info(f'   LR: {current_lr:.2e}{gem_p_str}')
        logger.info(f'   Time: {epoch_time:.1f}s')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save full checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': vars(args),
            }

            ckpt_path = save_dir / f'best_epoch_{epoch}_loss_{val_loss:.4f}.pt'
            torch.save(checkpoint, ckpt_path)
            logger.info(f'   ✅ Saved best model: {ckpt_path.name}')

            # Also save just the patch encoder for compatibility with eval_retrieval.py
            # Wrap in dict with 'model_state' key to match expected format
            # Include all hyperparameters that may be needed at evaluation time
            encoder_ckpt = {
                'epoch': epoch,
                'model_state': model.encoder.state_dict(),
                'val_loss': val_loss,
                'args': {
                    # Model architecture
                    'emb_dim': args.emb_dim,
                    'backbone': args.encoder,
                    
                    # Aggregation parameters
                    'agg_type': args.agg_type,
                    'gem_p': args.gem_p,
                    'num_clusters': args.num_clusters,
                    
                    # Sampling parameters
                    'sampler': args.sampler,
                    'patch_size': args.patch_size,
                    'patches_per_page': args.patches_per_page,
                    
                    # Training hyperparameters (for reference)
                    'margin': args.margin,
                    'batch_size': args.batch_size,
                    'm_per_class': args.m_per_class,
                },
            }
            encoder_path = save_dir / f'encoder_epoch_{epoch}_loss_{val_loss:.4f}.pt'
            torch.save(encoder_ckpt, encoder_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f'')
                logger.info(f'⚠️  Early stopping triggered (patience={args.patience})')
                break

    # =========================================================================
    # Training Complete
    # =========================================================================
    total_time = time.time() - training_start

    logger.info('')
    logger.info('=' * 70)
    logger.info('🎉 Training Complete!')
    logger.info('=' * 70)
    logger.info(f'   Best epoch: {best_epoch}')
    logger.info(f'   Best val loss: {best_val_loss:.4f}')
    logger.info(f'   Total time: {total_time / 60:.1f} minutes')
    logger.info('')
    logger.info(f'📁 Checkpoints saved to: {save_dir}')
    logger.info('   - best_*.pt: Full E2E model checkpoint')
    logger.info('   - encoder_*.pt: Patch encoder only (for eval_retrieval.py)')
    logger.info('')
    logger.info('Next steps:')
    logger.info('   1. Extract descriptors: python -m scripts.extract_descs ...')
    logger.info('   2. Evaluate retrieval: python -m src.evaluation.eval_retrieval ...')


if __name__ == '__main__':
    main()
