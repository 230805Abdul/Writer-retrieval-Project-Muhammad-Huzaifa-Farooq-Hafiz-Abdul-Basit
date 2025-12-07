"""
Extract page-level descriptors for writer retrieval.

Usage:
    python -m scripts.extract_descs \
        --csv data/cvl/cvl_test.csv \
        --checkpoint checkpoints/best.pt \
        --out experiments/cvl/cvl_descs.npy

Can also be imported and called from notebooks with keyword arguments.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.evaluation.retrieval_engine import (
    extract_descriptors,
    RetrievalConfig
)
from src.features.resnet_patch_extractor import create_resnet_patch_encoder
from src.utils.logging_config import setup_script_logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract page-level descriptors for writer retrieval'
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with image_path,writer_id columns')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output path for descriptors (.npy)')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Optional root directory for image paths')
    parser.add_argument('--agg-type', type=str, default='sum',
                        choices=['mean', 'sum', 'gem', 'vlad'],
                        help='Aggregation type (default: sum)')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'dense', 'contour', 'char'],
                        help='Patch sampling mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (default: output dir/logs)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs in console (always saved to file)')
    return parser.parse_args()


def main(csv_path=None, checkpoint=None, out_path=None, root_dir=None,
         agg_type='sum', mode='auto', device='cuda', num_workers=2,
         batch_size=1, log_dir=None, debug=False):
    """
    Extract descriptors from a trained model.
    
    Can be called from command line (uses argparse) or directly with kwargs.
    """
    global logger
    
    # If called without args, parse from command line
    if csv_path is None:
        args = parse_args()
        csv_path = args.csv
        checkpoint = args.checkpoint
        out_path = args.out
        root_dir = args.root_dir
        agg_type = args.agg_type
        mode = args.mode
        device = args.device
        num_workers = args.num_workers
        batch_size = args.batch_size
        log_dir = args.log_dir
        debug = args.debug
    
    # Setup logging with file output
    out_path_obj = Path(out_path)
    log_dir_path = Path(log_dir) if log_dir else out_path_obj.parent / 'logs'
    
    logger = setup_script_logging(
        script_name='extract_descs',
        log_dir=str(log_dir_path),
        debug=debug
    )
    
    logger.info('=' * 60)
    logger.info('📤 Extracting Page Descriptors')
    logger.info('=' * 60)
    logger.info(f'   CSV: {csv_path}')
    logger.info(f'   Checkpoint: {checkpoint}')
    logger.info(f'   Output: {out_path}')
    logger.info(f'   Aggregation: {agg_type}')
    logger.info(f'   Mode: {mode}')
    logger.info('')
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location='cpu')
    emb_dim = ckpt.get('args', {}).get('emb_dim', 128)
    
    logger.info(f'📂 Loading model (emb_dim={emb_dim})...')
    model = create_resnet_patch_encoder(emb_dim=emb_dim)
    model.load_state_dict(ckpt['model_state'], strict=True)
    
    cfg = RetrievalConfig(
        patch_size=32,
        dense_stride=24,
        contour_step=12,
        max_patches=1500,
        agg_type=agg_type,
        mode=mode,
        use_power_norm=True,
        power_alpha=0.4,
    )
    
    labels, paths, descs = extract_descriptors(
        model=model,
        csv_path=csv_path,
        root_dir=root_dir,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        device=str(device),
        verbose=True
    )
    
    # Save descriptors
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, descs)
    
    logger.info('')
    logger.info(f'✅ Saved {len(descs)} descriptors to {out_path}')
    logger.info(f'   Shape: {descs.shape}')
    
    return labels, paths, descs


if __name__ == '__main__':
    main()
