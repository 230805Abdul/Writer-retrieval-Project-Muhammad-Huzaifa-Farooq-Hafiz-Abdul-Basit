# src/evaluation/eval_retrieval.py
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from ..features.resnet_patch_extractor import create_resnet_patch_encoder
from ..aggregation.vlad import VladParams
from ..evaluation.retrieval_engine import RetrievalConfig, evaluate_retrieval
from ..evaluation.metrics import mean_average_precision
from ..reranking.qe import apply_qe
from ..reranking.sgr import apply_sgr
from ..reranking.sgr_plus import apply_sgr_plus


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate writer retrieval')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV with image_path,writer_id')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Optional root dir for image paths')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained patch encoder checkpoint (.pt)')
    parser.add_argument('--agg-type', type=str, default='mean',
                        choices=['mean', 'sum', 'gem', 'vlad', 'netvlad', 'netrvlad'],
                        help='Aggregation type: mean, sum, gem, vlad, netvlad, netrvlad')
    parser.add_argument('--vlad-centers', type=str, default=None,
                        help='Path to npz file with VLAD centers (key \'centers\') - for vlad only')
    parser.add_argument('--num-clusters', type=int, default=None,
                        help='Number of clusters for netvlad/netrvlad (default: from checkpoint or 100)')
    parser.add_argument('--gem-p', type=float, default=None,
                        help='GeM pooling power (default: from checkpoint or 3.0)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'dense', 'contour', 'char', 'attention'],
                        help='Sampling mode for patch extraction')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (default: experiments/logs)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs in console (always saved to file)')

    # reranking
    parser.add_argument('--qe', action='store_true',
                        help='Apply simple query expansion')
    parser.add_argument('--rerank', type=str, default='none',
                        choices=['none', 'sgr', 'sgr_plus'],
                        help='Apply SGR / SGR+ style re-ranking')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # logger.info('=' * 70)
    # logger.info('📊 Evaluating writer retrieval with configuration:')
    # logger.info('=' * 70)
    # for k, v in vars(args).items():
    #     logger.info(f'   {k:20s}: {v}')
    # logger.info(f'💻 Device: {device}')
    # logger.info('=' * 70)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Get model configuration from checkpoint
    ckpt_args = ckpt.get('args', {})
    emb_dim = ckpt_args.get('emb_dim', 128)
    
    # Detect backbone from checkpoint
    # E2E training saves 'resnet18' backbone, patch-level training uses 'small'
    backbone = ckpt_args.get('backbone', None)
    
    # Auto-detect backbone from state_dict structure if not specified
    if backbone is None:
        state_dict = ckpt.get('model_state', ckpt)
        # Check for layer4 (ResNet18/34 have it, ResNetSmall does not)
        has_layer4 = any('layer4' in k for k in state_dict.keys())
        # ResNet34 has more blocks than ResNet18:
        # - layer3.5 exists in ResNet34 (6 blocks: 0-5), but not in ResNet18 (2 blocks: 0-1)
        # - layer2.3 exists in ResNet34 (4 blocks: 0-3), but not in ResNet18 (2 blocks: 0-1)
        has_layer3_5 = any('layer3.5' in k for k in state_dict.keys())
        has_layer2_3 = any('layer2.3' in k for k in state_dict.keys())
        # ResNet18/34 conv1 has shape [64, 1, 7, 7], ResNetSmall has [32, 1, 3, 3]
        conv1_shape = state_dict.get('conv1.weight', torch.zeros(1)).shape
        
        if has_layer3_5 or has_layer2_3:
            # ResNet34 has deeper layers
            backbone = 'resnet34'
            logger.info(f'   Auto-detected backbone: resnet34 (from checkpoint structure)')
        elif has_layer4 or (len(conv1_shape) == 4 and conv1_shape[0] == 64):
            backbone = 'resnet18'
            logger.info(f'   Auto-detected backbone: resnet18 (from checkpoint structure)')
        else:
            backbone = 'small'
            logger.info(f'   Auto-detected backbone: small (from checkpoint structure)')
    else:
        logger.info(f'   Backbone from checkpoint: {backbone}')

    model = create_resnet_patch_encoder(emb_dim=emb_dim, backbone=backbone)
    model.load_state_dict(ckpt['model_state'], strict=True)

    # Read additional parameters from checkpoint (with command-line override)
    # GeM pooling power: prefer command-line arg > checkpoint > default 3.0
    gem_p = args.gem_p
    if gem_p is None:
        gem_p = ckpt_args.get('gem_p', 3.0)
    
    # Number of clusters for NetVLAD: prefer command-line arg > checkpoint > default 100
    num_clusters = args.num_clusters
    if num_clusters is None:
        num_clusters = ckpt_args.get('num_clusters', 100)
    
    # Patch size: use checkpoint value if available, else default 32
    patch_size = ckpt_args.get('patch_size', 32)
    
    # Sampling mode: use command-line arg, or infer from checkpoint sampler
    mode = args.mode
    if mode == 'auto':
        # Try to use the sampler from checkpoint if available
        ckpt_sampler = ckpt_args.get('sampler', None)
        if ckpt_sampler in ['dense', 'contour', 'char']:
            mode = ckpt_sampler
            logger.info(f'   Sampling mode: {mode} (from checkpoint)')
    
    # Log parameters being used
    if args.agg_type == 'gem':
        logger.info(f'   GeM p-parameter: {gem_p} (from {"args" if args.gem_p else "checkpoint" if "gem_p" in ckpt_args else "default"})')
    elif args.agg_type in ['netvlad', 'netrvlad']:
        logger.info(f'   Num clusters: {num_clusters} (from {"args" if args.num_clusters else "checkpoint" if "num_clusters" in ckpt_args else "default"})')

    # Build RetrievalConfig
    cfg_kwargs = dict(
        patch_size=patch_size,
        max_side=1600,
        line_threshold=4,
        dense_stride=24,
        contour_step=12,
        max_patches=1500,
        agg_type=args.agg_type,
        vlad_params=None,
        mode=mode,
        num_clusters=num_clusters,
        gem_p=gem_p,
        debug=args.debug,
    )

    if args.agg_type == 'vlad':
        if args.vlad_centers is None:
            raise ValueError('--vlad-centers is required when agg-type=vlad')
        npz = np.load(args.vlad_centers)
        centers = npz['centers'].astype(np.float32)
        vlad_params = VladParams(centers=centers)
        cfg_kwargs['vlad_params'] = vlad_params
    
    if args.agg_type in ['netvlad', 'netrvlad']:
        logger.info(f'🔧 Using {args.agg_type} with {num_clusters} clusters')

    cfg = RetrievalConfig(**cfg_kwargs)

    # baseline extraction
    labels, paths, descs, metrics = evaluate_retrieval(
        model=model,
        csv_path=args.csv,
        root_dir=args.root_dir,
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    logger.info('')
    logger.info('=== Baseline retrieval ===')
    logger.info(f"  #queries: {metrics['n_queries']}")
    logger.info(f"  mAP:   {metrics['mAP']:.4f}")
    logger.info(f"  Top-1: {metrics['Top1']:.4f}")
    logger.info(f"  Top-5: {metrics['Top5']:.4f}")
    logger.info(f"  Top-10:{metrics['Top10']:.4f}")

    # optional QE / reranking
    descs_mod = descs.copy()
    if args.qe:
        logger.info('')
        logger.info('🔄 Applying QE...')
        descs_mod = apply_qe(descs_mod, top_k=5)

    if args.rerank == 'sgr':
        logger.info('🔄 Applying SGR...')
        # k=3 matches reference paper (2-5 recommended), gamma=0.1 for sharper kernel
        descs_mod = apply_sgr(descs_mod, k=3, gamma=0.1, num_iterations=2)
    elif args.rerank == 'sgr_plus':
        logger.info('🔄 Applying SGR+...')
        # Fixed: k=3 instead of k=20 (reference paper uses k=2)
        descs_mod = apply_sgr_plus(descs_mod, qe_top_k=5, sgr_k=3, alpha=0.5)

    if args.qe or args.rerank != 'none':
        mAP2, metrics2 = mean_average_precision(labels, descs_mod)
        logger.info('')
        logger.info('=== After QE / reranking ===')
        logger.info(f"  mAP:   {metrics2['mAP']:.4f}")
        logger.info(f"  Top-1: {metrics2['Top1']:.4f}")
        logger.info(f"  Top-5: {metrics2['Top5']:.4f}")
        logger.info(f"  Top-10:{metrics2['Top10']:.4f}")

    return labels, paths, descs_mod, metrics


if __name__ == '__main__':
    main()
