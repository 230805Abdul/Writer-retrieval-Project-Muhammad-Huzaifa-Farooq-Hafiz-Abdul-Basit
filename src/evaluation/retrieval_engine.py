# src/evaluation/retrieval_engine.py
"""
Descriptor extraction and retrieval evaluation engine.

Reference: "Evaluating Feature Aggregation Methods for Deep Learning based 
           Writer Retrieval" (IJDAR 2024)

Pipeline:
1. Extract patches from pages using adaptive sampling
2. Encode patches with CNN backbone
3. Aggregate patch features (mean/GeM/VLAD/NetRVLAD)
4. Apply power normalization + L2 normalization
5. Optional: PCA whitening (fit on train, apply to test)
6. Compute similarities and evaluate mAP
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import time

import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..utils.preprocessing import load_image, binarize_otsu, resize_max_side
from ..sampling.adaptive_sampler import adaptive_sample, AdaptiveSamplingConfig
from ..aggregation.pooling import mean_pool_aggregate, gem_pool_aggregate, sum_pool_aggregate
from ..aggregation.vlad import vlad_aggregate_torch, VladParams
from ..aggregation.netvlad import NetVLAD
from ..aggregation.netrvlav import NetRVLAD
from ..evaluation.metrics import mean_average_precision


logger = logging.getLogger(__name__)

# Global learnable aggregation layers (initialized lazily)
_netvlad_layer = None
_netrvlad_layer = None


@dataclass
class RetrievalConfig:
    patch_size: int = 32
    max_side: int = 1600
    line_threshold: int = 4
    dense_stride: int = 24
    contour_step: int = 12
    max_patches: int = 1500
    agg_type: str = 'mean'  # 'mean', 'sum', 'gem', 'vlad', 'netvlad', 'netrvlad'
    vlad_params: VladParams | None = None
    mode: str = 'auto'  # 'auto', 'dense', 'contour', 'char'
    
    # Power normalization settings
    use_power_norm: bool = True
    power_alpha: float = 0.4
    
    # GeM pooling parameter
    gem_p: float = 3.0
    
    # NetVLAD/NetRVLAD settings
    num_clusters: int = 100  # Number of clusters for learnable VLAD
    
    # PCA whitening settings
    use_pca_whitening: bool = True
    pca_dim: Optional[int] = None  # None = keep all dimensions
    pca_whiten: bool = True  # Whether to whiten (divide by sqrt of eigenvalues)
    
    # Debug mode
    debug: bool = False

class PageDataset(Dataset):
    """
    Dataset that yields (bw_image, writer_id, image_path).
    Used only for descriptor extraction (no training).
    """

    def __init__(self, csv_path, root_dir=None, cfg: RetrievalConfig | None = None):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.cfg = cfg or RetrievalConfig()
        self.sampler_cfg = AdaptiveSamplingConfig(
            patch_size=self.cfg.patch_size,
            dense_stride=self.cfg.dense_stride,
            contour_step=self.cfg.contour_step,
            max_patches=self.cfg.max_patches,
            line_threshold=self.cfg.line_threshold,
            mode=self.cfg.mode,
        )

        self.samples = []
        with self.csv_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']
                writer_id = int(row['writer_id'])
                self.samples.append((img_path, writer_id))

        if not self.samples:
            raise ValueError(f'No samples loaded from {self.csv_path}')

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, rel_or_abs):
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        if self.root_dir is not None:
            return self.root_dir / p
        return p

    def __getitem__(self, idx):
        img_rel_path, writer_id = self.samples[idx]
        img_path = self._resolve_path(img_rel_path)

        img = load_image(img_path)
        img = resize_max_side(img, max_side=self.cfg.max_side)
        bw = binarize_otsu(img)

        return bw, writer_id, str(img_path)


def _get_netvlad_layer(dim: int, num_clusters: int, device: torch.device) -> NetVLAD:
    """Get or create NetVLAD layer (lazily initialized)."""
    global _netvlad_layer
    if _netvlad_layer is None or _netvlad_layer.dim != dim:
        logger.debug(f'[DEBUG] Creating NetVLAD layer: dim={dim}, clusters={num_clusters}')
        _netvlad_layer = NetVLAD(num_clusters=num_clusters, dim=dim).to(device)
        _netvlad_layer.eval()
    return _netvlad_layer


def _get_netrvlad_layer(dim: int, num_clusters: int, device: torch.device) -> NetRVLAD:
    """Get or create NetRVLAD layer (lazily initialized)."""
    global _netrvlad_layer
    if _netrvlad_layer is None or _netrvlad_layer.dim != dim:
        logger.debug(f'[DEBUG] Creating NetRVLAD layer: dim={dim}, clusters={num_clusters}')
        _netrvlad_layer = NetRVLAD(num_clusters=num_clusters, dim=dim).to(device)
        _netrvlad_layer.eval()
    return _netrvlad_layer


def aggregate_patches(
        patch_emb: torch.Tensor,
        cfg: RetrievalConfig,
        device: torch.device = None,
) -> torch.Tensor:
    """
    Aggregate [N, D] patch embeddings into a single descriptor.

    Returns [D] for mean/sum/gem, [K*D] for VLAD/NetVLAD/NetRVLAD.
    
    Power normalization is applied inside the pooling functions
    when cfg.use_power_norm is True.
    
    Args:
        patch_emb: [N, D] patch embeddings
        cfg: RetrievalConfig with aggregation settings
        device: torch device (needed for learnable aggregation layers)
        
    Returns:
        desc: [D'] aggregated descriptor (L2-normalized)
    """
    if cfg.debug:
        logger.debug(f'[DEBUG] aggregate_patches: input shape={patch_emb.shape}, '
                    f'agg_type={cfg.agg_type}')
    
    if cfg.agg_type == 'mean':
        desc = mean_pool_aggregate(
            patch_emb, 
            use_power_norm=cfg.use_power_norm,
            power_alpha=cfg.power_alpha
        )
    elif cfg.agg_type == 'sum':
        desc = sum_pool_aggregate(
            patch_emb,
            use_power_norm=cfg.use_power_norm,
            power_alpha=cfg.power_alpha
        )
    elif cfg.agg_type == 'gem':
        desc = gem_pool_aggregate(
            patch_emb,
            p=cfg.gem_p,
            use_power_norm=cfg.use_power_norm,
            power_alpha=cfg.power_alpha
        )
    elif cfg.agg_type == 'vlad':
        if cfg.vlad_params is None:
            raise ValueError('vlad_params must be provided for agg_type=\'vlad\'')
        desc = vlad_aggregate_torch(patch_emb, cfg.vlad_params)
    elif cfg.agg_type == 'netvlad':
        if device is None:
            device = patch_emb.device
        layer = _get_netvlad_layer(patch_emb.shape[1], cfg.num_clusters, device)
        with torch.no_grad():
            desc = layer(patch_emb)
        if cfg.debug:
            logger.debug(f'[DEBUG] NetVLAD output shape: {desc.shape}')
    elif cfg.agg_type == 'netrvlad':
        if device is None:
            device = patch_emb.device
        layer = _get_netrvlad_layer(patch_emb.shape[1], cfg.num_clusters, device)
        with torch.no_grad():
            desc = layer(patch_emb)
        if cfg.debug:
            logger.debug(f'[DEBUG] NetRVLAD output shape: {desc.shape}')
    else:
        raise ValueError(f'Unknown agg_type: {cfg.agg_type}. '
                        f'Supported: mean, sum, gem, vlad, netvlad, netrvlad')
    
    if cfg.debug:
        logger.debug(f'[DEBUG] aggregate_patches: output shape={desc.shape}')
    
    return desc


def extract_descriptors(
        model: torch.nn.Module,
        csv_path: str,
        root_dir: str | None,
        cfg: RetrievalConfig,
        batch_size: int = 1,
        num_workers: int = 0,
        device: str = 'cuda',
        mode: str = 'auto',
        verbose: bool = True
) -> Tuple[List[int], List[str], np.ndarray]:
    """
    Extract one descriptor per page.

    Returns:
        labels: list of writer_id
        paths: list of image paths (string)
        descs: [M, D'] numpy array of descriptors (L2-normalized rows)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dataset = PageDataset(csv_path, root_dir=root_dir, cfg=cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_labels: List[int] = []
    all_paths: List[str] = []
    all_descs: List[np.ndarray] = []
    
    if verbose:
        logger.info(f'📄 Extracting descriptors from {len(dataset)} pages...')
        logger.info(f'   Aggregation: {cfg.agg_type}')
        if cfg.agg_type == 'gem':
            logger.info(f'   GeM p-parameter: {cfg.gem_p}')
        if cfg.agg_type in ('netvlad', 'netrvlad'):
            logger.info(f'   Num clusters: {cfg.num_clusters}')
        logger.info(f'   Power normalization: {"ON" if cfg.use_power_norm else "OFF"} (α={cfg.power_alpha})')
        logger.info(f'   PCA whitening: {"ON" if cfg.use_pca_whitening else "OFF"}')
        logger.info(f'   Sampling mode: {cfg.mode}')
    
    if cfg.debug:
        logger.debug(f'[DEBUG] extract_descriptors: device={device}')
        logger.debug(f'[DEBUG] Config: {cfg}')
    
    start_time = time.time()
    patch_counts = []

    with torch.no_grad():
        iterator = tqdm(loader, desc='Extracting', disable=not verbose)
        for bw_batch, writer_ids, paths in iterator:
            # bw_batch: list of possibly different-sized images if batch_size>1
            # Easiest: process one by one
            for bw, wid, path in zip(bw_batch, writer_ids, paths):
                bw_np = bw.numpy() if isinstance(bw, torch.Tensor) else bw
                # ensure numpy array
                if not isinstance(bw_np, np.ndarray):
                    bw_np = np.array(bw_np)

                centers, patches, info = adaptive_sample(bw_np, cfg=dataset.sampler_cfg)
                patch_counts.append(patches.shape[0])
                
                if cfg.debug and len(patch_counts) <= 3:
                    logger.debug(f'[DEBUG] Image {path}: {patches.shape[0]} patches, '
                                f'mode={info["mode"]}, lines={info["line_count"]}')

                if patches.shape[0] == 0:
                    # fallback: single global patch
                    from ..utils.preprocessing import resize_max_side
                    # bw_np already resized earlier, but just in case
                    ps = cfg.patch_size
                    resized = cv2.resize(bw_np, (ps, ps))
                    patches = np.expand_dims(resized, axis=0)
                    if cfg.debug:
                        logger.warning(f'[DEBUG] No patches found for {path}, using fallback')

                # convert patches to tensor [N, 1, H, W]
                patches_t = torch.from_numpy(patches.astype(np.float32) / 255.0)
                patches_t = 1.0 - patches_t  # ink=1, bg=0
                patches_t = patches_t.unsqueeze(1).to(device)  # [N, 1, H, W]

                # forward encoder
                patch_emb = model(patches_t)  # [N, D], L2-normalized
                
                if cfg.debug and len(patch_counts) <= 3:
                    logger.debug(f'[DEBUG] Patch embeddings: shape={patch_emb.shape}, '
                                f'norm_mean={patch_emb.norm(dim=1).mean():.4f}')

                # aggregate - pass device for learnable aggregation
                desc = aggregate_patches(patch_emb, cfg, device=device)  # [D'] tensor
                desc_np = desc.cpu().numpy()
                
                if cfg.debug and len(patch_counts) <= 3:
                    logger.debug(f'[DEBUG] Descriptor: shape={desc_np.shape}, '
                                f'norm={np.linalg.norm(desc_np):.4f}')

                all_labels.append(int(wid))
                all_paths.append(path)
                all_descs.append(desc_np)

    elapsed = time.time() - start_time
    descs_arr = np.stack(all_descs, axis=0)
    # L2-normalize rows again (safety)
    norms = np.linalg.norm(descs_arr, axis=1, keepdims=True) + 1e-12
    descs_arr = descs_arr / norms
    
    if verbose:
        avg_patches = np.mean(patch_counts)
        logger.info(f'   ✓ Extracted {len(all_descs)} descriptors in {elapsed:.1f}s')
        logger.info(f'   Average patches per page: {avg_patches:.1f}')
        logger.info(f'   Descriptor dimension: {descs_arr.shape[1]}')
        unique_writers = len(set(all_labels))
        logger.info(f'   Unique writers: {unique_writers}')
    
    return all_labels, all_paths, descs_arr


def fit_pca_whitening(descs: np.ndarray, 
                       n_components: Optional[int] = None,
                       whiten: bool = True,
                       verbose: bool = True) -> PCA:
    """
    Fit PCA whitening transform on descriptors.
    
    Args:
        descs: [N, D] descriptors (should be from training set)
        n_components: number of components to keep (None = all)
        whiten: whether to apply whitening (divide by sqrt of eigenvalues)
        verbose: whether to log progress
        
    Returns:
        Fitted PCA object
    """
    if n_components is None:
        n_components = min(descs.shape[0], descs.shape[1])
    
    if verbose:
        logger.info(f'🔬 Fitting PCA whitening...')
        logger.info(f'   Input dimension: {descs.shape[1]}')
        logger.info(f'   Output dimension: {n_components}')
        logger.info(f'   Whitening: {"ON" if whiten else "OFF"}')
    
    start_time = time.time()
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(descs)
    elapsed = time.time() - start_time
    
    if verbose:
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        logger.info(f'   ✓ PCA fitted in {elapsed:.1f}s')
        logger.info(f'   Explained variance: {explained_var:.1f}%')
    
    return pca


def apply_pca_whitening(descs: np.ndarray, pca: PCA, verbose: bool = True) -> np.ndarray:
    """
    Apply PCA whitening transform and re-normalize.
    
    Args:
        descs: [N, D] descriptors
        pca: fitted PCA object
        verbose: whether to log progress
        
    Returns:
        [N, D'] transformed and L2-normalized descriptors
    """
    if verbose:
        logger.info(f'🔄 Applying PCA whitening to {descs.shape[0]} descriptors...')
    
    start_time = time.time()
    descs_pca = pca.transform(descs)
    
    # L2 normalize after PCA
    norms = np.linalg.norm(descs_pca, axis=1, keepdims=True) + 1e-12
    descs_pca = descs_pca / norms
    
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info(f'   ✓ Applied in {elapsed:.1f}s')
        logger.info(f'   Output shape: {descs_pca.shape}')
    
    return descs_pca


def extract_and_whiten_descriptors(
        model: torch.nn.Module,
        train_csv_path: str,
        test_csv_path: str,
        root_dir: str | None,
        cfg: RetrievalConfig,
        batch_size: int = 1,
        num_workers: int = 0,
        device: str = 'cuda',
        verbose: bool = True
) -> Tuple[List[int], List[str], np.ndarray, Optional[PCA]]:
    """
    Extract descriptors with PCA whitening.
    
    Fits PCA on train descriptors, applies to test descriptors.
    
    Args:
        model: encoder model
        train_csv_path: path to training set CSV (for PCA fitting)
        test_csv_path: path to test set CSV (for evaluation)
        root_dir: root directory for image paths
        cfg: retrieval configuration
        batch_size: batch size for data loading
        num_workers: number of data loading workers
        device: device to run inference on
        verbose: whether to log progress
        
    Returns:
        labels: list of writer IDs (test set)
        paths: list of image paths (test set)
        descs: [M, D'] whitened descriptors (test set)
        pca: fitted PCA object (or None if whitening disabled)
    """
    if verbose:
        logger.info('=' * 60)
        logger.info('🔬 Descriptor Extraction Pipeline')
        logger.info('=' * 60)
    
    # Extract train descriptors for PCA fitting
    if cfg.use_pca_whitening:
        if verbose:
            logger.info('')
            logger.info('📊 Step 1/3: Extracting TRAIN descriptors (for PCA fitting)...')
        _, _, train_descs = extract_descriptors(
            model=model,
            csv_path=train_csv_path,
            root_dir=root_dir,
            cfg=cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            verbose=verbose
        )
        
        if verbose:
            logger.info('')
            logger.info('📊 Step 2/3: Fitting PCA whitening...')
        pca = fit_pca_whitening(
            train_descs, 
            n_components=cfg.pca_dim,
            whiten=cfg.pca_whiten,
            verbose=verbose
        )
    else:
        pca = None
        if verbose:
            logger.info('⏭️  PCA whitening disabled, skipping train extraction')
    
    # Extract test descriptors
    if verbose:
        step = '2/2' if not cfg.use_pca_whitening else '3/3'
        logger.info('')
        logger.info(f'📊 Step {step}: Extracting TEST descriptors...')
    labels, paths, descs = extract_descriptors(
        model=model,
        csv_path=test_csv_path,
        root_dir=root_dir,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        verbose=verbose
    )
    
    # Apply PCA whitening if enabled
    if pca is not None:
        if verbose:
            logger.info('')
            logger.info('🔄 Applying PCA whitening to test descriptors...')
        descs = apply_pca_whitening(descs, pca, verbose=verbose)
    
    if verbose:
        logger.info('')
        logger.info('✅ Descriptor extraction complete!')
        logger.info(f'   Final descriptor shape: {descs.shape}')
        logger.info('=' * 60)
    
    return labels, paths, descs, pca


def evaluate_retrieval(
        model: torch.nn.Module,
        csv_path: str,
        root_dir: str | None,
        cfg: RetrievalConfig,
        batch_size: int = 1,
        num_workers: int = 0,
        device: str = 'cuda',
        mode: str = 'auto',
        verbose: bool = True
):
    """
    Convenience wrapper:
    Extract descriptors for all pages in CSV and compute mAP / Top-k.
    """
    if verbose:
        logger.info('=' * 60)
        logger.info('🎯 Writer Retrieval Evaluation')
        logger.info('=' * 60)
    
    labels, paths, descs = extract_descriptors(
        model=model,
        csv_path=csv_path,
        root_dir=root_dir,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        verbose=verbose
    )

    if verbose:
        logger.info('')
        logger.info('📊 Computing retrieval metrics...')
    
    start_time = time.time()
    mAP, metrics = mean_average_precision(labels, descs)
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info(f'   ✓ Metrics computed in {elapsed:.1f}s')
        logger.info('')
        logger.info('📈 Results:')
        logger.info(f'   mAP:    {mAP * 100:.2f}%')
        if 'top1' in metrics:
            logger.info(f'   Top-1:  {metrics["top1"] * 100:.2f}%')
        if 'top5' in metrics:
            logger.info(f'   Top-5:  {metrics["top5"] * 100:.2f}%')
        if 'top10' in metrics:
            logger.info(f'   Top-10: {metrics["top10"] * 100:.2f}%')
        logger.info('=' * 60)
    
    return labels, paths, descs, metrics
