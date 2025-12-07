# src/train/fit_vlad_centers.py
import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.preprocessing import load_image, binarize_otsu, resize_max_side
from ..sampling.adaptive_sampler import adaptive_sample, AdaptiveSamplingConfig
from ..features.resnet_patch_extractor import create_resnet_patch_encoder
from ..aggregation.vlad import fit_kmeans_for_vlad


logger = logging.getLogger(__name__)


class PagePatchDataset(Dataset):
    """
    Dataset that returns (patches, writer_id) for each page.

    For each page:
        - load image
        - resize
        - binarize
        - adaptive_sample -> patches [N, H, W]
    We will later encode all patches from each page.
    """

    def __init__(self, csv_path, root_dir=None, max_side=1600, sampler_cfg=None):
        import csv

        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.max_side = max_side
        self.sampler_cfg = sampler_cfg or AdaptiveSamplingConfig()

        self.samples = []
        with self.csv_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']
                writer_id = int(row['writer_id'])
                self.samples.append((img_path, writer_id))

        if not self.samples:
            raise ValueError(f'No samples found in {csv_path}')

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
        img_rel, writer_id = self.samples[idx]
        img_path = self._resolve_path(img_rel)

        img = load_image(img_path)
        img = resize_max_side(img, max_side=self.max_side)
        bw = binarize_otsu(img)

        centers, patches, info = adaptive_sample(bw, self.sampler_cfg)
        # patches: [N, H, W] uint8

        return patches, writer_id, str(img_path)


def parse_args():
    p = argparse.ArgumentParser(description='Fit KMeans for VLAD centers')
    p.add_argument('--csv', type=str, required=True,
                   help='CSV with image_path,writer_id for training pages')
    p.add_argument('--root-dir', type=str, default=None,
                   help='Optional root directory for images')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Trained patch encoder checkpoint (.pt)')
    p.add_argument('--out', type=str, default='vlad_centers_k32.npz',
                   help='Output npz file for centers')
    p.add_argument('--num-clusters', type=int, default=32,
                   help='Number of VLAD clusters (K)')
    p.add_argument('--num-samples', type=int, default=100000,
                   help='Target number of patch embeddings to sample for KMeans')
    p.add_argument('--max-side', type=int, default=1600,
                   help='Max side for resizing pages')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--batch-pages', type=int, default=1,
                   help='Pages per batch (1 is simplest because page sizes vary)')
    p.add_argument('--log-dir', type=str, default=None,
                   help='Directory for log files (default: output dir/logs)')
    p.add_argument('--debug', action='store_true',
                   help='Show debug logs in console (always saved to file)')
    return p.parse_args()


def main():
    """
    Main function to fit VLAD centers using KMeans on patch embeddings.
    What it does:
    - Loads a pretrained patch encoder.
    - Loads pages from a CSV file, samples patches using adaptive sampling.
    - Encodes patches to get embeddings.
    - Collects a specified number of embeddings.
    - Fits KMeans to find VLAD centers.

    What is a patch?
    A patch is a small image region extracted from a document page, typically used for feature extraction in handwriting analysis.
    What is VLAD?
    VLAD (Vector of Locally Aggregated Descriptors) is a method for aggregating local image descriptors into a fixed-length global descriptor.
    What is KMeans?
    KMeans is a clustering algorithm that partitions data into K clusters by minimizing the variance within each cluster.
    What is a patch encoder?
    A patch encoder is a neural network model that transforms image patches into feature embeddings.
    What is feature embedding?
    A feature embedding is a numerical representation of data (like image patches) in a lower-dimensional space, capturing essential characteristics.
    What is VLAD center?
    A VLAD center is a cluster centroid obtained from KMeans clustering, used in the VLAD aggregation process.

    What is the benefit of using VLAD centers?
    VLAD centers help in efficiently summarizing local features into a compact global representation, improving retrieval and classification tasks.
    Returns: None

    """
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info('=' * 70)
    logger.info('🎯 Fitting VLAD centers with configuration:')
    logger.info('=' * 70)
    for k, v in vars(args).items():
        logger.info(f'   {k:20s}: {v}')
    logger.info(f'💻 Device: {device}')
    logger.info('=' * 70)

    # Load patch encoder
    ckpt = torch.load(args.checkpoint, map_location=device)
    emb_dim = ckpt.get('args', {}).get('emb_dim', 128)
    model = create_resnet_patch_encoder(emb_dim=emb_dim)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(device)
    model.eval()

    sampler_cfg = AdaptiveSamplingConfig(
        patch_size=32,
        dense_stride=24,
        contour_step=12,
        max_patches=1500,
        line_threshold=4,
    )

    dataset = PagePatchDataset(
        csv_path=args.csv,
        root_dir=args.root_dir,
        max_side=args.max_side,
        sampler_cfg=sampler_cfg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_pages,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_embs = []
    total = 0
    target = args.num_samples

    with torch.no_grad():
        for patches_batch, writer_ids, paths in loader:
            # each element in patches_batch is [N_i, H, W]; with batch_pages>1 this is messy
            # so we process each page in the batch independently
            for patches, wid, path in zip(patches_batch, writer_ids, paths):
                patches_np = patches.numpy()  # [N, H, W]
                if patches_np.shape[0] == 0:
                    continue

                # convert to tensor [N, 1, H, W]
                patches_t = torch.from_numpy(patches_np.astype(np.float32) / 255.0)
                patches_t = 1.0 - patches_t
                patches_t = patches_t.unsqueeze(1).to(device)

                emb = model(patches_t)  # [N, D]
                emb_cpu = emb.cpu().numpy()
                all_embs.append(emb_cpu)
                total += emb_cpu.shape[0]

                logger.debug(f'Collected {total} embeddings so far...')

                if total >= target:
                    break
            if total >= target:
                break

    if not all_embs:
        raise RuntimeError('No embeddings collected for KMeans!')

    all_embs_arr = np.concatenate(all_embs, axis=0)
    if all_embs_arr.shape[0] > target:
        # random subset
        idx = np.random.choice(all_embs_arr.shape[0], target, replace=False)
        all_embs_arr = all_embs_arr[idx]

    logger.info(f'📊 Fitting KMeans on {all_embs_arr.shape[0]} embeddings of dim {all_embs_arr.shape[1]}')

    vlad_params = fit_kmeans_for_vlad(
        embeddings=all_embs_arr,
        num_clusters=args.num_clusters,
        max_iter=100,
        random_state=42,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, centers=vlad_params.centers)
    logger.info(f'✅ Saved VLAD centers to {out_path}')


if __name__ == '__main__':
    main()
