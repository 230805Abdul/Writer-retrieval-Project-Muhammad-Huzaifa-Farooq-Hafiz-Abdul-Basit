"""
Compute retrieval rankings from descriptors.

Usage:
    python -m scripts.make_ranks \
        --descs experiments/cvl/cvl_descs.npy \
        --csv data/cvl/cvl_test.csv \
        --out experiments/cvl/cvl_ranks.json

Can also be imported and called from notebooks with keyword arguments.
"""
import argparse
import json
import csv
import logging
import numpy as np
from pathlib import Path

from src.utils.logging_config import setup_script_logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute retrieval rankings from descriptors'
    )
    parser.add_argument('--descs', type=str, required=True,
                        help='Path to descriptors (.npy)')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with image_path,writer_id')
    parser.add_argument('--out', type=str, required=True,
                        help='Output path for ranks (.json)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (default: output dir/logs)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs in console (always saved to file)')
    return parser.parse_args()


def main(descs_path=None, csv_path=None, out_path=None, log_dir=None, debug=False):
    """
    Compute rankings from descriptors.
    
    Can be called from command line or directly with kwargs.
    """
    global logger
    
    # If called without args, parse from command line
    if descs_path is None:
        args = parse_args()
        descs_path = args.descs
        csv_path = args.csv
        out_path = args.out
        log_dir = args.log_dir
        debug = args.debug
    
    # Setup logging with file output
    out_path_obj = Path(out_path)
    log_dir_path = Path(log_dir) if log_dir else out_path_obj.parent / 'logs'
    
    logger = setup_script_logging(
        script_name='make_ranks',
        log_dir=str(log_dir_path),
        debug=debug
    )
    
    descs_path = Path(descs_path)
    csv_path = Path(csv_path)
    out_path = out_path_obj

    logger.info('=' * 60)
    logger.info('🔢 Computing Retrieval Rankings')
    logger.info('=' * 60)
    logger.info(f'   Descriptors: {descs_path}')
    logger.info(f'   CSV: {csv_path}')
    logger.info(f'   Output: {out_path}')
    logger.info('')

    # 1) Load descriptors
    descs = np.load(descs_path)
    N, D = descs.shape
    logger.info(f'📂 Loaded descriptors: shape {descs.shape}')

    # 2) Load doc_ids from CSV (must match order used in extract_descs)
    doc_ids = []
    with csv_path.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_ids.append(row['image_path'])

    if len(doc_ids) != N:
        raise ValueError(
            f'Mismatch: {len(doc_ids)} doc_ids from {csv_path} '
            f'but {N} descriptors in {descs_path}'
        )

    # 3) Compute cosine similarity matrix
    logger.info('📊 Computing similarity matrix...')
    sim = descs @ descs.T  # [N, N]

    ranks = {}
    for i, qid in enumerate(doc_ids):
        # Exclude self
        scores = sim[i].copy()
        scores[i] = -1.0

        # Sort by similarity descending
        order = np.argsort(-scores)
        ranked_docs = [doc_ids[j] for j in order]
        ranks[qid] = ranked_docs

    # 4) Save ranks.json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(ranks, f, indent=2)

    logger.info(f'✅ Saved ranks for {len(ranks)} queries → {out_path}')
    
    return ranks


if __name__ == '__main__':
    main()
