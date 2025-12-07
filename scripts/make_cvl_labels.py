"""
Create a JSON file mapping image paths to writer IDs from a CSV file.

Usage:
    python -m scripts.make_cvl_labels \
        --csv data/cvl/cvl_all.csv \
        --out data/cvl/cvl_labels.json
"""
import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a JSON file mapping image paths to writer IDs'
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file (image_path,writer_id,...)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output JSON file path')
    return parser.parse_args()


def main(csv_path=None, out_json=None):
    """
    Create a JSON file mapping image paths to writer IDs from a CSV file.
    
    Can be called from command line or directly with kwargs.
    """
    # If called without args, parse from command line
    if csv_path is None:
        args = parse_args()
        csv_path = args.csv
        out_json = args.out

    csv_path = Path(csv_path)
    out_path = Path(out_json)

    logger.info('=' * 60)
    logger.info('🏷️  Creating Labels JSON')
    logger.info('=' * 60)
    logger.info(f'   CSV: {csv_path}')
    logger.info(f'   Output: {out_path}')
    logger.info('')

    labels = {}

    with open(csv_path, 'r') as f:
        lines = f.read().strip().split('\n')[1:]

    for line in lines:
        parts = line.split(',')
        img_path = parts[0]
        wid = parts[1]  # writer_id is second column
        labels[img_path] = int(wid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(labels, f, indent=2)

    logger.info(f'✅ Saved {len(labels)} labels → {out_path}')
    
    return labels


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()
