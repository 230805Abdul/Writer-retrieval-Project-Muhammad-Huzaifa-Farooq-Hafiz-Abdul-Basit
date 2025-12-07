"""
Estimate text quantity (line count) for each IAM image.

Usage:
    python -m scripts.make_iam_text_quantity \
        --csv data/iam/iam_all.csv \
        --out data/iam/iam_text_quantity.json
"""
import argparse
import json
import logging
from pathlib import Path

from src.utils.preprocessing import load_image, binarize_otsu
from src.sampling.text_quantity import estimate_text_quantity


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate text quantity for each IAM image'
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file (image_path,writer_id)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output JSON file path')
    return parser.parse_args()


def main(csv_path=None, out_json=None):
    """
    Estimate text quantity for each IAM image.
    
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
    logger.info('📊 Estimating IAM Text Quantity')
    logger.info('=' * 60)
    logger.info(f'   CSV: {csv_path}')
    logger.info(f'   Output: {out_path}')
    logger.info('')

    quantities = {}

    with csv_path.open('r') as f:
        lines = f.read().strip().split('\n')[1:]

    total = len(lines)
    for idx, line in enumerate(lines):
        img_path, _ = line.split(',')
        img = load_image(img_path)
        bw = binarize_otsu(img)
        q = estimate_text_quantity(bw)
        quantities[img_path] = q['line_count']

        # Progress update every 100 images
        if (idx + 1) % 100 == 0:
            logger.info(f'   Processed {idx + 1}/{total} images...')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(quantities, f, indent=2)

    logger.info(f'✅ Saved {len(quantities)} quantities → {out_path}')
    
    return quantities


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()
