"""
Estimate text quantity (line count) for each image in the dataset.

Usage:
    python -m scripts.make_cvl_text_quantity \
        --csv data/cvl/cvl_all.csv \
        --out data/cvl/cvl_text_quantity.json
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
        description='Estimate text quantity for each image'
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file (image_path,writer_id,...)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output JSON file path')
    return parser.parse_args()


def main(csv_path=None, out_json=None):
    """
    Estimate text quantity for each image.
    
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
    logger.info('📊 Estimating Text Quantity')
    logger.info('=' * 60)
    logger.info(f'   CSV: {csv_path}')
    logger.info(f'   Output: {out_path}')
    logger.info('')

    quantities = {}

    with open(csv_path, 'r') as f:
        lines = f.read().strip().split('\n')[1:]

    total = len(lines)
    for idx, line in enumerate(lines):
        parts = line.split(',')
        img_path = parts[0]  # image_path is first column
        img = load_image(img_path)
        bw = binarize_otsu(img)
        quantity = estimate_text_quantity(bw)

        # use line_count as "quantity"
        quantities[img_path] = quantity['line_count']

        # Progress update every 100 images
        if (idx + 1) % 100 == 0:
            logger.info(f'   Processed {idx + 1}/{total} images...')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
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
