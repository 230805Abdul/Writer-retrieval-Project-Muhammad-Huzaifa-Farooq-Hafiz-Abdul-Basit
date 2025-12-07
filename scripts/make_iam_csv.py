"""
Create CSV file for IAM dataset.

Usage:
    python -m scripts.make_iam_csv \
        --data-dir data/IAM \
        --out experiments/iam/data/iam_all.csv
"""
import argparse
import csv
import logging
from pathlib import Path

from src.utils.logging_config import setup_script_logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create CSV file for IAM dataset'
    )
    parser.add_argument('--data-dir', type=str, default='data/IAM',
                        help='Root directory containing IAM writer folders')
    parser.add_argument('--out', type=str, default='experiments/iam/data/iam_all.csv',
                        help='Output CSV file path')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs in console')
    return parser.parse_args()


def main(data_dir=None, out_csv=None, log_dir=None, debug=False):
    """
    Create CSV file for IAM dataset.
    
    IAM structure: data/IAM/{writer_id}/{form_id}.png
    e.g., data/IAM/000/a01-000u.png
    
    Can be called from command line or directly with kwargs.
    """
    global logger
    
    # If called without args, parse from command line
    if data_dir is None:
        args = parse_args()
        data_dir = args.data_dir
        out_csv = args.out
        log_dir = args.log_dir
        debug = args.debug

    root = Path(data_dir)
    out_csv = Path(out_csv)
    
    # Setup logging with file output
    log_dir_path = Path(log_dir) if log_dir else out_csv.parent / 'logs'
    logger = setup_script_logging(
        script_name='make_iam_csv',
        log_dir=str(log_dir_path),
        debug=debug
    )

    logger.info('=' * 60)
    logger.info('📝 Creating IAM CSV')
    logger.info('=' * 60)
    logger.info(f'   Data dir: {root}')
    logger.info(f'   Output: {out_csv}')
    logger.info('')

    rows = [('image_path', 'writer_id')]

    # IAM structure: writer folders contain form images
    # Folder name is the writer ID
    writer_folders = sorted([d for d in root.iterdir() if d.is_dir()])
    
    for wid, writer_dir in enumerate(writer_folders):
        for img_path in sorted(writer_dir.glob('*.png')):
            rows.append((str(img_path), wid))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        csv.writer(f).writerows(rows)

    logger.info(f'✅ Saved {len(rows) - 1} entries → {out_csv}')
    logger.info(f'   Found {len(writer_folders)} writers')
    
    return out_csv


if __name__ == '__main__':
    main()
