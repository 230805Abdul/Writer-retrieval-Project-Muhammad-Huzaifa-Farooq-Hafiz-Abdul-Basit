"""
Create CSV files for the CVL dataset.

CVL Database structure (original):
    data/CVL/
        trainset/
            pages/*.tif     (27 writers × 7 pages = 189 pages)
            lines/*.tif     (line-level images for training writers)
        testset/
            pages/*.tif     (283 writers × 5 pages = 1415 pages)
            lines/*.tif     (line-level images for test writers)

Filename format: XXXX-Y.tif (writer_id-page_number)

The trainset/testset split is NON-OVERLAPPING by design:
- Training: Learn generalizable handwriting features from 27 writers
- Testing: Evaluate retrieval on 283 completely unseen writers
"""

import csv
import logging
from pathlib import Path

from src.utils.logging_config import setup_script_logging


logger = logging.getLogger(__name__)


def scan_pages(pages_dir):
    """Scan page-level images from a pages/ directory."""
    if not pages_dir.exists():
        logger.warning(f'{pages_dir} does not exist!')
        return []
    
    rows = []
    for ext in ['*.tif', '*.TIF', '*.png', '*.PNG', '*.jpg', '*.JPG']:
        for img_path in sorted(pages_dir.glob(ext)):
            name = img_path.stem  # 'XXXX-Y'
            parts = name.split('-')
            if len(parts) < 2:
                continue
            try:
                writer_id = int(parts[0])
                page_id = int(parts[1])
                rows.append((str(img_path.resolve()), writer_id, page_id))
            except ValueError:
                continue
    return rows


def scan_lines(lines_dir):
    """Scan line-level images from a lines/ directory (hierarchical: lines/WRITER/*.tif)."""
    if not lines_dir.exists():
        logger.warning(f'{lines_dir} does not exist!')
        return []
    
    rows = []
    # Lines are in subdirectories by writer: lines/0052/0052-1-0.tif
    for writer_subdir in sorted(lines_dir.iterdir()):
        if not writer_subdir.is_dir():
            continue
        for ext in ['*.tif', '*.TIF', '*.png', '*.PNG', '*.jpg', '*.JPG']:
            for img_path in sorted(writer_subdir.glob(ext)):
                name = img_path.stem  # 'XXXX-Y-Z'
                parts = name.split('-')
                if len(parts) < 3:
                    continue
                try:
                    writer_id = int(parts[0])
                    page_id = int(parts[1])
                    line_id = int(parts[2])
                    rows.append((str(img_path.resolve()), writer_id, page_id, line_id))
                except ValueError:
                    continue
    return rows


def scan_words(words_dir):
    """Scan word-level images from a words/ directory (hierarchical: words/WRITER/*.tif)."""
    if not words_dir.exists():
        logger.warning(f'{words_dir} does not exist!')
        return []
    
    rows = []
    # Words are in subdirectories by writer: words/0052/0052-1-0-0-Imagine.tif
    for writer_subdir in sorted(words_dir.iterdir()):
        if not writer_subdir.is_dir():
            continue
        for ext in ['*.tif', '*.TIF', '*.png', '*.PNG', '*.jpg', '*.JPG']:
            for img_path in sorted(writer_subdir.glob(ext)):
                name = img_path.stem  # 'XXXX-Y-Z-W-TEXT'
                parts = name.split('-')
                if len(parts) < 4:
                    continue
                try:
                    writer_id = int(parts[0])
                    page_id = int(parts[1])
                    line_id = int(parts[2])
                    word_id = int(parts[3])
                    rows.append((str(img_path.resolve()), writer_id, page_id, line_id, word_id))
                except ValueError:
                    continue
    return rows


def main(data_dir=None, out_train=None, out_test=None, level='page', include_lines_for_train=False,
         log_dir=None, debug=False):
    """
    Create train and test CSV files for CVL dataset.
    
    Args:
        data_dir: Path to CVL directory (contains trainset/ and testset/)
        out_train: Path to save training CSV
        out_test: Path to save test CSV  
        level: 'page', 'line', or 'word' for test set
        include_lines_for_train: If True, also include line-level images in training
        log_dir: Directory for log files
        debug: Show debug logs in console
    """
    global logger
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'CVL'
    else:
        data_dir = Path(data_dir)

    if out_train is None:
        out_train = Path(__file__).parent.parent / 'experiments' / 'cvl' / 'data' / 'cvl_train.csv'
    else:
        out_train = Path(out_train)
        
    if out_test is None:
        out_test = Path(__file__).parent.parent / 'experiments' / 'cvl' / 'data' / 'cvl_test.csv'
    else:
        out_test = Path(out_test)

    # Setup logging with file output
    log_dir_path = Path(log_dir) if log_dir else out_train.parent / 'logs'
    logger = setup_script_logging(
        script_name='make_cvl_csv',
        log_dir=str(log_dir_path),
        debug=debug
    )

    trainset_dir = data_dir / 'trainset'
    testset_dir = data_dir / 'testset'

    logger.info(f'📂 Scanning CVL dataset from {data_dir}...')
    logger.info(f'   trainset: {trainset_dir}')
    logger.info(f'   testset: {testset_dir}')
    logger.info(f'   level: {level}')

    # === TRAINING DATA ===
    # Use the same level for training as for test
    if level == 'page':
        train_rows = scan_pages(trainset_dir / 'pages')
        logger.info(f'   Training: {len(train_rows)} pages')
    elif level == 'line':
        train_rows_full = scan_lines(trainset_dir / 'lines')
        train_rows = [(path, wid, pid) for path, wid, pid, lid in train_rows_full]
        logger.info(f'   Training: {len(train_rows)} lines')
    elif level == 'word':
        train_rows_full = scan_words(trainset_dir / 'words')
        train_rows = [(path, wid, pid) for path, wid, pid, lid, wdid in train_rows_full]
        logger.info(f'   Training: {len(train_rows)} words')
    else:
        raise ValueError(f"Unknown level: {level}. Use 'page', 'line', or 'word'.")

    if include_lines_for_train and level == 'page':
        train_lines = scan_lines(trainset_dir / 'lines')
        train_lines_simple = [(path, wid, pid) for path, wid, pid, lid in train_lines]
        train_rows = train_rows + train_lines_simple
        logger.info(f'   + {len(train_lines)} lines = {len(train_rows)} total')

    # === TEST DATA ===
    if level == 'page':
        test_rows = scan_pages(testset_dir / 'pages')
        logger.info(f'   Test: {len(test_rows)} pages')
    elif level == 'line':
        test_rows_full = scan_lines(testset_dir / 'lines')
        test_rows = [(path, wid, pid) for path, wid, pid, lid in test_rows_full]
        logger.info(f'   Test: {len(test_rows)} lines')
    elif level == 'word':
        test_rows_full = scan_words(testset_dir / 'words')
        test_rows = [(path, wid, pid) for path, wid, pid, lid, wdid in test_rows_full]
        logger.info(f'   Test: {len(test_rows)} words')

    if not train_rows:
        logger.error('No training images found!')
        return None, None
        
    if not test_rows:
        logger.error('No test images found!')
        return None, None

    # Write CSVs
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    header = ['image_path', 'writer_id', 'page_id']
    
    with open(out_train, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)
    
    with open(out_test, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_rows)

    # Statistics
    train_writers = set(row[1] for row in train_rows)
    test_writers = set(row[1] for row in test_rows)
    overlap = train_writers & test_writers
    
    logger.info('')
    logger.info('✅ Saved:')
    logger.info(f'   Train: {out_train} ({len(train_rows)} samples, {len(train_writers)} writers)')
    logger.info(f'   Test: {out_test} ({len(test_rows)} samples, {len(test_writers)} writers)')
    logger.info(f'   Writer overlap: {len(overlap)} (should be 0 for proper evaluation)')
    
    return out_train, out_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create CVL dataset CSVs')
    parser.add_argument('--data-dir', type=str, default='data/CVL',
                        help='Path to CVL directory')
    parser.add_argument('--out-train', type=str, default=None,
                        help='Output train CSV path')
    parser.add_argument('--out-test', type=str, default=None,
                        help='Output test CSV path')
    parser.add_argument('--level', type=str, default='page', choices=['page', 'line', 'word'],
                        help='Data level for train/test set (default: page)')
    parser.add_argument('--include-lines-for-train', action='store_true',
                        help='Include line-level images in training data (only for page level)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs in console')

    args = parser.parse_args()
    main(
        data_dir=args.data_dir, 
        out_train=args.out_train, 
        out_test=args.out_test,
        level=args.level,
        include_lines_for_train=args.include_lines_for_train,
        log_dir=args.log_dir,
        debug=args.debug
    )
