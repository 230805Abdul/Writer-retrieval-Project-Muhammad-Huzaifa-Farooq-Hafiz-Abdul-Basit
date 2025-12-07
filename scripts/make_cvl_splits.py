"""
Split a dataset into train/test by writer.

NOTE: For CVL dataset, use make_cvl_csv.py instead! 
      CVL already has a proper trainset/testset split (27/283 writers).
      This script creates OVERLAPPING splits which is INCORRECT for
      evaluating writer retrieval.

This script is only useful for:
- Creating validation splits within the training set
- Splitting other datasets that don't have predefined splits

Usage:
    python -m scripts.make_cvl_splits \\
        --csv experiments/cvl/data/cvl_train.csv \\
        --train-out experiments/cvl/data/cvl_train_subset.csv \\
        --test-out experiments/cvl/data/cvl_val.csv \\
        --train-ratio 0.9
"""
import argparse
import csv
import logging
from pathlib import Path
from collections import defaultdict


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split CVL dataset into train/test'
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file (image_path,writer_id,...)')
    parser.add_argument('--train-out', type=str, required=True,
                        help='Output path for train CSV')
    parser.add_argument('--test-out', type=str, required=True,
                        help='Output path for test CSV')
    parser.add_argument('--train-ratio', type=float, default=0.5,
                        help='Fraction of samples per writer for training (default: 0.5)')
    return parser.parse_args()


def main(csv_in=None, train_out=None, test_out=None, train_ratio=0.5):
    """
    Split dataset into train/test by writer.
    
    Can be called from command line or directly with kwargs.
    """
    # If called without args, parse from command line
    if csv_in is None:
        args = parse_args()
        csv_in = args.csv
        train_out = args.train_out
        test_out = args.test_out
        train_ratio = args.train_ratio

    csv_in = Path(csv_in)
    train_out = Path(train_out)
    test_out = Path(test_out)

    logger.info('=' * 60)
    logger.info('✂️  Splitting Dataset')
    logger.info('=' * 60)
    logger.info(f'   Input CSV: {csv_in}')
    logger.info(f'   Train output: {train_out}')
    logger.info(f'   Test output: {test_out}')
    logger.info(f'   Train ratio: {train_ratio}')
    logger.info('')

    # Read CSV and detect header
    with open(csv_in, 'r') as f:
        lines = f.read().strip().split('\n')
    
    header = lines[0]
    data_lines = lines[1:]
    
    # Group by writer_id (second column)
    writer_samples = defaultdict(list)
    for line in data_lines:
        parts = line.split(',')
        img_path = parts[0]
        wid = int(parts[1])  # writer_id is second column
        writer_samples[wid].append(line)

    train_rows = [header]
    test_rows = [header]

    for wid, samples in sorted(writer_samples.items()):
        samples = sorted(samples)
        k = len(samples)
        split = int(k * train_ratio)
        
        # Ensure at least 1 sample in each split if possible
        if split == 0 and k > 1:
            split = 1
        if split == k and k > 1:
            split = k - 1

        train_rows.extend(samples[:split])
        test_rows.extend(samples[split:])

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    with open(train_out, 'w', newline='') as f:
        f.write('\n'.join(train_rows))
    with open(test_out, 'w', newline='') as f:
        f.write('\n'.join(test_rows))

    logger.info(f'✅ Train: {len(train_rows)-1} samples → {train_out}')
    logger.info(f'✅ Test: {len(test_rows)-1} samples → {test_out}')
    logger.info(f'   Writers in train: {len(writer_samples)}')
    logger.info(f'   Writers in test: {len(writer_samples)} (100% overlap)')
    
    return train_out, test_out


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()
