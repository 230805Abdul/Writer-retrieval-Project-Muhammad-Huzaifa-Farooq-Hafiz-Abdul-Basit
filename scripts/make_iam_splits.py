"""
Split IAM dataset using Aachen/RWTH writer-independent protocol.

The IAM-C split (commonly used for writer retrieval) defines:
- Train: 6161 lines from 283 writers
- Valid: 940 lines from 43 writers  
- Test: 1861 lines from 128 writers

This script downloads the official split files from:
https://github.com/shonenkov/IAM-Splitting/tree/master/IAM-C

For page-level writer retrieval, we extract unique form IDs from line-level
splits and map them to our local data structure.

Usage:
    python -m scripts.make_iam_splits \
        --data-dir data/IAM \
        --out-dir data \
        --download  # Downloads official split files
"""
import argparse
import csv
import logging
import urllib.request
from pathlib import Path
from collections import defaultdict


logger = logging.getLogger(__name__)

# Official IAM-C split URLs
SPLIT_URLS = {
    'train': 'https://raw.githubusercontent.com/shonenkov/IAM-Splitting/master/IAM-C/train.txt',
    'valid': 'https://raw.githubusercontent.com/shonenkov/IAM-Splitting/master/IAM-C/valid.txt',
    'test': 'https://raw.githubusercontent.com/shonenkov/IAM-Splitting/master/IAM-C/test.txt',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split IAM dataset using Aachen/RWTH protocol'
    )
    parser.add_argument('--data-dir', type=str, default='data/IAM',
                        help='Root directory containing IAM writer folders')
    parser.add_argument('--out-dir', type=str, default='data',
                        help='Output directory for CSV files')
    parser.add_argument('--download', action='store_true',
                        help='Download official split files from GitHub')
    parser.add_argument('--split-dir', type=str, default=None,
                        help='Directory containing local split files (train.txt, valid.txt, test.txt)')
    return parser.parse_args()


def download_splits(out_dir: Path) -> dict:
    """Download official IAM-C split files."""
    split_dir = out_dir / 'iam_splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    
    split_files = {}
    for name, url in SPLIT_URLS.items():
        out_path = split_dir / f'{name}.txt'
        if not out_path.exists():
            logger.info(f'   Downloading {name}.txt...')
            urllib.request.urlretrieve(url, out_path)
        split_files[name] = out_path
    
    return split_files


def load_split_lines(split_file: Path) -> set:
    """Load line IDs from a split file."""
    with open(split_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def line_id_to_form_id(line_id: str) -> str:
    """
    Convert line ID to form ID.
    
    Line ID format: a01-000u-00
    Form ID format: a01-000u
    """
    parts = line_id.rsplit('-', 1)
    return parts[0] if len(parts) == 2 else line_id


def build_form_to_writer_map(data_dir: Path) -> dict:
    """
    Build mapping from form ID to writer ID.
    
    IAM structure: data/IAM/{writer_folder}/{form_id}.png
    Writer folder names are like: 000, 001, 002...
    Form IDs in filenames are like: a01-000u.png
    """
    form_to_writer = {}
    form_to_path = {}
    
    writer_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    for writer_folder in writer_folders:
        writer_id = writer_folder.name  # e.g., "000"
        
        for img_path in writer_folder.glob('*.png'):
            # Form ID is the filename without extension
            form_id = img_path.stem  # e.g., "a01-000u"
            form_to_writer[form_id] = writer_id
            form_to_path[form_id] = img_path
    
    return form_to_writer, form_to_path


def main(data_dir=None, out_dir=None, download=True, split_dir=None):
    """
    Split IAM dataset using Aachen/RWTH protocol.
    """
    if data_dir is None:
        args = parse_args()
        data_dir = args.data_dir
        out_dir = args.out_dir
        download = args.download
        split_dir = args.split_dir

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    
    logger.info('=' * 60)
    logger.info('✂️  Splitting IAM Dataset (Aachen/RWTH Protocol)')
    logger.info('=' * 60)
    logger.info(f'   Data dir: {data_dir}')
    logger.info(f'   Output dir: {out_dir}')
    logger.info('')
    
    # Get split files
    if split_dir:
        split_files = {
            'train': Path(split_dir) / 'train.txt',
            'valid': Path(split_dir) / 'valid.txt',
            'test': Path(split_dir) / 'test.txt',
        }
    elif download:
        logger.info('📥 Downloading official IAM-C split files...')
        split_files = download_splits(out_dir)
    else:
        raise ValueError('Either --download or --split-dir must be specified')
    
    # Load line IDs for each split
    logger.info('')
    logger.info('📋 Loading split definitions...')
    split_lines = {}
    for name, path in split_files.items():
        split_lines[name] = load_split_lines(path)
        logger.info(f'   {name}: {len(split_lines[name])} lines')
    
    # Extract unique form IDs from each split
    logger.info('')
    logger.info('📄 Extracting form IDs from line IDs...')
    split_forms = {}
    for name, lines in split_lines.items():
        forms = set(line_id_to_form_id(lid) for lid in lines)
        split_forms[name] = forms
        logger.info(f'   {name}: {len(forms)} unique forms')
    
    # Build form → writer mapping from local data
    logger.info('')
    logger.info('🗺️  Building form-to-writer mapping...')
    form_to_writer, form_to_path = build_form_to_writer_map(data_dir)
    logger.info(f'   Found {len(form_to_writer)} forms in local data')
    
    # Check for missing forms
    all_split_forms = set().union(*split_forms.values())
    missing = all_split_forms - set(form_to_writer.keys())
    if missing:
        logger.warning(f'   ⚠️  {len(missing)} forms in splits not found in local data')
        # Show a few examples
        for form in list(missing)[:5]:
            logger.warning(f'      Missing: {form}')
    
    # Create CSV files for each split
    logger.info('')
    logger.info('💾 Creating split CSV files...')
    
    split_stats = {}
    for name, forms in split_forms.items():
        rows = [('image_path', 'writer_id')]
        writers = set()
        
        for form_id in sorted(forms):
            if form_id in form_to_writer:
                writer_id = form_to_writer[form_id]
                img_path = form_to_path[form_id]
                # Use absolute path to avoid relative path issues
                abs_path = img_path.resolve()
                rows.append((str(abs_path), writer_id))
                writers.add(writer_id)
        
        # Save CSV
        out_path = out_dir / f'iam_{name}.csv'
        with out_path.open('w', newline='') as f:
            csv.writer(f).writerows(rows)
        
        split_stats[name] = {
            'pages': len(rows) - 1,
            'writers': len(writers),
        }
        logger.info(f'   ✅ {name}: {split_stats[name]["pages"]} pages, '
                   f'{split_stats[name]["writers"]} writers → {out_path}')
    
    # Verify writer disjointness
    logger.info('')
    logger.info('🔍 Verifying writer disjointness...')
    
    train_writers = set()
    valid_writers = set()
    test_writers = set()
    
    for form_id in split_forms['train']:
        if form_id in form_to_writer:
            train_writers.add(form_to_writer[form_id])
    for form_id in split_forms['valid']:
        if form_id in form_to_writer:
            valid_writers.add(form_to_writer[form_id])
    for form_id in split_forms['test']:
        if form_id in form_to_writer:
            test_writers.add(form_to_writer[form_id])
    
    train_valid_overlap = train_writers & valid_writers
    train_test_overlap = train_writers & test_writers
    valid_test_overlap = valid_writers & test_writers
    
    if train_valid_overlap:
        logger.warning(f'   ⚠️  Train-Valid overlap: {len(train_valid_overlap)} writers')
    else:
        logger.info('   ✅ Train-Valid: NO overlap')
    
    if train_test_overlap:
        logger.warning(f'   ⚠️  Train-Test overlap: {len(train_test_overlap)} writers')
    else:
        logger.info('   ✅ Train-Test: NO overlap')
    
    if valid_test_overlap:
        logger.warning(f'   ⚠️  Valid-Test overlap: {len(valid_test_overlap)} writers')
    else:
        logger.info('   ✅ Valid-Test: NO overlap')
    
    # Summary
    logger.info('')
    logger.info('=' * 60)
    logger.info('📊 Summary (Aachen/RWTH Protocol)')
    logger.info('=' * 60)
    for name, stats in split_stats.items():
        logger.info(f'   {name.capitalize():8s}: {stats["pages"]:4d} pages from {stats["writers"]:3d} writers')
    logger.info('')
    logger.info('   Protocol: IAM-C (Writer-Independent)')
    logger.info('   Reference: shonenkov/IAM-Splitting')
    logger.info('=' * 60)
    
    return {
        'train': out_dir / 'iam_train.csv',
        'valid': out_dir / 'iam_valid.csv', 
        'test': out_dir / 'iam_test.csv',
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    main()
