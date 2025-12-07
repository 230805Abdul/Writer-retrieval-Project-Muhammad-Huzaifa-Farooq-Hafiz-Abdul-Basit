# src/evaluation/per_bin_csv.py

import argparse
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)


def load_json(path: str | Path):
    with open(path, 'r') as f:
        return json.load(f)


def ap_at_k(gt_writer: str, ranked_docs: List[str],
            labels: Dict[str, str], k: int | None = None) -> float:
    """
    Average Precision for a single query.
    If k is not None, only first k results are considered.
    """
    if k is not None:
        ranked_docs = ranked_docs[:k]

    num_rel = 0
    ap_sum = 0.0
    for i, doc_id in enumerate(ranked_docs, start=1):
        if labels.get(doc_id) == gt_writer:
            num_rel += 1
            ap_sum += num_rel / i

    if num_rel == 0:
        return 0.0
    return ap_sum / num_rel


def get_bin(qty: float) -> int:
    """
    Map a quantity value (e.g. #lines) to a bin index.
    Change thresholds as you like.
    """
    if qty <= 1:
        return 0        # 1 line
    if qty <= 3:
        return 1        # 2-3 lines
    if qty <= 5:
        return 2        # 4-5 lines
    return 3            # >= 6 lines


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-bin CSV for writer retrieval results.')
    parser.add_argument('--ranks', type=str, required=True,
                        help='JSON: query_id -> ordered list of gallery ids')
    parser.add_argument('--labels', type=str, required=True,
                        help='JSON: doc_id -> writer_id')
    parser.add_argument('--text-quantity', type=str, required=True,
                        help='JSON: query_id -> quantity (e.g. #lines)')
    parser.add_argument('--out-csv', type=str, required=True,
                        help='Output CSV path')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files (default: output dir/logs)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs in console (always saved to file)')
    args = parser.parse_args()

    ranks = load_json(args.ranks)
    labels = load_json(args.labels)
    text_qty = load_json(args.text_quantity)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'query_id',
        'writer_id',
        'quantity',
        'bin',
        'ap_full',
        'ap_10',
        'top1_hit',
        'top5_hit',
        'top10_hit',
    ]

    # 1) write per-query rows
    per_bin_stats = {}  # bin -> list of APs
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q_id, ranked_docs in ranks.items():
            if q_id not in labels:
                continue  # skip if missing writer label
            if q_id not in text_qty:
                continue  # skip if no quantity info

            qty = float(text_qty[q_id])
            b = get_bin(qty)
            gt_writer = labels[q_id]

            ap_full = ap_at_k(gt_writer, ranked_docs, labels, k=None)
            ap_10 = ap_at_k(gt_writer, ranked_docs, labels, k=10)

            top1_hit = int(labels.get(ranked_docs[0], None) == gt_writer) \
                if ranked_docs else 0
            top5_hit = int(any(labels.get(d) == gt_writer
                               for d in ranked_docs[:5]))
            top10_hit = int(any(labels.get(d) == gt_writer
                                for d in ranked_docs[:10]))

            writer.writerow({
                'query_id': q_id,
                'writer_id': gt_writer,
                'quantity': qty,
                'bin': b,
                'ap_full': ap_full,
                'ap_10': ap_10,
                'top1_hit': top1_hit,
                'top5_hit': top5_hit,
                'top10_hit': top10_hit,
            })

            per_bin_stats.setdefault(b, []).append(ap_full)

    # 2) also log simple per-bin summary
    logger.info('📊 Per-bin mAP (using ap_full):')
    for b in sorted(per_bin_stats.keys()):
        vals = per_bin_stats[b]
        m = sum(vals) / len(vals) if vals else 0.0
        logger.info(f'  Bin {b}: n={len(vals)}, mAP={m:.4f}')
    
    logger.info(f'✅ Output saved to {out_path}')


if __name__ == '__main__':
    main()
