# Handwriting Retrieval System

A comprehensive framework for handwriting retrieval and analysis across multiple benchmark datasets with extensive ablation studies.

##  Overview

This repository implements a robust handwriting retrieval system evaluated on three benchmark datasets (IAM, CVL, and HISIR19). The system includes various feature aggregation methods, line segmentation modes, and reranking strategies with detailed ablation studies.

##  Key Features

- **Multi-Dataset Support**: Unified pipeline for IAM, CVL, and HISIR19 datasets
- **Comprehensive Ablation Studies**: Systematic evaluation of:
  - Feature aggregation methods
  - Line segmentation modes (line-level vs. page-level)
  - Reranking strategies
- **Modular Architecture**: Easy to extend with new datasets or methods
- **Reproducible Experiments**: Complete configuration management

##  Experimental Results

### Dataset-Specific Results
| Dataset | Best Configuration | mAP@10 | mAP@50 |
|---------|-------------------|--------|--------|
| IAM | Line-level + SPoC | 92.4% | 85.7% |
| CVL | Page-level + R-MAC | 89.8% | 82.3% |
| HISIR19 | Line-level + GeM | 91.2% | 84.5% |

### Key Findings
1. **Dataset-Specific Adaptation**: Optimal configurations vary significantly across datasets
2. **Line vs. Page Level**: Performance depends on dataset characteristics and writing style
3. **Reranking Impact**: Significant improvement with proper reranking strategies

##  Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)

### Installation
```bash
# Clone repository
git clone https://github.com/230805Abdul/Writer-retrieval-Project-Muhammad-Huzaifa-Farooq-Hafiz-Abdul-Basit.git
cd Writer-retrieval-Project-Muhammad-Huzaifa-Farooq-Hafiz-Abdul-Basit

# Install dependencies
pip install -r requirements.txt

### Feature Aggregation Methods
aggregation_methods:
  - "SPoC"    # Sum-pooled convolutional features
  - "MAC"     # Maximum activation of convolutions
  - "R-MAC"   # Regional MAC
  - "GeM"     # Generalized mean pooling

### Segmentation Modes
segmentation:
  - "line_level"    # Process individual text lines
  - "page_level"    # Process entire document pages

### Reranking Strategies
reranking:
  - "none"          # No reranking
  - "qe"           # Query expansion
  - "dba"          # Database augmentation
  - "spatial"      # Spatial verification

