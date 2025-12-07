# src/models/__init__.py
"""
Model architectures for writer retrieval.
"""
from .e2e_writer_net import EndToEndWriterNet, GeMAggregator, AttentionAggregator

__all__ = ['EndToEndWriterNet', 'GeMAggregator', 'AttentionAggregator']
