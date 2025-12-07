# src/utils/pca_whitening.py
import numpy as np
from sklearn.decomposition import PCA


def fit_pca_whitening(descs: np.ndarray, out_dim: int | None = None) -> PCA:
    """
    Fit PCA with whitening on descriptors.

    descs: [N, D]
    out_dim: optional target dimension (<= D).
    """
    n, d = descs.shape
    if out_dim is None or out_dim > d:
        out_dim = d
    pca = PCA(n_components=out_dim, whiten=True, random_state=42)
    pca.fit(descs)
    return pca


def apply_pca_whitening(pca: PCA, descs: np.ndarray) -> np.ndarray:
    """
    Apply fitted PCA-whitening and L2 normalize.

    descs: [N, D]
    returns: [N, out_dim]
    """
    z = pca.transform(descs)
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z / norms
