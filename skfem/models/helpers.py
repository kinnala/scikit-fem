"""Helper functions for defining forms."""

import numpy as np


def div(dw):
    """Divergence of vector."""
    return np.einsum('ii...', dw)

def ddot(A, B):
    """Double dot product."""
    return np.einsum('ij...,ij...', A, B)

def trace(T):
    """Trace of matrix."""
    return np.einsum('ii...', T)

def transpose(T):
    """Transpose of matrix."""
    return np.einsum('ij...->ji...', T)

def eye(w, n):
    """Create diagonal matrix with w on diagonal."""
    return np.array([[w if i==j else 0.0*w for i in range(n)] for j in range(n)])
