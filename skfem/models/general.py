"""
Bilinear and linear forms too general to put into a specific model.
"""
from skfem.assembly import bilinear_form

import numpy as np


@bilinear_form
def divergence(u, du, q, dq, w):
    return np.einsum('ii...', du) * q
