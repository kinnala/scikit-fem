from typing import Union

from numpy import ndarray

from scipy.sparse import csr_matrix
from .form import Form


def asm(kernel: Form,
        *args, **kwargs) -> Union[ndarray, csr_matrix]:
    return kernel.assemble(*args, **kwargs)
