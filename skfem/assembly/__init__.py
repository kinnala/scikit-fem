# -*- coding: utf-8 -*-
"""The basic workflow for finite element assembly is the following:

1. Initialize :class:`~skfem.mesh.Mesh` and :class:`~skfem.mesh.Element`.
2. Create :class:`~skfem.assembly.InteriorBasis` and/or
   :class:`~skfem.assembly.FacetBasis` objects.
3. Define the forms using the decorators :class:`~skfem.assembly.BilinearForm`,
   :class:`~skfem.assembly.LinearForm`, and :class:`~skfem.assembly.Functional`.
4. Assemble using :func:`~skfem.assembly.asm`.

"""

from typing import Union

from numpy import ndarray

from scipy.sparse import csr_matrix

from .basis import Basis, InteriorBasis, FacetBasis
from .dofs import Dofs
from .form import Form, BilinearForm, LinearForm, Functional,\
    bilinear_form, linear_form, functional


def asm(form: Form,
        *args, **kwargs) -> Union[ndarray, csr_matrix]:
    return form.assemble(*args, **kwargs)
