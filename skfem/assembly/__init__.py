# -*- coding: utf-8 -*-
"""This module contains tools for performing the finite element assembly. The
basic workflow is the following:

1. Initialize :class:`~skfem.mesh.Mesh` and :class:`~skfem.mesh.Element`.
2. Create :class:`~skfem.assembly.InteriorBasis` and/or
   :class:`~skfem.assembly.FacetBasis` objects.
3. Define the forms using :class:`~skfem.assembly.BilinearForm`,
   :class:`~skfem.assembly.LinearForm`, or :class:`~skfem.assembly.Functional`.
4. Assemble using :func:`~skfem.assembly.asm`.

This is demostrated in the following snippet:

>>> from skfem import *
>>> m = MeshTri()
>>> e = ElementTriP1()
>>> basis = InteriorBasis(m, e)
>>> form_a = BilinearForm(lambda u, v, w: u * v)
>>> asm(form_a, basis)
<4x4 sparse matrix of type '<class 'numpy.float64'>'
	with 14 stored elements in Compressed Sparse Row format>

The above snippet assembles the mass matrix corresponding
to the bilinear form

.. math::

    a(u,v) = \int_0^1 \int_0^1 u(x,y)\,v(x,y) \,\mathrm{d}x \,\mathrm{d}y

using two triangular elements and piecewise linear basis functions.

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
