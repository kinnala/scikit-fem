# scikit-fem

[![PyPI version](https://badge.fury.io/py/scikit-fem.svg)](https://badge.fury.io/py/scikit-fem)
[![Build Status](https://travis-ci.com/kinnala/scikit-fem.svg?branch=master)](https://travis-ci.com/kinnala/scikit-fem)
[![Documentation Status](https://readthedocs.org/projects/scikit-fem/badge/?version=latest)](https://scikit-fem.readthedocs.io/en/latest/?badge=latest)
[![Join the chat at https://gitter.im/scikit-fem/Lobby](https://badges.gitter.im/scikit-fem/Lobby.svg)](https://gitter.im/scikit-fem/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Easy to use finite element assemblers and the related tools.

## Yet another finite element library?

This library fills an important gap in the spectrum of finite element codes.
The library is **lightweight** meaning that it has **minimal dependencies**.
It contains **no compiled code**, which means that it's **easy to install** and
use on all platforms that support NumPy.  Despite being fully interpreted, the
code has a reasonably **good performance**.

In the following snippet, we create a tetrahedral mesh with over 1 million
elements and assemble a discrete Laplace operator. On Thinkpad X201, 2010
model, the last line finishes in under 5 seconds.

```
from skfem import *
import numpy as np

mesh = MeshTet.init_tensor(*((np.linspace(0, 1, 60),)*3))
basis = InteriorBasis(mesh, ElementTetP1())

@bilinear_form
def laplace(u, du, v, dv, w):
    return sum(du*dv)

A = asm(laplace, basis)
```

## Installation

The most recent release can be installed simply by `pip install scikit-fem`.
For more recent features, you can clone this repository.

## Getting started

Please see the [user documentation](https://scikit-fem.readthedocs.io).

## Contributors

- Tom Gustafsson (Maintainer)
- [Geordie McBain](https://github.com/gdmcbain)

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*
