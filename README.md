![scikit-fem](https://github.com/kinnala/scikit-fem/blob/master/skfemlogo.png?raw=true)

[![PyPI version](https://badge.fury.io/py/scikit-fem.svg)](https://badge.fury.io/py/scikit-fem)
[![Build Status](https://travis-ci.com/kinnala/scikit-fem.svg?branch=master)](https://travis-ci.com/kinnala/scikit-fem)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://kinnala.github.io/scikit-fem-docs)
[![Join the chat at https://gitter.im/scikit-fem/Lobby](https://badges.gitter.im/scikit-fem/Lobby.svg)](https://gitter.im/scikit-fem/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![DOI](https://zenodo.org/badge/115345426.svg)](https://zenodo.org/badge/latestdoi/115345426)

Easy to use finite element assemblers and the related tools.

## Features

This library fills an important gap in the spectrum of finite element codes.
The library is *lightweight* meaning that it has *minimal dependencies*.
It contains *no compiled code* meaning that it's *easy to install* and
use on all platforms that support NumPy.  Despite being fully interpreted, the
code has a reasonably *good performance*.

In the following snippet, we create a tetrahedral mesh with over 1 million
elements and assemble a discrete Laplace operator, all in just a few seconds.

```python
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

For more cutting edge features, you can clone this repository.

## Getting started

Please see the [user documentation](https://kinnala.github.io/scikit-fem-docs) for examples.

## Acknowledgements

This project was started while working under a grant from the [Finnish Cultural Foundation](https://skr.fi/). The approach used in the finite element assembly has been inspired by the [work of A. Hannukainen and M. Juntunen](https://au.mathworks.com/matlabcentral/fileexchange/36108-hjfem_lite).

## Contributors

- Tom Gustafsson (Author)
- [Geordie McBain](https://github.com/gdmcbain)

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*
