![scikit-fem](https://github.com/kinnala/scikit-fem/blob/master/logo.png?raw=true)

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

## Examples

In the following snippet, we create a tetrahedral mesh with over 1 million
elements and assemble a discrete Laplace operator, all in just a few seconds.

```python
from skfem import *
import numpy as np

mesh = MeshTet.init_tensor(*((np.linspace(0, 1, 60),) * 3))
basis = InteriorBasis(mesh, ElementTetP1())

@bilinear_form
def laplace(u, du, v, dv, w):
    return sum(du * dv)

A = asm(laplace, basis)
```

More examples can be found in the [documentation](https://kinnala.github.io/scikit-fem-docs/learning.html).

## Installation

The most recent release can be installed simply by `pip install scikit-fem`.

For more cutting edge features, you can clone this repository.

## Getting started

If you installed a (recent) release using `pip`, you can find the accompanying documentation under [Releases](https://github.com/kinnala/scikit-fem/releases)

The latest user documentation corresponding to the master branch can be found [online](https://kinnala.github.io/scikit-fem-docs).

## Acknowledgements

This project was started while working under a grant from the [Finnish Cultural Foundation](https://skr.fi/). The approach used in the finite element assembly has been inspired by the [work of A. Hannukainen and M. Juntunen](https://au.mathworks.com/matlabcentral/fileexchange/36108-hjfem_lite).

## In literature

The library has been used in the preparation of the following scientific works:

- Gustafsson, T., Stenberg, R., & Videman, J. (2019). On Nitsche's method for elastic contact problems. arXiv preprint [arXiv:1902.09312](https://arxiv.org/abs/1902.09312).
- Gustafsson, T., Stenberg, R., & Videman, J. (2019). Error analysis of Nitsche's mortar method. Numerische Mathematik. [Open access](https://link.springer.com/article/10.1007/s00211-019-01039-5).
- Gustafsson, T., Stenberg, R., & Videman, J. (2018). Nitsche's method for unilateral contact problems. arXiv preprint [arXiv:1805.04283](https://arxiv.org/abs/1805.04283).
- Gustafsson, T., Stenberg, R. & Videman, J. (2018). A posteriori estimates for conforming Kirchhoff plate elements. SIAM Journal on Scientific Computing, 40.3, A1386-A1407. arXiv preprint [arXiv:1707.08396](https://arxiv.org/abs/1707.08396).
- Gustafsson, T., Rajagopal, K. R., Stenberg, R., & Videman, J. (2018). An adaptive finite element method for the inequality-constrained Reynolds equation. Computer Methods in Applied Mechanics and Engineering, 336, 156-170. arXiv preprint [arXiv:1711.04274](https://arxiv.org/abs/1711.04274).
- Gustafsson, T., Stenberg, R., & Videman, J. (2018). A stabilised finite element method for the plate obstacle problem. BIT Numerical Mathematics, 1-28. arXiv preprint [arXiv:1711.04166](https://arxiv.org/abs/1711.04166).
- Gustafsson, T., Stenberg, R., & Videman, J. (2017). Nitsche’s Method for the Obstacle Problem of Clamped Kirchhoff Plates. In European Conference on Numerical Mathematics and Advanced Applications (pp. 407-415). Springer.
- Gustafsson, T., Stenberg, R., & Videman, J. (2017). A posteriori analysis of classical plate elements. Rakenteiden Mekaniikka, 50(3), 141-145. [Open access](https://rakenteidenmekaniikka.journal.fi/article/view/65004/26450).

In case you want to cite the library, you can use the DOI provided by [Zenodo](https://zenodo.org/badge/latestdoi/115345426).

## Contributors

- Tom Gustafsson (Author)
- [Geordie McBain](https://github.com/gdmcbain)

*By contributing code to scikit-fem, you are agreeing to release it under BSD-3-Clause, see LICENSE.md.*
