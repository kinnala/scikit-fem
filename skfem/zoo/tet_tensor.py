"""Create a tetrahedral tensor product mesh.

License
-------

Copyright (c) 2016-2018 Nico Schlömer

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

import numpy as np


def build(x, y, z):
    # Create the vertices.
    nx, ny, nz = len(x), len(y), len(z)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    p = np.array([x, y, z]).T.reshape(-1, 3).T

    # Create the elements.
    a0 = np.add.outer(np.array(range(nx - 1)), nx*np.array(range(ny - 1)))
    a = np.add.outer(a0, nx * ny * np.array(range(nz - 1)))

    elems0 = np.concatenate([a[..., None],
                             a[..., None] + nx,
                             a[..., None] + 1,
                             a[..., None] + nx * ny], axis=3)

    elems0[1::2, 0::2, 0::2, 0] += 1
    elems0[0::2, 1::2, 0::2, 0] += 1
    elems0[0::2, 0::2, 1::2, 0] += 1
    elems0[1::2, 1::2, 1::2, 0] += 1

    elems0[1::2, 0::2, 0::2, 1] += 1
    elems0[0::2, 1::2, 0::2, 1] += 1
    elems0[0::2, 0::2, 1::2, 1] += 1
    elems0[1::2, 1::2, 1::2, 1] += 1

    elems0[1::2, 0::2, 0::2, 2] -= 1
    elems0[0::2, 1::2, 0::2, 2] -= 1
    elems0[0::2, 0::2, 1::2, 2] -= 1
    elems0[1::2, 1::2, 1::2, 2] -= 1

    elems0[1::2, 0::2, 0::2, 3] += 1
    elems0[0::2, 1::2, 0::2, 3] += 1
    elems0[0::2, 0::2, 1::2, 3] += 1
    elems0[1::2, 1::2, 1::2, 3] += 1

    elems1 = np.concatenate([a[..., None] + nx,
                             a[..., None] + 1 + nx,
                             a[..., None] + 1,
                             a[..., None] + 1 + nx + nx*ny], axis=3)

    elems1[1::2, 0::2, 0::2, 0] += 1
    elems1[0::2, 1::2, 0::2, 0] += 1
    elems1[0::2, 0::2, 1::2, 0] += 1
    elems1[1::2, 1::2, 1::2, 0] += 1

    elems1[1::2, 0::2, 0::2, 1] -= 1
    elems1[0::2, 1::2, 0::2, 1] -= 1
    elems1[0::2, 0::2, 1::2, 1] -= 1
    elems1[1::2, 1::2, 1::2, 1] -= 1

    elems1[1::2, 0::2, 0::2, 2] -= 1
    elems1[0::2, 1::2, 0::2, 2] -= 1
    elems1[0::2, 0::2, 1::2, 2] -= 1
    elems1[1::2, 1::2, 1::2, 2] -= 1

    elems1[1::2, 0::2, 0::2, 3] -= 1
    elems1[0::2, 1::2, 0::2, 3] -= 1
    elems1[0::2, 0::2, 1::2, 3] -= 1
    elems1[1::2, 1::2, 1::2, 3] -= 1

    elems2 = np.concatenate([a[..., None] + nx,
                             a[..., None] + 1,
                             a[..., None] + nx*ny,
                             a[..., None] + 1 + nx + nx*ny], axis=3)

    elems2[1::2, 0::2, 0::2, 0] += 1
    elems2[0::2, 1::2, 0::2, 0] += 1
    elems2[0::2, 0::2, 1::2, 0] += 1
    elems2[1::2, 1::2, 1::2, 0] += 1

    elems2[1::2, 0::2, 0::2, 1] -= 1
    elems2[0::2, 1::2, 0::2, 1] -= 1
    elems2[0::2, 0::2, 1::2, 1] -= 1
    elems2[1::2, 1::2, 1::2, 1] -= 1

    elems2[1::2, 0::2, 0::2, 2] += 1
    elems2[0::2, 1::2, 0::2, 2] += 1
    elems2[0::2, 0::2, 1::2, 2] += 1
    elems2[1::2, 1::2, 1::2, 2] += 1

    elems2[1::2, 0::2, 0::2, 3] -= 1
    elems2[0::2, 1::2, 0::2, 3] -= 1
    elems2[0::2, 0::2, 1::2, 3] -= 1
    elems2[1::2, 1::2, 1::2, 3] -= 1

    elems3 = np.concatenate([a[..., None] + nx,
                             a[..., None] + nx*ny,
                             a[..., None] + nx + nx*ny,
                             a[..., None] + 1 + nx + nx*ny], axis=3)

    elems3[1::2, 0::2, 0::2, 0] += 1
    elems3[0::2, 1::2, 0::2, 0] += 1
    elems3[0::2, 0::2, 1::2, 0] += 1
    elems3[1::2, 1::2, 1::2, 0] += 1

    elems3[1::2, 0::2, 0::2, 1] += 1
    elems3[0::2, 1::2, 0::2, 1] += 1
    elems3[0::2, 0::2, 1::2, 1] += 1
    elems3[1::2, 1::2, 1::2, 1] += 1

    elems3[1::2, 0::2, 0::2, 2] += 1
    elems3[0::2, 1::2, 0::2, 2] += 1
    elems3[0::2, 0::2, 1::2, 2] += 1
    elems3[1::2, 1::2, 1::2, 2] += 1

    elems3[1::2, 0::2, 0::2, 3] -= 1
    elems3[0::2, 1::2, 0::2, 3] -= 1
    elems3[0::2, 0::2, 1::2, 3] -= 1
    elems3[1::2, 1::2, 1::2, 3] -= 1

    elems4 = np.concatenate([a[..., None] + 1,
                             a[..., None] + nx * ny,
                             a[..., None] + 1 + nx + nx * ny,
                             a[..., None] + 1 + nx * ny], axis=3)

    elems4[1::2, 0::2, 0::2, 0] -= 1
    elems4[0::2, 1::2, 0::2, 0] -= 1
    elems4[0::2, 0::2, 1::2, 0] -= 1
    elems4[1::2, 1::2, 1::2, 0] -= 1

    elems4[1::2, 0::2, 0::2, 1] += 1
    elems4[0::2, 1::2, 0::2, 1] += 1
    elems4[0::2, 0::2, 1::2, 1] += 1
    elems4[1::2, 1::2, 1::2, 1] += 1

    elems4[1::2, 0::2, 0::2, 2] -= 1
    elems4[0::2, 1::2, 0::2, 2] -= 1
    elems4[0::2, 0::2, 1::2, 2] -= 1
    elems4[1::2, 1::2, 1::2, 2] -= 1

    elems4[1::2, 0::2, 0::2, 3] -= 1
    elems4[0::2, 1::2, 0::2, 3] -= 1
    elems4[0::2, 0::2, 1::2, 3] -= 1
    elems4[1::2, 1::2, 1::2, 3] -= 1

    t = np.vstack([elems0.reshape(-1, 4),
                   elems1.reshape(-1, 4),
                   elems2.reshape(-1, 4),
                   elems3.reshape(-1, 4),
                   elems4.reshape(-1, 4)]).T

    p = np.ascontiguousarray(p)
    t = np.ascontiguousarray(t)

    return p, t
