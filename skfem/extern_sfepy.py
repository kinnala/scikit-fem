"""

This file contains code borrowed from SfePy project (https://github.com/sfepy/sfepy/).

Copyright (c) 2007 - 2014 SfePy Developers.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the SfePy nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as nm
from skfem.mesh import MeshTri, MeshTet, MeshQuad

def assert_(condition, msg='assertion failed!'):
    if not condition:
        raise ValueError(msg)

def skip_read_line(fd, no_eof=False):
    """
    Read the first non-empty line (if any) from the given file
    object. Return an empty string at EOF, if `no_eof` is False. If it
    is True, raise the EOFError instead.
    """
    ls = ''
    while 1:
        try:
            line = fd.readline()

        except EOFError:
            break

        if not line:
            if no_eof:
                raise EOFError

            else:
                break

        ls = line.strip()
        if ls and (ls[0] != '#'):
            break

    return ls

def read_token(fd):
    """
    Read a single token (sequence of non-whitespace characters) from the
    given file object.
    Notes
    -----
    Consumes the first whitespace character after the token.
    """
    out = ''
    # Skip initial whitespace.

    while 1:
        ch = fd.read(1)
        if ch.isspace(): continue
        elif len(ch) == 0: return out
        else: break

    while not ch.isspace():
        out = out + ch
        ch = fd.read(1)
        if len(ch) == 0: break

    return out

def read_array(fd, n_row, n_col, dtype):
    """
    Read a NumPy array of shape `(n_row, n_col)` from the given file
    object and cast it to type `dtype`.
    If `n_col` is None, determine the number of columns automatically.
    """
    if n_col is None:
        idx = fd.tell()
        row = fd.readline().split()
        fd.seek(idx)
        n_col = len(row)

    count = n_row * n_col
    val = nm.fromfile(fd, sep=' ', count=count)

    if val.shape[0] < count:
        raise ValueError('(%d, %d) array reading failed!' % (n_row, n_col))

    val = nm.asarray(val, dtype=dtype)
    val.shape = (n_row, n_col)

    return val

def read_gmsh(filename):
    msh_cells = {
        1: (2, 2),
        2: (2, 3),
        3: (2, 4),
        4: (3, 4),
        5: (3, 8),
        6: (3, 6),
    }
    prism2hexa = nm.asarray([0, 1, 2, 2, 3, 4, 5, 5])

    def read():
        fd = open(filename, 'r')

        conns = []
        descs = []
        mat_ids = []
        tags = []
        dims = []

        while 1:
            line = skip_read_line(fd).split()
            if not line:
                break

            ls = line[0]
            if ls == '$MeshFormat':
                skip_read_line(fd)
            elif ls == '$PhysicalNames':
                num = int(read_token(fd))
                for ii in range(num):
                    skip_read_line(fd)
            elif ls == '$Nodes':
                num = int(read_token(fd))
                coors = read_array(fd, num, 4, nm.float64)

            elif ls == '$Elements':
                num = int(read_token(fd))
                for ii in range(num):
                    line = [int(jj) for jj in skip_read_line(fd).split()]
                    if line[1] > 6:
                        continue
                    dimension, nc = msh_cells[line[1]]
                    dims.append(dimension)
                    ntag = line[2]
                    mat_id = line[3]
                    conn = line[(3 + ntag):]
                    desc = '%d_%d' % (dimension, nc)
                    if desc in descs:
                        idx = descs.index(desc)
                        conns[idx].append(conn)
                        mat_ids[idx].append(mat_id)
                        tags[idx].append(line[3:(3 + ntag)])
                    else:
                        descs.append(desc)
                        conns.append([conn])
                        mat_ids.append([mat_id])
                        tags.append(line[3:(3 + ntag)])

            elif ls == '$Periodic':
                periodic = ''
                while 1:
                    pline = skip_read_line(fd)
                    if '$EndPeriodic' in pline:
                        break
                    else:
                        periodic += pline

            elif line[0] == '#' or ls[:4] == '$End':
                pass

            else:
                print('skipping unknown entity: %s' % line)
                continue

        fd.close()

        dim = nm.max(dims)

        if '2_2' in descs:
            idx2 = descs.index('2_2')
            descs.pop(idx2)
            del(conns[idx2])
            del(mat_ids[idx2])

        if '3_6' in descs:
            idx6 = descs.index('3_6')
            c3_6as8 = nm.asarray(conns[idx6],
                                 dtype=nm.int32)[:,prism2hexa]
            if '3_8' in descs:
                descs.pop(idx6)
                c3_6m = nm.asarray(mat_ids.pop(idx6), type=nm.int32)
                idx8 = descs.index('3_8')
                c3_8 = nm.asarray(conns[idx8], type=nm.int32)
                c3_8m = nm.asarray(mat_ids[idx8], type=nm.int32)
                conns[idx8] = nm.vstack([c3_8, c3_6as8])
                mat_ids[idx8] = nm.hstack([c3_8m, c3_6m])
            else:
                descs[idx6] = '3_8'
                conns[idx6] = c3_6as8

        descs0, mat_ids0, conns0 = [], [], []
        for ii in range(len(descs)):
            if int(descs[ii][0]) == dim:
                conns0.append(nm.asarray(conns[ii], dtype=nm.int32) - 1)
                mat_ids0.append(nm.asarray(mat_ids[ii], dtype=nm.int32))
                descs0.append(descs[ii])


        p = coors[:, 1:].T
        t = conns0[0].T

        if nm.sum(p[2, :]) < 1e-10:
            p = p[0:2, :]

        tmp = nm.ascontiguousarray(p.T)
        tmp, ixa, ixb = nm.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]), return_index=True, return_inverse=True)
        p = p[:, ixa]
        t = ixb[t]

        p = nm.ascontiguousarray(p)
        t = nm.ascontiguousarray(t)

        if t.shape[0] == 3 and p.shape[0] == 2:
            return MeshTri(p, t)
        elif t.shape[0] == 4 and p.shape[0] == 3:
            return MeshTet(p, t)
        elif t.shape[0] == 4 and p.shape[0] == 2:
            return MeshQuad(p, t)
        else:
            raise Exception("mesh type not supported")

    return read()


def read_comsol(filename):

    fd = open(filename, 'r')

    def _read_commented_int():
        return int(skip_read_line(fd).split('#')[0])

    def _skip_comment():
        read_token(fd)
        fd.readline()

    def read():

        mode = 'header'

        coors = conns = None
        while 1:
            if mode == 'header':
                line = skip_read_line(fd)

                n_tags = _read_commented_int()
                for ii in range(n_tags):
                    skip_read_line(fd)
                n_types = _read_commented_int()
                for ii in range(n_types):
                    skip_read_line(fd)

                skip_read_line(fd)
                assert_(skip_read_line(fd).split()[1] == 'Mesh')
                skip_read_line(fd)
                dim = _read_commented_int()
                assert_((dim == 2) or (dim == 3))
                n_nod = _read_commented_int()
                i0 = _read_commented_int()
                mode = 'points'

            elif mode == 'points':
                _skip_comment()
                coors = read_array(fd, n_nod, dim, nm.float64)
                mode = 'cells'

            elif mode == 'cells':

                n_types = _read_commented_int()
                conns = []
                descs = []
                mat_ids = []
                for it in range(n_types):
                    t_name = skip_read_line(fd).split()[1]
                    n_ep = _read_commented_int()
                    n_el = _read_commented_int()

                    _skip_comment()
                    aux = read_array(fd, n_el, n_ep, nm.int32)
                    if t_name == 'tri':
                        conns.append(aux)
                        descs.append('2_3')
                        is_conn = True
                    elif t_name == 'quad':
                        # Rearrange element node order to match SfePy.
                        aux = aux[:,(0,1,3,2)]
                        conns.append(aux)
                        descs.append('2_4')
                        is_conn = True
                    elif t_name == 'hex':
                        # Rearrange element node order to match SfePy.
                        aux = aux[:,(0,1,3,2,4,5,7,6)]
                        conns.append(aux)
                        descs.append('3_8')
                        is_conn = True
                    elif t_name == 'tet':
                        conns.append(aux)
                        descs.append('3_4')
                        is_conn = True
                    else:
                        is_conn = False

                    # Skip parameters.
                    n_pv = _read_commented_int()
                    n_par = _read_commented_int()
                    for ii in range(n_par):
                        skip_read_line(fd)

                    n_domain = _read_commented_int()

                    #assert_(n_domain == n_el)
                    if is_conn:
                        _skip_comment()
                        mat_id = read_array(fd, n_domain, 1, nm.int32)
                        mat_ids.append(mat_id)
                    else:
                        for ii in range(n_domain):
                            skip_read_line(fd)

                    # Skip up/down pairs.
                    n_ud = _read_commented_int()
                    for ii in range(n_ud):
                        skip_read_line(fd)
                break

        fd.close()

        p = coors.T
        t = conns[0].T

        if p.shape[0] == 2:
            if t.shape[0] == 3:
                return MeshTri(p, t)
            elif t.shape[0] == 4:
                return MeshQuad(p, t)
            else:
                raise Exception("mesh type not supported")
        elif p.shape[0] == 3:
            if t.shape[0] == 4:
                return MeshTet(p, t)
            else:
                raise Exception("mesh type not supported")
        else:
            raise Exception("mesh type not supported")

    return read()
