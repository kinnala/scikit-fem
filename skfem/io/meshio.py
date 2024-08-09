"""Import/export any formats supported by meshio."""

import logging
from pathlib import Path

import meshio
import numpy as np

from skfem.mesh import (MeshTet1, MeshTet2, MeshHex1, MeshHex2, MeshWedge1,
                        MeshTri1, MeshTri2, MeshQuad1, MeshQuad2, MeshLine1)
from skfem.generic_utils import OrientedBoundary


logger = logging.getLogger(__name__)


MESH_TYPE_MAPPING = {
    'tetra': MeshTet1,
    'tetra10': MeshTet2,
    'hexahedron': MeshHex1,
    'hexahedron27': MeshHex2,
    'wedge': MeshWedge1,
    'triangle': MeshTri1,
    'triangle6': MeshTri2,
    'quad': MeshQuad1,
    'quad9': MeshQuad2,
    'line': MeshLine1,
}

BOUNDARY_TYPE_MAPPING = {
    'line': 'vertex',
    'triangle': 'line',
    'quad': 'line',
    'tetra': 'triangle',
    'hexahedron': 'quad',
    'tetra10': 'triangle',  # TODO support quadratic facets
    'triangle6': 'line',  # TODO
    'quad9': 'line',  # TODO
    'hexahedron27': 'quad',  # TODO
}

TYPE_MESH_MAPPING = {MESH_TYPE_MAPPING[k]: k
                     for k in dict(reversed(list(MESH_TYPE_MAPPING.items())))}


HEX_MAPPING = [0, 3, 6, 2, 1, 5, 7, 4,
               10, 16, 14, 9, 12, 18, 17, 11, 8, 15, 19, 13,
               20, 25, 22, 23, 21, 24,
               26]
INV_HEX_MAPPING = [HEX_MAPPING.index(i)
                   for i in range(len(HEX_MAPPING))]


def from_meshio(m,
                out=None,
                int_data_to_sets=False,
                force_meshio_type=None,
                ignore_orientation=False,
                ignore_interior_facets=False):

    if ignore_interior_facets:
        logger.warning("kwarg ignore_interior_facets is unused.")

    cells = m.cells_dict
    meshio_type = None

    if force_meshio_type is None:
        # detect 3D
        for k in cells:
            if k in {'tetra',
                     'hexahedron',
                     'tetra10',
                     'hexahedron27',
                     'wedge'}:
                meshio_type = k
                break

        if meshio_type is None:
            # detect 2D
            for k in cells:
                if k in {'triangle',
                         'quad',
                         'triangle6',
                         'quad9'}:
                    meshio_type = k
                    break

        if meshio_type is None:
            # detect 1D
            for k in cells:
                if k == 'line':
                    meshio_type = k
                    break
    else:
        meshio_type = force_meshio_type

    if meshio_type is None:
        raise NotImplementedError("Mesh type(s) not supported "
                                  "in import: {}.".format(cells.keys()))

    mesh_type = MESH_TYPE_MAPPING[meshio_type]

    # create p and t
    p = np.ascontiguousarray(mesh_type.strip_extra_coordinates(m.points).T)
    t = np.ascontiguousarray(cells[meshio_type].T)

    # reorder t if needed
    if meshio_type == 'hexahedron':
        t = t[INV_HEX_MAPPING[:8]]
    elif meshio_type == 'hexahedron27':
        t = t[INV_HEX_MAPPING]

    if int_data_to_sets:
        m.int_data_to_sets()

    subdomains = {}
    boundaries = {}

    # parse any subdomains from cell_sets
    if m.cell_sets:
        subdomains = {k: v[meshio_type].astype(np.int32)
                      for k, v in m.cell_sets_dict.items()
                      if meshio_type in v}

    # create temporary mesh for matching boundary elements
    mtmp = mesh_type(p, t, validate=False)
    bnd_type = BOUNDARY_TYPE_MAPPING[meshio_type]

    # parse boundaries from cell_sets
    if m.cell_sets and bnd_type in m.cells_dict:
        p2f = mtmp.p2f
        for k, v in m.cell_sets_dict.items():
            if bnd_type in v and k.split(":")[0] != "gmsh":
                facets = m.cells_dict[bnd_type][v[bnd_type]].T
                sorted_facets = np.sort(facets, axis=0)
                ind = p2f[:, sorted_facets[0]]
                for itr in range(sorted_facets.shape[0] - 1):
                    ind = ind.multiply(p2f[:, sorted_facets[itr + 1]])
                # does not preserve order:
                # boundaries[k] = np.nonzero(ind)[0].astype(np.int32)
                ii, jj = ind.nonzero()
                boundaries[k] = ii[np.argsort(jj)]

                if not ignore_orientation:
                    try:
                        ori = np.zeros_like(boundaries[k], dtype=np.float64)
                        t1, t2 = mtmp.f2t[:, boundaries[k]]
                        if facets.shape[0] == 2:
                            tangents = (mtmp.p[:, facets[1]]
                                        - mtmp.p[:, facets[0]])
                            normals = np.array([-tangents[1], tangents[0]])
                        elif facets.shape[0] == 3:
                            tangents1 = (mtmp.p[:, facets[1]]
                                         - mtmp.p[:, facets[0]])
                            tangents2 = (mtmp.p[:, facets[2]]
                                         - mtmp.p[:, facets[0]])
                            normals = -np.cross(tangents1.T, tangents2.T).T
                        else:
                            raise NotImplementedError
                        # perform a dot product between the proposed normal
                        # and the edges of the element to fix the orientation
                        for itr in range(mtmp.t.shape[0]):
                            ori += np.sum(normals
                                          * (mtmp.p[:, facets[0]]
                                             - mtmp.p[:, mtmp.t[itr, t1]]),
                                          axis=0)
                        ori = 1 * (ori > 0)
                        # check that the orientation is not 'illegal'
                        # this might happen for a hole
                        # as a consequence, choose the only valid ori
                        ori[t2 == -1] = 0
                        boundaries[k] = OrientedBoundary(boundaries[k],
                                                         ori)
                    except Exception:
                        logger.warning("Failure to orient a boundary.")

    # MSH 2.2 tag parsing
    if len(boundaries) == 0 and m.cell_data and m.field_data:
        try:
            elements_tag = m.cell_data_dict['gmsh:physical'][meshio_type]
            subdomains = {}
            tags = np.unique(elements_tag)

            def find_tagname(tag):
                for key in m.field_data:
                    if m.field_data[key][0] == tag:
                        return key
                return None

            for tag in tags:
                t_set = np.nonzero(tag == elements_tag)[0].astype(np.int32)
                subdomains[find_tagname(tag)] = t_set

            # find tagged boundaries
            if bnd_type in m.cell_data_dict['gmsh:physical']:
                facets = m.cells_dict[bnd_type]
                facets_tag = m.cell_data_dict['gmsh:physical'][bnd_type]

            # put meshio facets to dict
            dic = {tuple(np.sort(facets[i])): facets_tag[i]
                   for i in range(facets.shape[0])}

            # get index of corresponding Mesh.facets for each meshio
            # facet found in the dict
            index = np.array([[dic[tuple(np.sort(mtmp.facets[:, i]))], i]
                              for i in mtmp.boundary_facets()
                              if tuple(np.sort(mtmp.facets[:, i])) in dic])

            # read meshio tag numbers and names
            tags = index[:, 0]
            boundaries = {}
            for tag in np.unique(tags):
                tagindex = np.nonzero(tags == tag)[0].astype(np.int32)
                boundaries[find_tagname(tag)] = index[tagindex, 1]

        except Exception:
            logger.warning("Failure to parse tags from meshio.")

    # attempt parsing skfem tags
    if m.cell_data:
        _boundaries, _subdomains = mtmp._decode_cell_data(m.cell_data)
        boundaries.update(_boundaries)
        subdomains.update(_subdomains)

    # export mesh data
    if out is not None and isinstance(out, list):
        for i, field in enumerate(out):
            out[i] = getattr(m, field)

    return mesh_type(
        p,
        t,
        None if len(boundaries) == 0 else boundaries,
        None if len(subdomains) == 0 else subdomains,
    )


def from_file(filename, out, **kwargs):
    return from_meshio(meshio.read(filename), out, **kwargs)


def to_meshio(mesh,
              point_data=None,
              cell_data=None,
              encode_cell_data=True,
              encode_point_data=False):

    t = mesh.dofs.element_dofs.copy()
    if isinstance(mesh, MeshHex2):
        t = t[HEX_MAPPING]
    elif isinstance(mesh, MeshHex1):
        t = t[HEX_MAPPING[:8]]

    mtype = TYPE_MESH_MAPPING[type(mesh)]
    cells = {mtype: t.T}

    if encode_cell_data:
        if cell_data is None:
            cell_data = {}
        cell_data.update(mesh._encode_cell_data())

    if encode_point_data:
        if point_data is None:
            point_data = {}
        point_data.update(mesh._encode_point_data())

    mio = meshio.Mesh(
        mesh.p.T,
        cells,
        point_data=point_data,
        cell_data=cell_data,
    )

    return mio


def to_file(mesh,
            filename,
            point_data=None,
            cell_data=None,
            encode_cell_data=True,
            encode_point_data=False,
            **kwargs):

    path = Path(filename)
    if 'file_format' not in kwargs and path.suffix == '.msh':
        kwargs.update({'file_format': 'gmsh'})

    meshio.write(path,
                 to_meshio(mesh,
                           point_data,
                           cell_data,
                           encode_cell_data,
                           encode_point_data),
                 **kwargs)
