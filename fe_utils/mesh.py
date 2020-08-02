from scipy.spatial import Delaunay, KDTree
import numpy as np
import itertools
from .finite_elements import LagrangeElement
from .reference_elements import ReferenceTriangle, ReferenceInterval, ReferenceRectangle


class Mesh(object):
    """A one or two dimensional mesh composed of intervals or triangles
    respectively."""

    def __init__(self, vertex_coords, cell_vertices):
        """
        :param vertex_coords: an vertex_count x dim array of the coordinates of
          the vertices in the mesh.
        :param cell_vertices: an cell_count x (dim+1) array of the
          indices of the vertices of which each cell is made up.
        """

        self.dim = vertex_coords.shape[1]
        self.top_dim = cell_vertices.shape[1]
        """The geometric and topological dimension of the mesh. Immersed
        manifolds are not supported.
        """

        if self.dim not in (1, 2):
            raise ValueError("Only 1D and 2D meshes are supported")
        if self.top_dim not in (2, 3, 4):
            raise ValueError("Only interval or triangular or rectangular elements are supported")

        self.vertex_coords = vertex_coords
        """The coordinates of all the vertices in the mesh."""

        self.cell_vertices = np.sort(cell_vertices)
        """The indices of the vertices incident to cell."""

        if self.dim == 2:
            if self.top_dim == 3:
                self.edge_vertices = np.array(list(set(tuple(sorted(e))
                                                       for t in cell_vertices
                                                       for e in itertools.combinations(t, 2))))
                """The indices of the vertices incident to edge (only for 2D
                meshes)."""

                # Invert self.edge_vertices so that it is possible to look up
                # the edge index given the vertex indices.
                edge_dict = {tuple(e): i
                             for i, e_ in enumerate(self.edge_vertices)
                             for e in (e_, reversed(e_))}

                # List the local vertex indices associated with
                # each local edge index.
                local_edge_vertices = np.array([[1, 2], [0, 2], [0, 1]])

                self.cell_edges = np.fromiter(
                    (edge_dict[tuple(t.take(local_edge_vertices[e]))]
                     for t in self.cell_vertices
                     for e in range(3)),
                    dtype=np.int32,
                    count=self.cell_vertices.size).reshape((-1, 3))
                """The indices of the edges incident to each cell (only for 2D
                meshes)."""
            else:
                cv = np.empty(cell_vertices.shape, dtype=np.int32)
                for i, c in enumerate(cell_vertices):
                    rec_coords = np.array([vertex_coords[k] for k in c])
                    centroid = sum(rec_coords) / 4
                    for j in range(4):
                        if rec_coords[j, 0] < centroid[0] and rec_coords[j, 1] < centroid[1]:
                            cv[i, 0] = c[j]
                        if rec_coords[j, 0] > centroid[0] and rec_coords[j, 1] < centroid[1]:
                            cv[i, 1] = c[j]
                        if rec_coords[j, 0] > centroid[0] and rec_coords[j, 1] > centroid[1]:
                            cv[i, 2] = c[j]
                        if rec_coords[j, 0] < centroid[0] and rec_coords[j, 1] > centroid[1]:
                            cv[i, 3] = c[j]
                self.cell_vertices = cv

                # List the local vertex indices associated with
                # each local edge index.
                local_edge_vertices = np.array([[0, 1], [1, 2], [0, 3], [3, 2]])
                self.edge_vertices = np.array(list(set(tuple(sorted(t.take(local_edge_vertices[e])))
                                                       for t in cv
                                                       for e in range(4))))
                """The indices of the vertices incident to edge (only for 2D
                meshes)."""

                # Invert self.edge_vertices so that it is possible to look up
                # the edge index given the vertex indices.
                edge_dict = {tuple(e): i
                             for i, e_ in enumerate(self.edge_vertices)
                             for e in (e_, reversed(e_))}

                self.cell_edges = np.fromiter(
                    (edge_dict[tuple(t.take(local_edge_vertices[e]))]
                     for t in cv
                     for e in range(4)),
                    dtype=np.int32,
                    count=cv.size).reshape((-1, 4))
                """The indices of the edges incident to each cell (only for 2D
                meshes)."""

        if self.dim == 2:
            self.entity_counts = np.array((vertex_coords.shape[0],
                                           self.edge_vertices.shape[0],
                                           self.cell_vertices.shape[0]))
            """The number of entities of each dimension in the mesh. So
            :attr:`entity_counts[0]` is the number of vertices in the
            mesh."""
        else:
            self.entity_counts = np.array((vertex_coords.shape[0],
                                           self.cell_vertices.shape[0]))

        #: The :class:`~.reference_elements.ReferenceCell` of which this
        #: :class:`Mesh` is composed.
        if self.dim == 1:
            self.cell = ReferenceInterval
        elif self.top_dim == 3:
            self.cell = ReferenceTriangle
        else:
            self.cell = ReferenceRectangle

        #: Precalculating gradient for the jacobian
        cg1 = LagrangeElement(self.cell, 1)
        if self.dim == 2:
            self.grad = cg1.tabulate(np.array([[0, 0]]), True)[0, :]
        else:
            self.grad = cg1.tabulate(np.array([[0]]), True)[0, :]

    def adjacency(self, dim1, dim2):
        """Return the set of `dim2` entities adjacent to each `dim1`
        entity. For example if `dim1==2` and `dim2==1` then return the list of
        edges (1D entities) adjacent to each triangle (2D entity).

        The return value is a rank 2 :class:`numpy.array` such that
        ``adjacency(dim1, dim2)[e1, :]`` is the list of dim2 entities
        adjacent to entity ``(dim1, e1)``.

        This operation is only defined where `self.dim >= dim1 > dim2`.

        This method is simply a more systematic way of accessing
        :attr:`edge_vertices`, :attr:`cell_edges` and :attr:`cell_vertices`.
        """

        if dim2 >= dim1:
            raise ValueError("""dim2 must be less than dim1.""")
        if dim2 < 0:
            raise ValueError("""dim2 cannot be negative.""")
        if dim1 > self.dim:
            raise ValueError("""dim1 cannot exceed the mesh dimension.""")

        if dim1 == 1:
            if self.dim == 1:
                return self.cell_vertices
            else:
                return self.edge_vertices
        elif dim1 == 2:
            if dim2 == 0:
                return self.cell_vertices
            else:
                return self.cell_edges

    def jacobian(self, c):
        """Return the Jacobian matrix for the specified cell.

        :param c: The number of the cell for which to return the Jacobian.
        :result: The Jacobian for cell ``c``.
        """

        coords = self.vertex_coords[self.cell_vertices[c, :], :].T
        return np.dot(coords, self.grad)


class UnitIntervalMesh(Mesh):
    """A mesh of the unit interval."""

    def __init__(self, nx):
        """
        :param nx: The number of cells.
        """
        points = np.array(list((x,) for x in np.linspace(0, 1, nx + 1)))
        points.shape = (points.shape[0], 1)

        cells = np.array(list((a, a + 1) for a in range(nx)))

        super(UnitIntervalMesh, self).__init__(points,
                                               cells)


class UnitSquareMesh(Mesh):
    """A triangulated :class:`Mesh` of the unit square."""

    def __init__(self, nx, ny):
        """
        :param nx: The number of cells in the x direction.
        :param ny: The number of cells in the y direction.
        """
        points = list((x, y)
                      for x in np.linspace(0, 1, nx + 1)
                      for y in np.linspace(0, 1, ny + 1))

        mesh = Delaunay(points)

        super(UnitSquareMesh, self).__init__(mesh.points,
                                             mesh.simplices)


class LShapedMesh(Mesh):
    """A triangulated :class:`Mesh` of the L-shaped domain."""

    def __init__(self, nx, ny):
        """
        :param nx: The number of cells in the x direction.
        :param ny: The number of cells in the y direction.
        """
        points = list((x, y)
                      for x in np.linspace(-1, 1, nx + 1)
                      for y in np.linspace(-1, 1, ny + 1)
                      if x >= 0 or y >= 0)

        mesh = Delaunay(points)
        cell_vertices = np.empty((0, 3), int)
        for a, b, c in mesh.simplices:
            pa, pb, pc = points[a], points[b], points[c]
            xc = (pa[0] + pb[0] + pc[0]) / 3
            yc = (pa[1] + pb[1] + pc[1]) / 3
            if xc >= 0 or yc >= 0:
                cell_vertices = np.append(cell_vertices, np.array([[a, b, c]]), axis=0)

        super(LShapedMesh, self).__init__(mesh.points,
                                          cell_vertices)


class LShapedRecMesh(Mesh):
    """A rectangular :class:`Mesh` of the L-shaped domain."""

    def __init__(self, nx, ny):
        """
        :param nx: The number of cells in the x direction.
        :param ny: The number of cells in the y direction.
        """
        eps = 1e-10
        points = np.array([[x, y]
                           for x in np.linspace(-1, 1, nx + 1)
                           for y in np.linspace(-1, 1, ny + 1)
                           if x >= 0 or y >= 0])
        dx, dy = 2. / (2 * nx), 2. / (2 * ny)
        inpoints = np.array([[x, y]
                             for x in np.linspace(dx - 1, 1 - dx, nx)
                             for y in np.linspace(dy - 1, 1 - dy, ny)
                             if x > 0 or y > 0])
        tree = KDTree(points)
        cell_vertices = np.array([tree.query_ball_point(pt, np.sqrt(2) * max(dx, dy) + eps)
                                  for pt in inpoints])

        super(LShapedRecMesh, self).__init__(points, cell_vertices)
