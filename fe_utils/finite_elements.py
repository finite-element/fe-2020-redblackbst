# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle, ReferenceRectangle

np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    p = degree

    if cell is ReferenceInterval:
        return np.array([[i / p] for i in range(p + 1)])
    elif cell is ReferenceTriangle:
        v = [[0, 0], [1, 0], [0, 1]]
        e0 = [[(p - i) / p, i / p] for i in range(1, p)]
        e1 = [[0, i / p] for i in range(1, p)]
        e2 = [[i / p, 0] for i in range(1, p)]
        interior = [[i / p, j / p] for i in range(1, p) for j in range(1, p - i)]
        return np.array(v + e0 + e1 + e2 + interior)
        # return np.array([[i / p, j / p] for i in range(p + 1) for j in range(p + 1 - i)])
    elif cell is ReferenceRectangle:
        v = [[0, 0], [1, 0], [1, 1], [0, 1]]
        e0 = [[i / p, 0] for i in range(1, p)]
        e1 = [[1, i / p] for i in range(1, p)]
        e2 = [[0, i / p] for i in range(1, p)]
        e3 = [[i / p, 1] for i in range(1, p)]
        interior = [[i / p, j / p] for i in range(1, p) for j in range(1, p)]
        return np.array((v + e0 + e1 + e2 + e3 + interior))
    else:
        raise ValueError("Unknown cell type: %s" % cell)


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """

    p = degree

    if cell is ReferenceInterval:
        if grad:
            v = np.array([k * points[:, 0] ** max(k - 1, 0) for k in range(p + 1)]).T
            v.shape = v.shape + (1,)
        else:
            v = np.array([points[:, 0] ** k for k in range(p + 1)]).T
    elif cell is ReferenceTriangle:
        if grad:
            v = np.array([[(k - j) * points[:, 0] ** max(k - j - 1, 0) * points[:, 1] ** j,
                           points[:, 0] ** (k - j) * j * points[:, 1] ** max(j - 1, 0)]
                          for k in range(p + 1)
                          for j in range(k + 1)]).transpose(2, 0, 1)
        else:
            v = np.array([points[:, 0] ** (k - j) * points[:, 1] ** j
                          for k in range(p + 1)
                          for j in range(k + 1)]).T
    elif cell is ReferenceRectangle:
        if grad:
            v = np.array([[i * points[:, 0] ** max(i - 1, 0) * points[:, 1] ** j,
                           points[:, 0] ** i * j * points[:, 1] ** max(j - 1, 0)]
                          for i in range(p + 1)
                          for j in range(p + 1)]).transpose(2, 0, 1)
        else:
            v = np.array([points[:, 0] ** i * points[:, 1] ** j
                          for i in range(p + 1)
                          for j in range(p + 1)]).T
    else:
        raise ValueError("Unknown cell type: %s" % cell)

    return np.nan_to_num(v)


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of nodes
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim + 1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """

        if grad:
            return np.einsum("ijk,jl->ilk", vandermonde_matrix(self.cell, self.degree, points, True), self.basis_coefs)
        else:
            return np.dot(vandermonde_matrix(self.cell, self.degree, points), self.basis_coefs)

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        return np.array([fn(x) for x in self.nodes])

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        if cell is ReferenceInterval:
            entity_nodes = {0: {0: [0], 1: [degree]},
                            1: {0: range(1, degree)}}
        elif cell is ReferenceTriangle:
            entity_nodes = {0: {0: [0], 1: [1], 2: [2]},
                            1: {0: range(3, degree + 2), 1: range(degree + 2, 2 * degree + 1),
                                2: range(2 * degree + 1, 3 * degree)},
                            2: {0: range(3 * degree, round((degree + 1) * (degree + 2) / 2))}
                            }
        elif cell is ReferenceRectangle:
            entity_nodes = {0: {0: [0], 1: [1], 2: [2], 3: [3]},
                            1: {0: range(4, degree + 3),
                                1: range(degree + 3, 2 * degree + 2),
                                2: range(2 * degree + 2, 3 * degree + 1),
                                3: range(3 * degree + 1, 4 * degree)},
                            2: {0: range(4 * degree, (degree + 1) ** 2)}
                            }
        else:
            raise ValueError("Unknown cell type: %s" % cell)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)
