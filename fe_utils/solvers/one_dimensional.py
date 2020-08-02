"""Solve a elliptic eigenvalue problem with Dirichlet boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
from numpy import exp
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def boundary_nodes(fs):
    """Return the nodes corresponding to x = 0 and x = 1. This is a
    unit-interval-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the left boundary, 2 if on the right boundary, 0 otherwise."""
        if x[0] < eps:
            return 1
        elif x[0] > 1 - eps:
            return 2
        else:
            return 0

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values == 1), np.flatnonzero(f.values == 2), np.flatnonzero(f.values == 0)


def solve_one_dimensional(degree, resolution, analytic=False, return_error=False):
    """Solve an elliptic eigenvalue problem on a L-shaped mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitIntervalMesh(resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: exp(x[0]))

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: -exp(x[0]))

    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe.cell, 2 * fe.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi = fe.tabulate(Q.points)
    grad_phi = fe.tabulate(Q.points, True)

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)

    # Now loop over all the cells and assemble A and l
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes = fs.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        invJ = np.linalg.inv(J)
        detJ = np.abs(np.linalg.det(J))

        # Compute the actual cell quadrature.
        l[nodes] += np.einsum("qi,k,qk,q->i", phi, f.values[nodes], phi, Q.weights) * detJ
        p = np.einsum("ji,klj->kli", invJ, grad_phi)
        A[np.ix_(nodes, nodes)] += np.einsum("ijq,q->ij",
                                             2 * np.einsum("qid,qjd->ijq", p, p) + np.einsum("qi,qj->ijq", phi, phi),
                                             Q.weights) * detJ

    # Create the function to hold the solution.
    u = Function(fs)

    # Handle the boundary conditions
    eps = 1e-10
    lnode, rnode, inodes = boundary_nodes(fs)
    lb = Function(fs)
    rb = Function(fs)
    lb.interpolate(lambda x: exp(0) if x[0] < eps else 0)
    rb.interpolate(lambda x: exp(1) if x[0] > 1 - eps else 0)
    u.values[lnode] = lb.values[lnode]
    u.values[rnode] = rb.values[rnode]
    l -= sp.lil_matrix.dot(A, u.values)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[inodes] = splinalg.spsolve(A[np.ix_(inodes, inodes)], l[inodes])

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Solve 1D problem on the unit interval.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    eps = 1.e-10
    u, error = solve_one_dimensional(degree, resolution, analytic, plot_error)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    u.plot(ax)
    plt.show()
