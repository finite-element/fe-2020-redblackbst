"""Solve a elliptic eigenvalue problem with Dirichlet boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if (x[0] > 1 - eps or x[1] > 1 - eps
                or (x[0] < eps - 1 and x[1] >= 0)
                or (x[0] >= 0 and x[1] < eps - 1)
                or (-eps < x[0] < eps and x[1] <= 0)
                or (x[0] <= 0 and -eps < x[1] < eps)):

            # if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1
        else:
            return 0

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values), np.flatnonzero(f.values == 0)


def solve_eigenvalue(cell, degree, resolution):
    """Solve an elliptic eigenvalue problem on a L-shaped mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    if cell is ReferenceTriangle:
        mesh = LShapedMesh(resolution, resolution)
    elif cell is ReferenceRectangle:
        mesh = LShapedRecMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create an appropriate (complete) quadrature rule.
    fe = fs.element
    mesh = fs.mesh
    Q = gauss_quadrature(fe.cell, 2 * fe.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi = fe.tabulate(Q.points)
    grad_phi = fe.tabulate(Q.points, True)

    # Create the left hand side matrix and right hand side matrix.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    B = sp.lil_matrix((fs.node_count, fs.node_count))

    # Now loop over all the cells and assemble A and B
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes = fs.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        invJ = np.linalg.inv(J)
        detJ = np.abs(np.linalg.det(J))

        # Compute the actual cell quadrature.
        p = np.einsum("ji,klj->kli", invJ, grad_phi)
        A[np.ix_(nodes, nodes)] += np.einsum("ijq,q->ij", np.einsum("qid,qjd->ijq", p, p), Q.weights) * detJ
        B[np.ix_(nodes, nodes)] += np.einsum("ijq,q->ij", np.einsum("qi,qj->ijq", phi, phi), Q.weights) * detJ

    # Handle the boundary conditions
    bnodes, inodes = boundary_nodes(fs)

    # Create the function to hold the solution.
    u = [Function(fs) for _ in range(10)]
    z = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A[np.ix_(inodes, inodes)])
    B = sp.csr_matrix(B[np.ix_(inodes, inodes)])
    w, uc = splinalg.eigsh(A, 10, B, which='SM')
    for i in range(10):
        u[i].values[inodes] = uc[:, i]
        u[i].values[bnodes] = 0.

    # Return the eigenvalues, eigenfuctions and a zero function.
    return w, u, z


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Solve a Eigenvalue problem on the unit square.""")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]

    eps = 1.e-10
    w1, u1, z1 = solve_eigenvalue(ReferenceTriangle, degree, resolution)
    w2, u2, z2 = solve_eigenvalue(ReferenceRectangle, degree, resolution)
    for i in range(10):
        e1 = errornorm(u1[i], z1)
        e2 = errornorm(u2[i], z2)
        assert e1 - e2 < eps

    print("Triangular mesh: %s" % w1)
    print("Number of elements: %s" % u1[0].function_space.mesh.entity_counts[-1])
    print("Rectangular mesh: %s" % w2)
    print("Number of elements: %s" % u2[0].function_space.mesh.entity_counts[-1])
    fig = [plt.figure(figsize=(20, 30)) for _ in range(2)]
    for i in range(10):
        ax = fig[0].add_subplot(5, 2, i + 1, projection='3d')
        ax.set_title(f'lambda = {w1[i]}')
        u1[i].plot(ax)
    for i in range(10):
        ax = fig[1].add_subplot(5, 2, i + 1, projection='3d')
        ax.set_title(f'lambda = {w2[i]}')
        u2[i].plot(ax)
    plt.show()
