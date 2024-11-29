import time
import dolfinx
from dolfinx.fem.petsc import assemble_matrix, \
    assemble_vector, set_bc, apply_lifting
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

# Create mesh
domain = dolfinx.mesh.create_box(MPI.COMM_WORLD,
                                 [[0, 0, 0],
                                  [1, 1, 1]],
                                 [15, 15, 15],
                                 dolfinx.mesh.CellType.tetrahedron)

# Mark boundaries


def z_1(x):
    return np.isclose(x[2], 1)


def y_1(x):
    return np.isclose(x[1], 1)


def x_1(x):
    return np.isclose(x[0], 1)


fdim = domain.topology.dim - 1
z_1_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, z_1)
y_1_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, y_1)
x_1_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, x_1)

marked_facets = np.hstack([z_1_facets, y_1_facets, x_1_facets])
marked_values = np.hstack([np.full_like(z_1_facets, 3),
                           np.full_like(y_1_facets, 2),
                           np.full_like(x_1_facets, 1)])

sorted_facets = np.argsort(marked_facets)
boundaries = dolfinx.mesh.meshtags(domain, fdim,
                                   marked_facets[sorted_facets],
                                   marked_values[sorted_facets])

# Define function space: Lagrange, Polynomial degree 1
V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Define source term
x = ufl.SpatialCoordinate(domain)
f = 10. * ufl.exp(2 * ((x[0] - 0.40) * (x[0] - 0.40) +
                       (x[1] - 0.55) * (x[1] - 0.55) +
                       (x[2] - 0.27) * (x[2] - 0.27)))

# Bilinear and Linear forms
a_cpp = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
l_cpp = dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)

# Specify boundary conditions
dofs_x1 = dolfinx.fem.locate_dofs_topological(V, fdim,
                                              boundaries.find(1))


def f1(x):
    values = np.zeros((1, x.shape[1]))
    return values


f_h1 = dolfinx.fem.Function(V)
f_h1.interpolate(f1)
bc_x1 = dolfinx.fem.dirichletbc(f_h1, dofs_x1)

dofs_y1 = dolfinx.fem.locate_dofs_topological(V, fdim,
                                              boundaries.find(2))


def f2(x):
    values = np.zeros((1, x.shape[1]))
    return values


f_h2 = dolfinx.fem.Function(V)
f_h2.interpolate(f2)
bc_y1 = dolfinx.fem.dirichletbc(f_h1, dofs_y1)

dofs_z1 = dolfinx.fem.locate_dofs_topological(V, fdim,
                                              boundaries.find(3))


def f3(x):
    values = np.zeros((1, x.shape[1]))
    return values


f_h3 = dolfinx.fem.Function(V)
f_h3.interpolate(f3)
bc_z1 = dolfinx.fem.dirichletbc(f_h3, dofs_z1)

bcs = [bc_x1, bc_y1, bc_z1]

# Bilinear and Linear side assembly
A = assemble_matrix(a_cpp, bcs=bcs)
A.assemble()

L = assemble_vector(l_cpp)
apply_lifting(L, [a_cpp], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD,
              mode=PETSc.ScatterMode.REVERSE)
set_bc(L, bcs)

# PETSc solver (with preconditioner) setup
ksp = PETSc.KSP()
ksp.create(domain.comm)
ksp.setOperators(A)
ksp.setType("cg")
ksp.getPC().setType("gamg")
ksp.setFromOptions()

# Solve the problem
u_h = dolfinx.fem.Function(V)
solve_start_time = time.process_time()
ksp.solve(L, u_h.vector)
solve_end_time = time.process_time()
print(f"Number of iterations: {ksp.getIterationNumber()}")
print(f"Convergence reason: {ksp.getConvergedReason()}")
ksp.destroy()
A.destroy()
L.destroy()
u_h.x.scatter_forward()

print(f"Solution time: {solve_end_time - solve_start_time}")
u_norm = domain.comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u_h, u_h) * ufl.dx)))
print(f"Norm of u: {u_norm}")

# Save solution in XDMF format
with dolfinx.io.XDMFFile(domain.comm, "poisson/u.xdmf",
                         "w") as sol_file:
    sol_file.write_mesh(domain)
    sol_file.write_function(u_h)
