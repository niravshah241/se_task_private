# Weak form in UFL terms
from basix.ufl import element
from ufl import (Coefficient, Constant, FunctionSpace, Mesh, TestFunction,
                 TrialFunction, ds, dx, grad, inner)

# Specify finite element, function space
# (Lagrange, Polynomial degree 1)
e = element("Lagrange", "tetrahedron", 1)
coord_element = element("Lagrange", "tetrahedron",
                        1, shape=(3,))
mesh = Mesh(coord_element)
V = FunctionSpace(mesh, e)
u, v = TrialFunction(V), TestFunction(V)

# Coefficients, bilinear and linear forms
f = Coefficient(V)
g = Coefficient(V)
kappa = Constant(mesh)
a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
