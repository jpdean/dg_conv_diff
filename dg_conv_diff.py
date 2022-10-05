# TODO Test diffusion, advection, advection-diffusion, and add Neumann
# BC for Diffusion

from dolfinx import mesh, fem, io
import ufl
from ufl import inner, dx, grad, dot, dS, jump, avg
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def u_e_expr(x):
    """Analytical solution to steady state problem from Donea and Huerta"""
    return x[0] - (1 - np.exp(100 * x[0])) / (1 - np.exp(100))


def gamma_D_marker(x):
    return np.isclose(x[0], 0.0) |  np.isclose(x[0], 1.0)


def gamma_N_marker(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


class TimeDependentExpression():
    """Simple class to represent time dependent functions"""

    def __init__(self, expression):
        self.t = 0.0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)


n = 64
k = 1
t_end = 10.0
num_time_steps = 32

msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

V = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

w = fem.Constant(msh, np.array([1.0, 0.0], dtype=PETSc.ScalarType))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(
    msh, PETSc.ScalarType(10.0 * k**2))  # TODO Check k dependency
kappa = fem.Constant(msh, PETSc.ScalarType(0.01))

# u_D_expr = TimeDependentExpression(
#     lambda x, t: )
u_D = fem.Function(V)
u_D.interpolate(u_e_expr)

tdim = msh.topology.dim
msh.topology.create_entities(tdim - 1)
facet_imap = msh.topology.index_map(tdim - 1)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
indices = np.arange(0, num_facets)
values = np.arange(0, num_facets, dtype=np.intc)
dirichlet_facets = mesh.locate_entities_boundary(msh, tdim - 1, gamma_D_marker)
neumann_facets = mesh.locate_entities_boundary(msh, tdim - 1, gamma_N_marker)
boundary_id = {"gamma_D": 1, "gamma_N": 2}
values[dirichlet_facets] = boundary_id["gamma_D"]
values[neumann_facets] = boundary_id["gamma_N"]
mt = mesh.meshtags(msh, tdim - 1, indices, values)

ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 1, 0)
# FIXME CHECK CONV TERM / CHANGING THIS TO VERSION WITH NORMAL
a = fem.form(inner(u / delta_t, v) * dx -
             inner(w * u, grad(v)) * dx +
             inner(lmbda("+") * dot(w("+"), n("+")) * u("+") -
                   lmbda("-") * dot(w("-"), n("-")) * u("-"), jump(v)) * dS +
             inner(lmbda * dot(w, n) * u, v) * ds +
             kappa * (inner(grad(u), grad(v)) * dx -
                      inner(avg(grad(u)), jump(v, n)) * dS -
                      inner(jump(u, n), avg(grad(v))) * dS +
                      (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS -
                      inner(grad(u), v * n) * ds(boundary_id["gamma_D"]) -
                      inner(grad(v), u * n) * ds(boundary_id["gamma_D"]) +
                      (alpha / h) * inner(u, v) * ds(boundary_id["gamma_D"])))

f = fem.Constant(msh, PETSc.ScalarType(1.0))
# x = ufl.SpatialCoordinate(msh)
g = fem.Constant(msh, PETSc.ScalarType(0.0))
L = fem.form(inner(f + u_n / delta_t, v) * dx -
             inner((1 - lmbda) * dot(w, n) * u_D, v) * ds +
             kappa * (- inner(u_n * n, grad(v)) * ds(boundary_id["gamma_D"]) +
             (alpha / h) * inner(u_D, v) * ds(boundary_id["gamma_D"]) +
             inner(g, v) * ds(boundary_id["gamma_N"])))

A = fem.petsc.assemble_matrix(a)
A.assemble()
b = fem.petsc.create_vector(L)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u_file = io.VTXWriter(msh.comm, "u.bp", [u_n._cpp_object])

t = 0.0
u_file.write(t)
for n in range(num_time_steps):
    t += delta_t.value

    # u_D_expr.t = t
    # u_D.interpolate(u_D_expr)

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    fem.petsc.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp.solve(b, u_n.vector)
    u_n.x.scatter_forward()

    u_file.write(t)

u_file.close()

# Function spaces for exact solution
V_e = fem.FunctionSpace(msh, ("Lagrange", k + 3))

u_e = fem.Function(V_e)
u_e.interpolate(u_e_expr)

# Compute errors
e_u = norm_L2(msh.comm, u_n - u_e)

if msh.comm.rank == 0:
    print(f"e_u = {e_u}")
