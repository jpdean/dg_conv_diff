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
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)


def gamma_N_marker(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def gamma_R_marker(x):
    return np.full(x[0].shape, False)


class TimeDependentExpression():
    """Simple class to represent time dependent functions"""

    def __init__(self, expression):
        self.t = 0.0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)


n = 32
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


tdim = msh.topology.dim
msh.topology.create_entities(tdim - 1)
facet_imap = msh.topology.index_map(tdim - 1)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
indices = np.arange(0, num_facets)
values = np.arange(0, num_facets, dtype=np.intc)
dirichlet_facets = mesh.locate_entities_boundary(msh, tdim - 1, gamma_D_marker)
neumann_facets = mesh.locate_entities_boundary(msh, tdim - 1, gamma_N_marker)
robin_facets = mesh.locate_entities_boundary(msh, tdim - 1, gamma_R_marker)
boundary_id = {"gamma_D": 1, "gamma_N": 2, "gamma_R": 3}
values[dirichlet_facets] = boundary_id["gamma_D"]
values[neumann_facets] = boundary_id["gamma_N"]
values[robin_facets] = boundary_id["gamma_R"]
mt = mesh.meshtags(msh, tdim - 1, indices, values)

ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

dirichlet_bcs = {(boundary_id["gamma_D"], u_e_expr)}
neumann_bcs = {(boundary_id["gamma_N"], lambda x: np.zeros_like(x[0]))}
# Robin BCs (alpha_R * u + kappa * \partial u \ \partial n = beta_R on Gamma_R,
# see Ern2004 p. 114)
alpha_R = fem.Constant(msh, PETSc.ScalarType(0.0))
beta_R = fem.Constant(msh, PETSc.ScalarType(0.0))
robin_bcs = {(boundary_id["gamma_R"], (alpha_R, beta_R))}

lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 1, 0)
# FIXME CHECK CONV TERM / CHANGING THIS TO VERSION WITH NORMAL
a = inner(u / delta_t, v) * dx - \
    inner(w * u, grad(v)) * dx + \
    inner(lmbda("+") * dot(w("+"), n("+")) * u("+") -
          lmbda("-") * dot(w("-"), n("-")) * u("-"), jump(v)) * dS + \
    inner(lmbda * dot(w, n) * u, v) * ds + \
    kappa * (inner(grad(u), grad(v)) * dx -
             inner(avg(grad(u)), jump(v, n)) * dS -
             inner(jump(u, n), avg(grad(v))) * dS +
             (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS)

f = fem.Constant(msh, PETSc.ScalarType(1.0))
L = inner(f + u_n / delta_t, v) * dx

for bc in dirichlet_bcs:
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    a += kappa * (- inner(grad(u), v * n) * ds(bc[0]) -
                  inner(grad(v), u * n) * ds(bc[0]) +
                  (alpha / h) * inner(u, v) * ds(bc[0]))
    L += - inner((1 - lmbda) * dot(w, n) * u_D, v) * ds(bc[0]) + \
        kappa * (- inner(u_n * n, grad(v)) * ds(bc[0]) +  # FIXME Why is this u_n?
                 (alpha / h) * inner(u_D, v) * ds(bc[0]))

for bc in neumann_bcs:
    g = fem.Function(V)
    g.interpolate(bc[1])
    L += kappa * inner(g, v) * ds(bc[0])

for bc in robin_bcs:
    alpha_R, beta_R = bc[1]
    a += kappa * inner(alpha_R * u, v) * ds(bc[0])
    L += kappa * inner(beta_R, v) * ds(bc[0])

a = fem.form(a)
L = fem.form(L)

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
