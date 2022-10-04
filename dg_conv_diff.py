# TODO Test diffusion, advection, advection-diffusion, and add Neumann
# BC for Diffusion

from dolfinx import mesh, fem, io
import ufl
from ufl import inner, dx, grad, dot, dS, jump, ds, avg
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


class TimeDependentExpression():
    """Simple class to represent time dependent functions"""

    def __init__(self, expression):
        self.t = 0.0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)


n = 32
k = 1
t_end = 1.0
num_time_steps = 32

msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

V = fem.FunctionSpace(msh, ("Lagrange", k))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

u_D_expr = TimeDependentExpression(
    lambda x, t: np.sin(np.pi * t) * np.cos(np.pi * x[1]))
u_D = fem.Function(V)
u_D.interpolate(u_D_expr)

w = fem.Constant(msh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(
    msh, PETSc.ScalarType(10.0 * k**2))  # TODO Check k dependency
kappa = fem.Constant(msh, PETSc.ScalarType(1.0))

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
                      inner(grad(u), v * n) * ds -
                      inner(grad(v), u * n) * ds +
                      (alpha / h) * inner(u, v) * ds))

f = fem.Constant(msh, PETSc.ScalarType(0.0))
L = fem.form(inner(f + u_n / delta_t, v) * dx -
             inner((1 - lmbda) * dot(w, n) * u_D, v) * ds -
             inner(u_n * n, grad(v)) * ds +
             (alpha / h) * inner(u_D, v) * ds)

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

    u_D_expr.t = t
    u_D.interpolate(u_D_expr)

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    fem.petsc.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp.solve(b, u_n.vector)
    u_n.x.scatter_forward()

    u_file.write(t)

u_file.close()
