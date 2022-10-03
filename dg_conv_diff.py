from dolfinx import mesh, fem, io
import ufl
from ufl import inner, dx, grad, dot, dS, jump, ds, avg
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

n = 16
k = 1
t_end = 1.0
num_time_steps = 32

msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

V = fem.FunctionSpace(msh, ("Lagrange", k))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

w = ufl.as_vector((1.0, 0.0))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(
    msh, PETSc.ScalarType(10.0 * k**2))  # TODO Check k dependency
kappa = fem.Constant(msh, PETSc.ScalarType(0.01))

w_uw = (dot(w, n) + abs(dot(w, n))) / 2.0
# FIXME CHECK CONV TERM / CHANGING THIS TO VERSION WITH NORMAL
a = fem.form(inner(u / delta_t, v) * dx -
             inner(w * u, grad(v)) * dx +
             inner(w_uw("+") * u("+") - w_uw("-") * u("-"), jump(v)) * dS +
             inner(w_uw * u, v) * ds +
             kappa * (inner(grad(u), grad(v)) * dx -
                      inner(avg(grad(u)), jump(v, n)) * dS -
                      inner(jump(u, n), avg(grad(v))) * dS +
                      (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS -
                      inner(grad(u), v * n) * ds -
                      inner(grad(v), u * n) * ds +
                      (alpha / h) * inner(u, v) * ds))

f = fem.Constant(msh, PETSc.ScalarType(0.0))
L = fem.form(inner(f + u_n / delta_t, v) * dx)

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

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    fem.petsc.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp.solve(b, u_n.vector)
    u_n.x.scatter_forward()

    u_file.write(t)

u_file.close()
