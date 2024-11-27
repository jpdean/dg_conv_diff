from dolfinx import mesh, fem, io
import ufl
from ufl import inner, dx, grad, dot, dS, jump, avg
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector


def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def u_e_expr(x):
    "Analytical solution to steady state problem from Donea and Huerta"
    gamma = w_x / kappa
    return 1 / w_x * (x[0] - (1 - np.exp(gamma * x[0])) / (1 - np.exp(gamma)))


def marker_Gamma_D(x):
    "Marker for Dirichlet boundary"
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)


def marker_Gamma_N(x):
    "Marker for Neumann boundary"
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def marker_Gamma_R(x):
    "Marker for Robin boundary"
    return np.full(x[0].shape, False)


# Simulation parameters
n = 32  # Number of cells in each direction
k = 3  # Polynomial degree
t_end = 10.0
num_time_steps = 32
# Diffusivity
# NOTE: Analytical solution contains exp(w_x / kappa), which can
# overflow for small kappa
kappa = 0.01
# Velocity field components
w_x = 1.0
w_y = 0.0

msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

V = fem.functionspace(msh, ("Discontinuous Lagrange", k))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Create function to store solution at previous time step (and interpolate
# initial condition)
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

w = fem.Constant(msh, np.array([w_x, w_y], dtype=PETSc.ScalarType))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

# Simulation constants
delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(
    msh, PETSc.ScalarType(10.0 * k**2))
kappa_const = fem.Constant(msh, PETSc.ScalarType(kappa))


# Create meshtags
tdim = msh.topology.dim
msh.topology.create_entities(tdim - 1)
facet_imap = msh.topology.index_map(tdim - 1)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
indices = np.arange(0, num_facets)
values = np.arange(0, num_facets, dtype=np.intc)
dirichlet_facets = mesh.locate_entities_boundary(msh, tdim - 1, marker_Gamma_D)
neumann_facets = mesh.locate_entities_boundary(msh, tdim - 1, marker_Gamma_N)
robin_facets = mesh.locate_entities_boundary(msh, tdim - 1, marker_Gamma_R)
boundary_id = {"Gamma_D": 1, "Gamma_N": 2, "Gamma_R": 3}
values[dirichlet_facets] = boundary_id["Gamma_D"]
values[neumann_facets] = boundary_id["Gamma_N"]
values[robin_facets] = boundary_id["Gamma_R"]
mt = mesh.meshtags(msh, tdim - 1, indices, values)

ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

# Specify boundary conditions
dirichlet_bcs = {(boundary_id["Gamma_D"], u_e_expr)}
neumann_bcs = {(boundary_id["Gamma_N"], lambda x: np.zeros_like(x[0]))}
# Robin BCs (alpha_R * u + kappa * \partial u \ \partial n = beta_R on Gamma_R,
# see Ern2004 p. 114)
alpha_R = fem.Constant(msh, PETSc.ScalarType(0.0))
beta_R = fem.Constant(msh, PETSc.ScalarType(0.0))
robin_bcs = {(boundary_id["Gamma_R"], (alpha_R, beta_R))}

# Specify weak form of the problem
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 1, 0)
a = inner(u / delta_t, v) * dx - \
    inner(w * u, grad(v)) * dx + \
    inner(2 * avg(lmbda * w * u), jump(v, n)) * dS + \
    inner(lmbda * dot(w, n) * u, v) * ds + \
    kappa_const * (inner(grad(u), grad(v)) * dx -
                   inner(avg(grad(u)), jump(v, n)) * dS -
                   inner(jump(u, n), avg(grad(v))) * dS +
                   (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS)

f = fem.Constant(msh, PETSc.ScalarType(1.0))
L = inner(f + u_n / delta_t, v) * dx

# Apply BCs
for bc in dirichlet_bcs:
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    a += kappa_const * (- inner(grad(u), v * n) * ds(bc[0]) -
                        inner(grad(v), u * n) * ds(bc[0]) +
                        (alpha / h) * inner(u, v) * ds(bc[0]))
    L += - inner((1 - lmbda) * dot(w, n) * u_D, v) * ds(bc[0]) + \
        kappa_const * (- inner(u_D * n, grad(v)) * ds(bc[0]) +
                       (alpha / h) * inner(u_D, v) * ds(bc[0]))

for bc in neumann_bcs:
    g = fem.Function(V)
    g.interpolate(bc[1])
    L += kappa_const * inner(g, v) * ds(bc[0])

for bc in robin_bcs:
    alpha_R, beta_R = bc[1]
    a += kappa_const * inner(alpha_R * u, v) * ds(bc[0])
    L += kappa_const * inner(beta_R, v) * ds(bc[0])

a = fem.form(a)
L = fem.form(L)

A = assemble_matrix(a)
A.assemble()
b = create_vector(L)

# Create solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

u_file = io.VTXWriter(msh.comm, "u.bp", [u_n._cpp_object], engine="BP4")

# Time stepping loop
t = 0.0
u_file.write(t)
for n in range(num_time_steps):
    t += delta_t.value

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp.solve(b, u_n.x.petsc_vec)
    u_n.x.scatter_forward()

    u_file.write(t)

u_file.close()

# Function spaces for exact solution
V_e = fem.functionspace(msh, ("Lagrange", k + 3))

u_e = fem.Function(V_e)
u_e.interpolate(u_e_expr)

# Compute errors
e_u = norm_L2(msh.comm, u_n - u_e)

if msh.comm.rank == 0:
    print(f"e_u = {e_u}")
