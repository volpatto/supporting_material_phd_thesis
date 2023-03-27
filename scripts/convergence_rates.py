from attrs import define
import copy as cp
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
from scipy.stats import linregress
import pandas as pd
from typing import Dict, Any, Sequence
import os


@define
class MatrixInfo(object):
    petsc_mat: Any
    is_symmetric: bool
    size: int
    nnz: int
    number_of_dofs: int


@define
class ApproximationErrorResult(object):
    values: Dict[str, float]
    degree: int
    number_of_dofs: int
    cell_size: float


def calculate_max_mesh_size(mesh: Mesh) -> float:
    """
    Convenient function to compute the max of the mesh size.
    
    This is collected from a Firedrake discussion.
    See here: https://github.com/firedrakeproject/firedrake/discussions/2547
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    hmax = project(MaxCellEdgeLength(mesh), P0)
    with hmax.dat.vec_ro as v:
        global_hmax = v.max()
        
    return global_hmax[1]


def norm_trace(v, norm_type="L2", mesh=None):
    r"""Compute the norm of ``v``.

    :arg v: a ufl expression (:class:`~.ufl.classes.Expr`) to compute the norm of
    :arg norm_type: the type of norm to compute, see below for
         options.
    :arg mesh: an optional mesh on which to compute the norm
         (currently ignored).

    Available norm types are:

    - Lp :math:`||v||_{L^p} = (\int |v|^p)^{\frac{1}{p}} \mathrm{d}s`

    """
    typ = norm_type.lower()
    p = 2
    if typ == 'l2':
        expr = inner(v, v)
    elif typ.startswith('l'):
        try:
            p = int(typ[1:])
            if p < 1:
                raise ValueError
        except ValueError:
            raise ValueError("Don't know how to interpret %s-norm" % norm_type)
    else:
        raise RuntimeError("Unknown norm type '%s'" % norm_type)

    return assemble((expr("+")**(p/2))*dS)**(1/p) + assemble((expr**(p/2))*ds)**(1/p)


def errornorm_trace(u, uh, norm_type="L2", degree_rise=None, mesh=None):
    """Compute the error :math:`e = u - u_h` in the specified norm on the mesh skeleton.

    :arg u: a :class:`.Function` or UFL expression containing an "exact" solution
    :arg uh: a :class:`.Function` containing the approximate solution
    :arg norm_type: the type of norm to compute, see :func:`.norm` for
         details of supported norm types.
    :arg degree_rise: ignored.
    :arg mesh: an optional mesh on which to compute the error norm
         (currently ignored).
    """
    urank = len(u.ufl_shape)
    uhrank = len(uh.ufl_shape)

    if urank != uhrank:
        raise RuntimeError("Mismatching rank between u and uh")

    if not isinstance(uh, function.Function):
        raise ValueError("uh should be a Function, is a %r", type(uh))

    if isinstance(u, function.Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            warning("Degree of exact solution less than approximation degree")

    return norm_trace(u - uh, norm_type=norm_type, mesh=mesh)


def exact_solutions_expressions(mesh):
    x, y = SpatialCoordinate(mesh)
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)  # original
    # p_exact = 1.0 / (2.0 * pi * pi) * sin(pi * x) * sin(pi * y)  # aloc
    # p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
    # p_exact = 0.5 / (pi * pi) * cos(pi * x) * cos(pi * y)  # Nunez
    # p_exact = x * x * x - 3 * x * y * y
    # p_exact = - (x * x / 2 - x * x * x / 3) * (y * y / 2 - y * y * y / 3)
    flux_exact = -grad(p_exact)
    return p_exact, flux_exact


def calculate_exact_solution(mesh, pressure_family, velocity_family, pressure_degree, velocity_degree, is_hdiv_space=False):
    '''
    For compatibility only. Should be removed.
    '''
    return exact_solutions_expressions(mesh)


def calculate_exact_solution_with_trace(mesh, pressure_family, velocity_family, pressure_degree, velocity_degree):
    '''
    For compatibility only. Should be removed.
    '''
    p_exact, flux_exact = exact_solutions_expressions(mesh)
    trace_exact = p_exact
    return p_exact, flux_exact, trace_exact


def solve_with_static_condensation(
    a: Function, L: Function, primal_function_space: FunctionSpace, trace_function_space: FunctionSpace
) -> Function:
    """
    Alternative way with Slate.
    """
    V = primal_function_space
    T = trace_function_space
    
    # Local computations (solving explicitly with Slate)
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))
    A = _A.blocks
    Fc = _F.blocks
    schur_factor = A[1, 0] * A[0, 0].inv
    Sexp = A[1, 1] - schur_factor * A[0, 1]
    Eexp = Fc[1] - schur_factor * Fc[0]
    S = assemble(Sexp)
    E = assemble(Eexp)
    lambda_local = Function(T)

    # Solving for the multiplier
    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
    solve(S, lambda_local, E, solver_parameters=solver_parameters)

    # Recovering the solution for the primal variable
    Lambda = AssembledVector(lambda_local)
    p_h = Function(V)
    A00 = A[0, 0]
    p_sys = A00.solve(Fc[0] - A[0, 1] * Lambda, decomposition="FullPivLU")
    assemble(p_sys, p_h)
    
    return p_h, lambda_local


def solve_primal_problem_with_eigen(
    a: Function, L: Function, primal_function_space: FunctionSpace
) -> Function:
    V = primal_function_space
    
    # Local computations (solving explicitly with Slate)
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))
    A = _A.blocks
    F = _F.blocks
    S = assemble(A[0, 0])
    E = assemble(F[0])
    p_sol = Function(V)

    # Solving
    p_sys = A[:, :].solve(F[:], decomposition="FullPivLU")
    assemble(p_sys, p_sol)
    
    return p_sol


def solve_poisson_cg(mesh, degree=1, use_quads=False):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bcs = DirichletBC(V, project(exact_solution, V), "on_boundary")

    # Variational form
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs, constant_jacobian=False)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_ls(mesh, degree=1):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    h = CellDiameter(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    # bcs = DirichletBC(W[0], project(sigma_e, U), "on_boundary")
    # bcs = []
    bcs = DirichletBC(W[1], exact_solution, "on_boundary")

    # Stabilization parameters
    delta_1 = Constant(1)
    delta_2 = Constant(1)
    delta_3 = Constant(1)

    # Least-squares terms
    a = delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L = delta_2 * f * div(v) * dx

    # Weakly imposed BC
    huge_number = 1e10
    nitsche_penalty = Constant(huge_number)
    p_e = exact_solution
    delta_4 = (nitsche_penalty / h)
    # delta_4 = Constant(1)
    a += delta_4 * p * q * ds
    L += delta_4 * p_e * q * ds

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(W)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h = solution.split()

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_e, sigma_h, 
        norm_type="L2"
    )

    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = W.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_dls(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bcs = []

    # Least-Squares weights
    delta = Constant(1.0e0)
    delta_1 = delta
    delta_2 = delta
    delta_3 = delta * Constant(1)
    nitsche_penalty = Constant(1e10)
    beta = nitsche_penalty / h
    delta_4 = beta
    # delta_4 = Constant(1)
    eta_p = Constant(1)
    eta_u = Constant(1) * eta_p
    delta_5 = eta_p / h
    delta_6 = eta_u / h

    # Least-squares terms
    a = delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    a += avg(delta_5) * dot(jump(p, n), jump(q, n)) * dS
    a += avg(delta_6) * (jump(u, n) * jump(v, n)) * dS
    a += delta_4 * p * q * ds  # may decrease convergente rates
    # RHS
    L = delta_2 * f * div(v) * dx
    L += delta_4 * exact_solution * q * ds  # may decrease convergente rates

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(W)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h = solution.split()

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_e, sigma_h, 
        norm_type="L2"
    )

    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = W.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_cgls(mesh, degree=1):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bc1 = DirichletBC(W[0], project(sigma_e, U), "on_boundary")
    bc2 = DirichletBC(W[1], project(exact_solution, V), "on_boundary")
    bcs = [bc1, bc2]

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    L = -f * q * dx - exact_solution * dot(v, n) * ds
    # Stabilizing terms
    a += -0.5 * inner((u + grad(p)), v + grad(q)) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx
    L += 0.5 * f * div(v) * dx

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(W)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h = solution.split()

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_e, sigma_h, 
        norm_type="L2"
    )

    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = W.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_sipg(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Edge stabilizing parameter
    beta0 = Constant(1e2)
    beta = beta0 / h

    # Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
    s = Constant(-1)

    # Classical volumetric terms
    a = inner(grad(p), grad(q)) * dx
    L = f * q * dx
    # DG edge terms
    a += s * dot(jump(p, n), avg(grad(q))) * dS - dot(avg(grad(p)), jump(q, n)) * dS
    # Edge stabilizing terms
    a += avg(beta) * dot(jump(p, n), jump(q, n)) * dS
    # Weak boundary conditions
    a += s * dot(p * n, grad(q)) * ds - dot(grad(p), q * n) * ds
    a += beta * p * q * ds
    L += beta * exact_solution * q * ds

    # Solving the system
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a, L, solution, bcs=[])
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        div(-grad(exact_solution)), project(div(-grad(p_h)), V), 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = V.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_dls_primal(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    element_family = 'DQ' if use_quads else 'DG'
    mesh_cell = mesh.ufl_cell()
    DiscontinuousElement = FiniteElement(element_family, mesh_cell, degree)
    U = VectorFunctionSpace(mesh, DiscontinuousElement)
    V = FunctionSpace(mesh, DiscontinuousElement)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = MinCellEdgeLength(mesh)
    # h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = exact_solutions_expressions(mesh)

    # Forcing function
    f = div(-grad(exact_solution))

    # Stabilizing parameter
    # PDLS
    penalty_constant = 1e1
    penalty_constant_ip = penalty_constant
    delta_base = Constant(penalty_constant * degree * degree)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    delta_1 = Constant(1)
    delta_2 = delta_base / h / h
    # delta_3 = (1 / delta_base) * h * Constant(1)
    delta_3 = (1 / delta_base) * Constant(1)
    delta_4 = delta_2
    
    # DLS-IP
    # penalty_constant = 1e6
    # # delta_base = Constant(penalty_constant)
    # delta_base = Constant(penalty_constant * degree * degree)
    # enable_dg_ip = Constant(1)  # enable (1) or disable (0)
    # delta_0 = delta_base / delta_base * enable_dg_ip
    # # delta_1 = Constant(0.5) * h * h
    # delta_1 = Constant(1.0) * h * h
    # delta_2 = delta_base / h / h
    # # delta_1 = Constant(1.0)
    # # delta_2 = delta_base / h
    # delta_3 = (1 / delta_base) * Constant(1)
    # delta_4 = delta_2

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Residual definition
    Lp = div(u)
    Lq = div(v)

    # Classical DG-IP term
    a = delta_0 * dot(grad(p), grad(q)) * dx
    L = delta_0 * f * q * dx

    # Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
    s = Constant(-1)

    # DG edge terms
    a += s * delta_0 * dot(jump(p, n), avg(v)) * dS
    a += -delta_0 * dot(avg(u), jump(q, n)) * dS

    # Mass balance least-square
    a += delta_1 * Lp * Lq * dx
    L += delta_1 * f * Lq * dx

    # Hybridization terms
    a += avg(delta_2) * dot(jump(p, n=n), jump(q, n=n)) * dS
    a += delta_4 * (p - exact_solution) * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS

    # Ensuring that the formulation is properly decomposed in LHS and RHS
    F = a - L
    a_form = lhs(F)
    L_form = rhs(F)

    # Solving the system
    solver_parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_max_it": 5,
        "snes_lag_jacobian": -2,
        "snes_lag_preconditioner": -2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-16,
        "snes_atol": 1e-25
    }
    solution = Function(V)
    problem = LinearVariationalProblem(
        a_form, L_form, solution, bcs=[],
    )
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        div(-grad(exact_solution)), project(div(-grad(p_h)), V), 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = V.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_interpolator_primal(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    element_family = 'DQ' if use_quads else 'DG'
    DiscontinuousElement = FiniteElement(element_family, mesh.ufl_cell(), degree)
    # CG = FiniteElement("CG", mesh.ufl_cell(), degree)
    # CG = CG['facet']
    # DiscontinuousElement = BrokenElement(CG)
    U = VectorFunctionSpace(mesh, DiscontinuousElement)
    V = FunctionSpace(mesh, DiscontinuousElement)

    # Exact solution
    exact_solution, sigma_e = exact_solutions_expressions(mesh)

    # Interpolating the solution
    p_h = interpolate(exact_solution, V)
    sigma_h = interpolate(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        div(-grad(exact_solution)), project(div(-grad(p_h)), V), 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = V.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_interpolator_mixed(mesh, degree=1, is_multiplier_continuous=False):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)

    # Exact solution
    exact_solution, sigma_e, p_expression = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Interpolating the solution
    p_h = interpolate(exact_solution, V)
    lambda_h = interpolate(exact_solution, T)
    sigma_h = interpolate(sigma_e, U)

    # Calculating L2-error for primal variable
    # p_error_L2 = errornorm_trace(project(p_expression, T), lambda_h, norm_type="L2")
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_h, sigma_e, 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = V.dim()  # does not matter
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_cls_primal(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'CG' if use_quads else 'CG'
    velocity_family = 'CG' if use_quads else 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))
    
    # Stabilizing parameter (for PhD results)
    element_size_factor = h
    beta0 = Constant(1e5)
    beta = beta0 / h
    penalty_constant = 1e0
    delta_base = Constant(penalty_constant * degree * degree)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    delta_1 = Constant(1)
    delta_2 = delta_base / h / h * Constant(1)
    delta_3 = Constant(1) * beta * h
    # delta_3 = Constant(1e-2)

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Residual definition
    Lp = div(u)
    Lq = div(v)

    # Classical DG-IP term
    a = delta_0 * dot(grad(p), grad(q)) * dx
    L = delta_0 * f * q * dx

    # Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
    s = Constant(-1)

    # DG edge terms
    a += s * delta_0 * dot(jump(p, n), avg(v)) * dS
    a += -delta_0 * dot(avg(u), jump(q, n)) * dS

    # Mass balance least-square
    a += delta_1 * Lp * Lq * dx
    L += delta_1 * f * Lq * dx

    # Hybridization terms
    a += delta_2 * (p - exact_solution) * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS

    # Ensuring that the formulation is properly decomposed in LHS and RHS
    F = a - L
    a_form = lhs(F)
    L_form = rhs(F)

    # Solving the system
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a_form, L_form, solution, bcs=[])
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_h, sigma_e, 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = V.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_sdhm(
    mesh, 
    degree=1,
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Hybridization parameter
    beta_0 = Constant(0.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilization parameters
    delta_0 = Constant(-1)
    delta_1 = Constant(-0.5)  #* h * h
    delta_2 = Constant(0.5)
    delta_3 = Constant(0.5)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
    a += delta_0 * div(u) * q * dx
    L = delta_0 * f * q * dx

    # Least-squares terms
    a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L += delta_2 * f * div(v) * dx

    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS

    # Nitsche's term to transmission condition edge stabilization
    a += -beta('+') * (p('+') - lambda_h('+')) * q('+') * dS

    # Weakly imposed BC
    # a += lambda_h * dot(v, n) * ds  # required term
    L += -exact_solution * dot(v, n) * ds  # required as the above, but just one of them should be used (works for continuous multiplier)
    a += -beta * p * q * ds  # required term... note that u (the unknown) is used
    L += -beta * exact_solution * q * ds  # Required, this one is paired with the above term
    a += dot(u, n) * mu_h * ds
    a += -lambda_h * mu_h * ds  # Classical required term
    L += -exact_solution * mu_h * ds  # Pair for the above classical required term

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        },
    }
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    # p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")
    # interpolate(exact_trace, T)
    p_error_L2 = errornorm_trace(project(exact_trace, T), lambda_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_h, sigma_e, 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = T.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_hdg(
    mesh, 
    degree=1,
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
    a += -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    L = f * q * dx
    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS
    # Weakly imposed BC
    a += lambda_h * dot(v, n) * ds
    a += dot(u_hat, n) * q * ds

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        },
    }
    problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")
    # p_error_L2 = errornorm_trace(interpolate(exact_trace, T), lambda_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_h, sigma_e, 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = T.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_hdg_sdhm(
    mesh, 
    degree=1,
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Hybridization parameter
    beta_0 = Constant(1.0e1)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilization parameters
    delta_1 = Constant(-0.5)  #* h * h
    delta_2 = Constant(0.5) * h * h
    delta_3 = Constant(0.5) * h * h

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
    a += -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    L = f * q * dx

    # Least-squares terms
    a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L += delta_2 * f * div(v) * dx

    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS

    # Weakly imposed BC
    # a += lambda_h * dot(v, n) * ds  # required term
    L += -exact_solution * dot(v, n) * ds  # required as the above, but just one of them should be used (works for continuous multiplier)
    a += dot(u, n) * q * ds
    a += dot(u, n) * mu_h * ds
    a += beta * p * q * ds  # required term... note that u (the unknown) is used
    L += beta * exact_solution * q * ds  # Required, this one is paired with the above term
    a += -lambda_h * mu_h * ds  # Classical required term
    L += -exact_solution * mu_h * ds  # Pair for the above classical required term

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        },
    }
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")
    # p_error_L2 = errornorm_trace(interpolate(exact_trace, T), lambda_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        sigma_h, sigma_e, 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = T.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_lsh(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))
    # f = Function(V).interpolate(f)

    # BCs
    p_exact = Constant(0)
    # bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")
    bcs = DirichletBC(W.sub(2), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta_1 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilizing parameter
    delta = Constant(1)
    # delta = h * h
    delta_0 = delta * Constant(0)  #* h * h
    # delta_1 = delta
    delta_1 = Constant(1)
    delta_2 = delta
    delta_3 = delta * Constant(1)  #* h * h
    delta_4 = beta_1 / h * Constant(1)
    # delta_4 = Constant(1)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Flux least-squares
    a = (
        (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
        * delta_1
        * dx
    )
    # These terms below are unsymmetric
    a += delta_1("+") * jump(u_hat, n=n) * q("+") * dS
    a += delta_1 * dot(u, n) * q * ds
    a += delta_1 * beta * (p - p_exact) * q * ds
    a += delta_1("+") * lambda_h("+") * jump(v, n=n) * dS
    # a += delta_1 * lambda_h * dot(v, n) * ds
    L = -delta_1 * p_exact * dot(v, n) * ds   # required as the above, but just one of them should be used (works for continuous multiplier)

    # Flux Least-squares as in DG
    a += delta_0 * inner(u + grad(p), v + grad(q)) * dx

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx
    L += delta_2 * f * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS

    a += Constant(1) * delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    a += delta_4 * (p - p_exact) * q * ds
    a += delta_4 * (lambda_h - p_exact) * mu_h * ds

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = T.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_lsh_primal(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    p_degree = degree
    V = FunctionSpace(mesh, pressure_family, p_degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    solution = Function(W)
    p, lambda_h = split(solution)
    # p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    # h = CellDiameter(mesh)
    h = MinCellEdgeLength(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    p_exact = exact_trace
    # bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")
    bcs = DirichletBC(W.sub(1), p_exact, "on_boundary")

    # Hybridization parameter
    penalty_constant = 1e0
    # penalty_term = Constant(1.0 * penalty_constant * p_degree * p_degree)
    penalty_term = Constant(penalty_constant)
    beta_0 = penalty_term * Constant(1e1)
    beta = beta_0 / h
    
    # PDLS-h
    delta_base = Constant(1)
    delta_0 = Constant(0)
    delta_1 = delta_base * Constant(1)
    delta_2 = penalty_term / h / h  # triangles
    # delta_2 = penalty_term
    # delta_2 = penalty_term / h  # quad
    
    # PDGLS-h
    # delta_base = Constant(1)
    # delta_0 = Constant(1)
    # # delta_1 = delta_base * Constant(1) * h * h # triangles
    # delta_1 = delta_base * Constant(1)  # quads
    # delta_2 = penalty_term / h / h  # triangles
    # # delta_2 = penalty_term / h  # quads
    # # delta_2 = penalty_term
    # # delta_2 = Constant(1)  # quads

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Symmetry parameter: s = -1 (symmetric) or s = 1 (unsymmetric). Disable with 0.
    s = Constant(-1)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Classical term
    a = delta_0 * dot(grad(p), grad(q)) * dx - delta_0('+') * jump(u_hat, n) * q("+") * dS
    # a += delta_0 * dot(u_hat, n) * q * ds
    a += -delta_0 * dot(u, n) * q * ds +  delta_0 * beta * (p - exact_solution) * q * ds  # expand u_hat product in ds
    L = delta_0 * f * q * dx

    # Mass balance least-squares
    a += delta_1 * div(u) * div(v) * dx
    # a += delta_1 * inner(curl(u), curl(v)) * dx
    L += delta_1 * f * div(v) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    # a += mu_h * (lambda_h - p_exact) * ds
    # a += mu_h * dot(u_hat - grad(exact_solution), n) * ds  # is this worthy?
    a += delta_2("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    # a += delta_2 * (p - p_exact) * (q - mu_h) * ds
    a += delta_2 * (p - exact_solution) * q * ds  # needed if not included as strong BC
    a += delta_2 * (lambda_h - exact_solution) * mu_h * ds  # needed if not included as strong BC

    # Consistent symmetrization
    a += s * delta_0('+') * jump(v, n) * (p('+') - lambda_h("+")) * dS
    a += s * delta_0 * dot(v, n) * (p - exact_solution) * ds

    F = a - L

    params = {
        'snes_type': 'ksponly',
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0",
        "condensed_field": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_max_it": 5,
            "snes_lag_jacobian": -2,
            "snes_lag_preconditioner": -2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": "2000",
            "snes_rtol": 1e-16,
            "snes_atol": 1e-25
        },
    }
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    p_h, lambda_h = solution.split()
    sigma_h = Function(U, name='Velocity')
    sigma_h.project(-grad(p_h))
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    # Calculating L2-error for LS term
    LS_error_L2 = errornorm(
        div(-grad(exact_solution)), project(div(-grad(p_h)), V), 
        norm_type="L2"
    )
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
        'LS-error p': LS_error_L2,
    }
    num_dofs = T.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def estimate_error_slope_with_regression(
    mesh_sizes: Sequence[float], 
    error_values: Sequence[float], 
    degree: int, 
    error_name: str = ''
) -> float:
    """
    Convenient function to estimate convergence rates in runtime using linear regressions.
    """
    mesh_sizes_arr = np.array(mesh_sizes)
    mesh_sizes_arr_log10 = np.log10(mesh_sizes_arr)
    
    error_values_arr = np.array(error_values)
    error_values_arr_log10 = np.log10(error_values_arr)
    
    error_slope, _, _, _, _ = linregress(mesh_sizes_arr_log10, error_values_arr_log10)
    PETSc.Sys.Print(f"\nDegree {degree}: slope {error_name} {error_slope:.3f}")
    
    return error_slope


def _remove_empty_entries_in_errors_dict(error_dict_data: Dict[str, Sequence]) -> Dict[str, Sequence]:
    new_error_dict_data = cp.deepcopy(error_dict_data)
    for entry_key, value in error_dict_data.items():
        if value == []:
            del new_error_dict_data[entry_key]
    return new_error_dict_data


def compute_convergence_hp(
    solver,
    min_degree=1,
    max_degree=4,
    numel_xy=(2, 4, 8, 16, 32, 64, 128, 256),
    quadrilateral=True,
    name="",
    reorder_mesh=False,
    **kwargs
):
    computed_errors_dict = {
        "Element": list(),
        "Degree": list(),
        "Cells": list(),
        "log Cells": list(),
        "Mesh size": list(),
        "Num DOFs": list(),
        "log Num DOFs": list(),
        "L2-error p": list(),
        "log L2-error p": list(),
        "L2-error p order": list(),
        "H1-error p": list(),
        "log H1-error p": list(),
        "H1-error p order": list(),
        "L2-error u": list(),
        "log L2-error u": list(),
        "L2-error u order": list(),
        "Hdiv-error u": list(),
        "log Hdiv-error u": list(),
        "Hdiv-error u order": list(),
        "L2-error trace": list(),
        "log L2-error trace": list(),
        "L2-error trace order": list(),
        "LS-error p": list(),
        "log LS-error p": list(),
        "LS-error p order": list(),
    }
    element_kind = "Quad" if quadrilateral else "Tri"
    for degree in range(min_degree, max_degree):
        # The dict below is jut to estimate the convergence rate in runtime
        errors_arrays = {
            "L2-error p": list(),
            "log L2-error p": list(),
            "L2-error p order": list(),
            "H1-error p": list(),
            "log H1-error p": list(),
            "H1-error p order": list(),
            "L2-error u": list(),
            "log L2-error u": list(),
            "L2-error u order": list(),
            "Hdiv-error u": list(),
            "log Hdiv-error u": list(),
            "Hdiv-error u order": list(),
            "L2-error trace": list(),
            "log L2-error trace": list(),
            "L2-error trace order": list(),
            "LS-error p": list(),
            "log LS-error p": list(),
            "LS-error p order": list(),
        }
        num_cells = np.array([])
        mesh_size = np.array([])
        for n in numel_xy:
            nel_x = nel_y = n
            mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=quadrilateral, reorder=reorder_mesh)
            current_num_cells = mesh.num_cells()
            num_cells = np.append(num_cells, current_num_cells)
            current_mesh_size = mesh.cell_sizes.dat.data_ro.min() if not quadrilateral else 1 / n
            mesh_size = np.append(mesh_size, current_mesh_size)

            current_errors_data = solver(mesh=mesh, degree=degree, **kwargs)
            
            computed_errors_dict["Element"].append(element_kind)
            computed_errors_dict["Degree"].append(current_errors_data.degree)
            computed_errors_dict["Cells"].append(current_num_cells)
            computed_errors_dict["log Cells"].append(np.log10(current_num_cells) / 2)
            computed_errors_dict["Mesh size"].append(current_errors_data.cell_size)
            computed_errors_dict["Num DOFs"].append(current_errors_data.number_of_dofs)
            computed_errors_dict["log Num DOFs"].append(np.log10(current_errors_data.number_of_dofs))
            
            errors_values = current_errors_data.values
            for error_type, error_value in errors_values.items():
                if error_value is None:
                    continue
                
                is_log_in_string = 'log' in error_type
                if not is_log_in_string:
                    computed_errors_dict[error_type].append(error_value)
                    errors_arrays[error_type].append(error_value)
                    computed_errors_dict[f'log {error_type}'].append(np.log10(error_value))
                    errors_arrays[f'log {error_type}'].append(np.log10(error_value))


        PETSc.Sys.Print("\n--------------------------------------")

        num_mesh_evaluations = len(numel_xy)
        for error_type, error_values in errors_arrays.items():
            if error_values == []:
                continue
            is_log_in_string = 'log' in error_type
            if not is_log_in_string:
                error_slope = estimate_error_slope_with_regression(
                    mesh_size, error_values, current_errors_data.degree, error_type
                )
                computed_errors_dict[f"{error_type} order"] += num_mesh_evaluations * [np.abs(error_slope)]

        PETSc.Sys.Print("\n--------------------------------------")

    dir_name = f"./conv_rate_results/conv_results_{name}"
    os.makedirs(dir_name, exist_ok=True)
    clean_computed_errors_dict = _remove_empty_entries_in_errors_dict(computed_errors_dict)
    df_computed_errors = pd.DataFrame(data=clean_computed_errors_dict)
    path_to_save_results = f"{dir_name}/errors.csv"
    df_computed_errors.to_csv(path_to_save_results)

    return


# Solver options
available_solvers = {
    "cg": solve_poisson_cg,
    "cgls": solve_poisson_cgls,
    "sdhm": solve_poisson_sdhm,
    "hdg_sdhm": solve_poisson_hdg_sdhm,
    "clsq": solve_poisson_ls,
    "dls": solve_poisson_dls,
    "lsh": solve_poisson_lsh,
    "new_lsh_primal": solve_poisson_lsh_primal,
    "pdlsh": solve_poisson_lsh_primal,
    "pdglsh": solve_poisson_lsh_primal,
    "pdls": solve_poisson_dls_primal,
    "dls_ip": solve_poisson_dls_primal,
    "interpolator_primal": solve_poisson_interpolator_primal,
    "interpolator_mixed": solve_poisson_interpolator_mixed,
    "cls_ip_primal": solve_poisson_cls_primal,
    "hdg": solve_poisson_hdg,
    "sipg": solve_poisson_sipg,
}

degree = 1
last_degree = 4
mesh_quad = [False, True]  # Triangles, Quads
elements_for_each_direction = [4, 8, 16, 32, 64]
for element in mesh_quad:
    for current_solver in available_solvers:

        if element:
            element_kind = "quad"
        else:
            element_kind = "tri"

        # Setting the output file name
        name = f"{current_solver}_{element_kind}"

        # Selecting the solver
        solver = available_solvers[current_solver]

        PETSc.Sys.Print("*******************************************\n")
        PETSc.Sys.Print(f"*** Begin case: {name} ***\n")

        # Performing the convergence study
        compute_convergence_hp(
            solver,
            min_degree=degree,
            max_degree=last_degree + 1,
            quadrilateral=element,
            numel_xy=elements_for_each_direction,
            name=name,
            reorder_mesh=True
        )

        PETSc.Sys.Print(f"\n*** End case: {name} ***")
        PETSc.Sys.Print("*******************************************\n")
