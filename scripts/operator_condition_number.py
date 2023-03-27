import attr
import os
os.environ["OMP_NUM_THREADS"] = "8"

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import svd, eig
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csr_matrix
from slepc4py import SLEPc
import pandas as pd
from tqdm import tqdm

matplotlib.use('Agg')


@attr.s
class ConditionNumberResult(object):
    form = attr.ib()
    assembled_form = attr.ib()
    condition_number = attr.ib()
    sparse_operator = attr.ib()
    number_of_dofs = attr.ib()
    nnz = attr.ib()
    is_operator_symmetric = attr.ib()
    bcs = attr.ib(default=list())
    assembled_condensed_form = attr.ib(default=None)    


def check_symmetric(A: np.ndarray, rtol: float=1e-05, atol: float=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def check_max_unsymmetric_relative_discrepancy(A):
    rel_discrepancy = np.linalg.norm(A - A.T, np.inf) / np.linalg.norm(A, np.inf)
    return rel_discrepancy.max()


def assemble_form_to_petsc_matrix(form, bcs=[], mat_type="aij"):
    assembled_form = assemble(form, bcs=bcs, mat_type=mat_type)
    petsc_mat = assembled_form.M.handle
    return petsc_mat


def convert_petsc_matrix_to_dense_array(petsc_mat) -> np.ndarray:
    size = petsc_mat.getSize()
    matrix_csr = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    matrix_csr.eliminate_zeros()
    matrix_numpy = matrix_csr.toarray()
    return matrix_numpy


def generate_dense_array_from_form(
    form, 
    bcs=[], 
    mat_type="aij"
):
    petsc_mat = assemble_form_to_petsc_matrix(form, bcs=bcs, mat_type=mat_type)
    numpy_mat = convert_petsc_matrix_to_dense_array(petsc_mat)
    return numpy_mat


def calculate_matrix_symmetric_part(dense_numpy_matrix: np.ndarray) -> np.ndarray:
    A = dense_numpy_matrix
    A_T = A.T
    sym_A = (1. / 2.) * (A + A_T)
    return sym_A


def calculate_numpy_matrix_all_eigenvalues(numpy_matrix: np.ndarray, is_sparse: bool=False):
    if is_sparse:
        sparse_mat = csr_matrix(numpy_matrix)
        sparse_mat.eliminate_zeros()
        num_dofs = sparse_mat.shape[0]
        eigenvalues = eigs(sparse_mat, k=num_dofs - 1, return_eigenvectors=False)
    else:
        eigenvalues = eig(numpy_matrix, right=False)
    return eigenvalues


def filter_real_part_in_array(array: np.ndarray, imag_threshold: float = 1e-5) -> np.ndarray:
    """Utility function to filter real part in a numpy array.

    :param array: 
        Array with real and complex numbers.

    :param imag_threshold:
        Threshold to cut off imaginary part in complex number.

    :return:
        Filtered array with only real numbers.
    """
    real_part_array = array.real[abs(array.imag) < 1e-5]
    return real_part_array


def calculate_condition_number(
    A, 
    num_of_factors, 
    backend: str = "scipy",
    use_sparse: bool = False,
    zero_tol: float = 1e-5
):
    backend = backend.lower()

    if backend == "scipy":
        size = A.getSize()
        Mnp = csr_matrix(A.getValuesCSR()[::-1], shape=size)
        Mnp.eliminate_zeros()

        if use_sparse:
            singular_values = svds(
                A=Mnp, 
                k=num_of_factors, 
                which="LM", 
                maxiter=5000, 
                return_singular_vectors=False, 
                solver="lobpcg"
            )
        else:
            M = Mnp.toarray()
            singular_values = svd(M, compute_uv=False, check_finite=False)

        singular_values = singular_values[singular_values > zero_tol]

        condition_number = singular_values.max() / singular_values.min()
    elif backend == "slepc":
        S = SLEPc.SVD()
        S.create()
        S.setOperator(A)
        S.setType(SLEPc.SVD.Type.LAPACK)
        S.setDimensions(nsv=num_of_factors)
        S.setTolerances(max_it=5000)
        S.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
        S.solve()

        num_converged_values = S.getConverged()
        singular_values_list = list()
        if num_converged_values > 0:
            for i in range(num_converged_values):
                singular_value = S.getValue(i)
                singular_values_list.append(singular_value)
        else:
            raise RuntimeError("SLEPc SVD has not converged.")

        singular_values = np.array(singular_values_list)

        singular_values = singular_values[singular_values > zero_tol]
        condition_number = singular_values.max() / singular_values.min()
    else:
        raise NotImplementedError("The required method for condition number estimation is currently unavailable.")

    return condition_number


def solve_poisson_cg(mesh, degree=1, use_quads=False):
    # Function space declaration
    V = FunctionSpace(mesh, "CG", degree)

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Dirichlet BCs
    bcs = DirichletBC(V, 0.0, "on_boundary")

    # Variational form
    a = inner(grad(u), grad(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = V.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )

    return result


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
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    bcs = DirichletBC(W[0], sigma_e, "on_boundary")

    # Stabilization parameters
    delta_1 = Constant(1)
    delta_2 = Constant(1)
    delta_3 = Constant(1)

    # Least-squares terms
    a = delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


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
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    bcs = DirichletBC(W[0], sigma_e, "on_boundary")

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    # Stabilizing terms
    a += -0.5 * inner((u + grad(p)), v + grad(q)) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_sipg(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

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
    a += beta("+") * dot(jump(p, n), jump(q, n)) * dS
    # Weak boundary conditions
    a += s * dot(p * n, grad(q)) * ds - dot(grad(p), q * n) * ds
    a += beta * p * q * ds

    A = assemble(a, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = V.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_pdls(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    mesh_cell = mesh.ufl_cell()
    DiscontinuousElement = FiniteElement(pressure_family, mesh_cell, degree)
    U = VectorFunctionSpace(mesh, DiscontinuousElement)
    V = FunctionSpace(mesh, DiscontinuousElement)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = MinCellEdgeLength(mesh)

    # Stabilizing parameter
    # PDLS
    penalty_constant = 1e1
    penalty_constant_ip = penalty_constant
    # delta_base = Constant(penalty_constant * degree * degree)
    delta_base = Constant(1)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    delta_1 = Constant(1)
    delta_2 = delta_base / h / h
    # delta_3 = (1 / delta_base) * h * Constant(1)
    delta_3 = (1 / delta_base) * Constant(1)
    delta_4 = delta_2

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Residual definition
    Lp = div(u)
    Lq = div(v)

    # Classical DG-IP term
    a = delta_0 * dot(grad(p), grad(q)) * dx

    # Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
    s = Constant(-1)

    # DG edge terms
    a += s * delta_0 * dot(jump(p, n), avg(v)) * dS
    a += -delta_0 * dot(avg(u), jump(q, n)) * dS

    # Mass balance least-square
    a += delta_1 * Lp * Lq * dx

    # Hybridization terms
    a += avg(delta_2) * dot(jump(p, n=n), jump(q, n=n)) * dS
    a += delta_4 * p * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS

    # Ensuring that the formulation is properly decomposed in LHS and RHS
    a_form = lhs(a)

    A = assemble(a_form, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = V.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a_form,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_pdls_ip(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    mesh_cell = mesh.ufl_cell()
    DiscontinuousElement = FiniteElement(pressure_family, mesh_cell, degree)
    U = VectorFunctionSpace(mesh, DiscontinuousElement)
    V = FunctionSpace(mesh, DiscontinuousElement)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = MinCellEdgeLength(mesh)
    
    # DLS-IP
    penalty_constant = 1e6
    # delta_base = Constant(penalty_constant)
    delta_base = Constant(penalty_constant)
    enable_dg_ip = Constant(1)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    # delta_1 = Constant(0.5) * h * h
    delta_1 = Constant(1.0) * h * h
    delta_2 = delta_base / h / h
    # delta_1 = Constant(1.0)
    # delta_2 = delta_base / h
    delta_3 = (1 / delta_base) * Constant(1)
    delta_4 = delta_2

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Residual definition
    Lp = div(u)
    Lq = div(v)

    # Classical DG-IP term
    a = delta_0 * dot(grad(p), grad(q)) * dx

    # Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
    s = Constant(-1)

    # DG edge terms
    a += s * delta_0 * dot(jump(p, n), avg(v)) * dS
    a += -delta_0 * dot(avg(u), jump(q, n)) * dS

    # Mass balance least-square
    a += delta_1 * Lp * Lq * dx

    # Hybridization terms
    a += avg(delta_2) * dot(jump(p, n=n), jump(q, n=n)) * dS
    a += delta_4 * p * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS

    # Ensuring that the formulation is properly decomposed in LHS and RHS
    a_form = lhs(a)

    A = assemble(a_form, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = V.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a_form,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


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
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    # bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0

    # Jump stabilizing parameters based on Badia-Codina stabilized dG method
    # L0 = 1
    # eta_p = L0 * h_avg  # method B in the Badia-Codina paper
    eta_p = 1
    # eta_p = L0 * L0  # method D in the Badia-Codina paper
    # eta_u = h_avg / L0  # method B in the Badia-Codina paper
    eta_u = 1
    # eta_u_bc = h / L0  # method B in the Badia-Codina paper
    eta_u_bc = 1

    # Least-Squares weights
    delta = Constant(1.0)
    # delta = h
    delta_1 = delta
    delta_2 = delta
    delta_3 = delta
    nitsche_penalty = Constant(1e10)
    delta_4 = nitsche_penalty / h
    delta_5 = 1 / h
    delta_6 = 1 / h

    # Least-squares terms
    a = delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    # Edge stabilizing terms
    # ** Badia-Codina based (better results) **
    a += eta_p * avg(delta_5) * dot(jump(p, n), jump(q, n)) * dS
    a += eta_u * avg(delta_6) * (jump(u, n) * jump(v, n)) * dS
    a += eta_u_bc * delta_4 * p * q * ds  # may decrease convergente rates
    # a += eta_u_bc * delta_4 * dot(u, n) * dot(v, n) * ds
    # ** Mesh independent **
    # a += jump(u, n) * jump(v, n) * dS
    # a += dot(jump(p, n), jump(q, n)) * dS
    # a += p * q * ds

    A = assemble(a, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-12)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


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
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # BCs
    u_projected = sigma_e
    p_boundaries = p_exact
    bcs = DirichletBC(T, p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e-18)
    # beta = beta_0 / h
    beta = beta_0

    # Stabilization parameters
    delta_0 = Constant(-1)
    delta_1 = Constant(-0.5) * h * h
    delta_2 = Constant(0.5) * h * h
    delta_3 = Constant(0.5) * h * h

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + delta_0 * q * div(u)) * dx
    L = delta_0 * f * q * dx
    # Stabilizing terms
    a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L += delta_2 * f * div(v) * dx
    # Hybridization terms
    a += lambda_h("+") * dot(v, n)("+") * dS + mu_h("+") * dot(u, n)("+") * dS
    a += beta("+") * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
    # Weakly imposed BC
    a += (p_boundaries * dot(v, n) + mu_h * (dot(u, n) - dot(u_projected, n))) * ds
    a += beta * (lambda_h - p_boundaries) * mu_h * ds

    F = a - L
    a_form = lhs(F)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bcs)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bcs
    )
    return result


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
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Dirichlet BCs
    # bc_multiplier = DirichletBC(W.sub(2), p_exact, "on_boundary")
    bc_multiplier = DirichletBC(T, p_exact, "on_boundary")

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
    a_form = lhs(F)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bc_multiplier)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bc_multiplier
    )
    return result


def solve_poisson_primal_lsh(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Dirichlet BCs
    bc_multiplier = DirichletBC(T, p_exact, "on_boundary")
    bcs = DirichletBC(W.sub(1), p_exact, "on_boundary")
    # bc_multiplier = []

    # Hybridization parameter
    beta_0 = Constant(0 * degree)  # should not be zero when used with LS terms
    beta = beta_0 / h
    # beta = beta_0

    # Stabilizing parameter
    delta_base = h
    # delta_base = Constant(1)
    delta_0 = Constant(1)
    delta_1 = delta_base * Constant(1)
    # delta_2 = delta_base * Constant(1) / h
    # delta_2 = delta_1 * Constant(0)  # so far this is the best combination
    delta_2 = Constant(1e1 * degree * degree) / h
    
    # From convergence analysis
    delta_0 = Constant(0)
    delta_1 = Constant(1)
    # delta_2 = delta_base * Constant(1) / h
    # delta_2 = delta_1 * Constant(0)  # so far this is the best combination
    delta_2 = Constant(1e3) / h / h

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Symmetry parameter: s = 1 (unsymmetric) or s = -1 (symmetric)
    s = Constant(-1)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Classical term
    a = delta_0 * dot(grad(p), grad(q)) * dx - delta_0('+') * jump(u_hat, n) * q("+") * dS
    # a += delta_0 * dot(u_hat, n) * q * ds
    a += -delta_0 * dot(u, n) * q * ds + delta_0 * beta * p * q * ds  # expand u_hat product in ds
    L = delta_0 * f * q * dx
    L += delta_0 * beta * exact_solution * q * ds

    # Mass balance least-square
    a += delta_1 * div(u) * div(v) * dx
    # a += delta_1 * inner(curl(u), curl(v)) * dx
    L += delta_1 * f * div(v) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    # a += mu_h * lambda_h * ds

    # Least-Squares on constrains
    a += delta_2("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    # a += delta_2 * p * (q - mu_h) * ds  # needed if not included as strong BC
    a += delta_2 * p * q * ds  # needed if not included as strong BC
    L += delta_2 * exact_solution * q * ds  # needed if not included as strong BC
    a += delta_2 * lambda_h * mu_h * ds  # needed if not included as strong BC
    L += delta_2 * p_exact * mu_h * ds  # needed if not included as strong BC

    # Consistent symmetrization
    a += s * delta_0 * jump(v, n) * (p('+') - lambda_h("+")) * dS
    a += s * delta_0 * dot(v, n) * (p - exact_solution) * ds

    F = a - L
    a_form = lhs(F)

    Amat = assemble(a_form, bcs=bcs, mat_type="aij")

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[1, 1] - A[1, :1] * A[:1, :1].inv * A[:1, 1]
    Smat = assemble(S, bcs=bc_multiplier)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Amat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bc_multiplier,
        assembled_condensed_form=Smat
    )
    return result


def solve_poisson_pdglsh(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Dirichlet BCs
    bc_multiplier = DirichletBC(T, p_exact, "on_boundary")
    bcs = DirichletBC(W.sub(1), p_exact, "on_boundary")
    # bc_multiplier = []

    # Hybridization parameter
    beta_0 = Constant(1)  # should not be zero when used with LS terms
    beta = beta_0 / h
    # beta = beta_0

    # Stabilizing parameter
    delta_base = h
    # delta_base = Constant(1)
    delta_0 = Constant(1)
    delta_1 = delta_base * Constant(1)
    # delta_2 = delta_base * Constant(1) / h
    # delta_2 = delta_1 * Constant(0)  # so far this is the best combination
    delta_2 = Constant(1e1 * degree * degree) / h
    
    # From convergence analysis
    delta_0 = Constant(1)
    delta_1 = Constant(1) * h * h
    # delta_2 = delta_base * Constant(1) / h
    # delta_2 = delta_1 * Constant(0)  # so far this is the best combination
    delta_2 = Constant(1e6) / h / h

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Symmetry parameter: s = 1 (unsymmetric) or s = -1 (symmetric)
    s = Constant(-1)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Classical term
    a = delta_0 * dot(grad(p), grad(q)) * dx - delta_0('+') * jump(u_hat, n) * q("+") * dS
    # a += delta_0 * dot(u_hat, n) * q * ds
    a += -delta_0 * dot(u, n) * q * ds + delta_0 * beta * p * q * ds  # expand u_hat product in ds
    L = delta_0 * f * q * dx
    L += delta_0 * beta * exact_solution * q * ds

    # Mass balance least-square
    a += delta_1 * div(u) * div(v) * dx
    # a += delta_1 * inner(curl(u), curl(v)) * dx
    L += delta_1 * f * div(v) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    # a += mu_h * lambda_h * ds

    # Least-Squares on constrains
    a += delta_2("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    # a += delta_2 * p * (q - mu_h) * ds  # needed if not included as strong BC
    a += delta_2 * p * q * ds  # needed if not included as strong BC
    L += delta_2 * exact_solution * q * ds  # needed if not included as strong BC
    a += delta_2 * lambda_h * mu_h * ds  # needed if not included as strong BC
    L += delta_2 * p_exact * mu_h * ds  # needed if not included as strong BC

    # Consistent symmetrization
    a += s * delta_0 * jump(v, n) * (p('+') - lambda_h("+")) * dS
    a += s * delta_0 * dot(v, n) * (p - exact_solution) * ds

    F = a - L
    a_form = lhs(F)

    Amat = assemble(a_form, bcs=bcs, mat_type="aij")

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[1, 1] - A[1, :1] * A[:1, :1].inv * A[:1, 1]
    Smat = assemble(S, bcs=bc_multiplier)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Amat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bc_multiplier,
        assembled_condensed_form=Smat
    )
    return result


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
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # BCs
    bcs = DirichletBC(W.sub(2), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta_1 = Constant(1.0e0)
    beta = beta_0 / h

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

    # Flux Least-squares as in DG
    a += delta_0 * inner(u + grad(p), v + grad(q)) * dx

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS

    a += Constant(1) * delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    a += delta_4 * (p - p_exact) * q * ds
    a += delta_4 * (lambda_h - p_exact) * mu_h * ds

    a_form = lhs(a)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bcs)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bcs
    )
    return result


def hp_refinement_cond_number_calculation(
    solver,
    min_degree=1,
    max_degree=4,
    numel_xy=(4, 6, 8, 10, 12, 14),
    quadrilateral=True,
    name="",
    **kwargs
):
    results_dict = {
        "Element": list(),
        "Number of Elements": list(),
        "Degree": list(),
        "Symmetric": list(),
        "nnz": list(),
        "dofs": list(),
        "h": list(),
        "Condition Number": list(),
    }
    element_kind = "Quad" if quadrilateral else "Tri"
    pbar = tqdm(range(min_degree, max_degree))
    for degree in pbar:
        for n in numel_xy:
            pbar.set_description(f"Processing {name} - degree = {degree} - N = {n}")
            mesh = UnitSquareMesh(n, n, quadrilateral=quadrilateral)
            result = solver(mesh, degree=degree)

            current_cell_size = mesh.cell_sizes.dat.data_ro.min() if not quadrilateral else 1 / n
            results_dict["Element"].append(element_kind)
            results_dict["Number of Elements"].append(n * n)
            results_dict["Degree"].append(degree)
            results_dict["Symmetric"].append(result.is_operator_symmetric)
            results_dict["nnz"].append(result.nnz)
            results_dict["dofs"].append(result.number_of_dofs)
            results_dict["h"].append(current_cell_size)
            results_dict["Condition Number"].append(result.condition_number)

    base_name = f"./cond_number_results/results_deg_var_{name}"
    os.makedirs(f"{base_name}", exist_ok=True)
    df_cond_number = pd.DataFrame(data=results_dict)
    path_to_save_results = f"{base_name}/cond_numbers.csv"
    df_cond_number.to_csv(path_to_save_results)

    return df_cond_number


# Solver options
solvers_options = {
    "cg": solve_poisson_cg,
    "cgls": solve_poisson_cgls,
    "clsq": solve_poisson_ls,
    "dls": solve_poisson_dls,
    "lsh": solve_poisson_lsh,
    "pdls": solve_poisson_pdls,
    "pdlsip": solve_poisson_pdls_ip,
    "pdlsh": solve_poisson_primal_lsh,
    "pdglsh": solve_poisson_pdglsh,
    "sipg": solve_poisson_sipg,
}

degree = 1
last_degree = 4
elements_for_each_direction = [6, 8, 10, 12, 14]
for current_solver in solvers_options:

    # Setting the output file name
    name = f"{current_solver}"

    # Selecting the solver and its kwargs
    solver = solvers_options[current_solver]

    # Performing the convergence study
    hp_refinement_cond_number_calculation(
        solver,
        min_degree=degree,
        max_degree=degree + last_degree,
        quadrilateral=False,
        numel_xy=elements_for_each_direction,
        name=name
    )
