from enum import Enum, auto
import itertools
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from firedrake.petsc import PETSc
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # plt.style.use('fivethirtyeight')
    plt.style.use('seaborn-v0_8-talk')
    
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    LARGER_SIZE = 20
    HUGE_SIZE = 22

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
except:
    warning("Matplotlib not imported")


class StabilizationMethods(Enum):
    CGLS = auto()
    CGLS_DIV = auto()
    MGLS = auto()
    eMGLS = auto()
    VMS = auto()
    VMS_DIV = auto()
    eVMS = auto()


def plot_scalar_field(scalar_field_function, name="field_func", xlim=None, ylim=None):
    fig, axes = plt.subplots()
    collection = tripcolor(scalar_field_function, axes=axes, cmap='coolwarm')
    fig.colorbar(collection, orientation='horizontal')
    
    if xlim is not None:
        assert len(xlim) == 2
        axes.set_xlim(xlim)
        
    if ylim is not None:
        assert len(ylim) == 2
        axes.set_ylim(ylim)
        
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{name}.png", dpi=600, bbox_inches='tight')
    plt.close()


def get_xy_coordinate_points(function_space, mesh):
    x, y = SpatialCoordinate(mesh)
    
    xfunc = Function(function_space).interpolate(x)
    x_points = np.unique(np.array(xfunc.dat.data))
    
    yfunc = Function(function_space).interpolate(y)
    y_points = np.unique(np.array(yfunc.dat.data))
    
    return x_points, y_points


def retrieve_solution_on_line_fixed_x(solution, function_space, mesh, x_value):
    _, y_points = get_xy_coordinate_points(function_space, mesh)
    solution_on_a_line = [solution.at([x_value, y_point]) for y_point in y_points]
    solution_on_a_line = np.array(solution_on_a_line)
    return solution_on_a_line


def make_k_macro(mesh, k_reference_value=Constant(0.2)):
    _, y = SpatialCoordinate(mesh)
    k = k_reference_value
    k_macro = conditional(
        y <= 0.8,
        80 * k,
        conditional(
            y <= 1.6,
            30 * k,
            conditional(
                y <= 2.4,
                5 * k,
                conditional(
                    y <= 3.2,
                    50 * k,
                    10 * k
                )
            )
        )
    )
    return k_macro


def make_k_micro(mesh, k_reference_value=Constant(0.2)):
    _, y = SpatialCoordinate(mesh)
    k = k_reference_value
    k_micro = conditional(
        y <= 0.8,
        16 * k,
        conditional(
            y <= 1.6,
            6 * k,
            conditional(
                y <= 2.4,
                1 * k,
                conditional(
                    y <= 3.2,
                    10 * k,
                    2 * k
                )
            )
        )
    )
    return k_micro


def get_mixed_method_stabilization_parameters(stabilization_method=StabilizationMethods.CGLS):
    if stabilization_method is StabilizationMethods.CGLS:
        delta_0 = Constant(1.0)
        delta_1 = Constant(-0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.5)
    elif stabilization_method is StabilizationMethods.CGLS_DIV:
        delta_0 = Constant(1.0)
        delta_1 = Constant(-0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.0)
    elif stabilization_method is StabilizationMethods.MGLS:
        delta_0 = Constant(1.0)
        delta_1 = Constant(0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.0)
    elif stabilization_method is StabilizationMethods.eMGLS:
        delta_0 = Constant(1.0)
        delta_1 = Constant(0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.5)
    elif stabilization_method is StabilizationMethods.VMS:
        delta_0 = Constant(-1.0)
        delta_1 = Constant(0.5)
        delta_2 = Constant(0.0)
        delta_3 = Constant(0.0)
    elif stabilization_method is StabilizationMethods.VMS_DIV:
        delta_0 = Constant(-1.0)
        delta_1 = Constant(0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.0)
    elif stabilization_method is StabilizationMethods.eVMS:
        delta_0 = Constant(-1.0)
        delta_1 = Constant(0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.5)
        
    return delta_0, delta_1, delta_2, delta_3


def calculate_velocity_patch_exact_solution(mesh, mesh_space, x_value):
    k_macro = make_k_macro(mesh_space)
    k_micro = make_k_micro(mesh_space)
    mu0 = Constant(1.0)
    solution_macro = interpolate(k_macro / mu0, mesh_space)
    solution_micro = interpolate(k_micro / mu0, mesh_space)
    
    macro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        solution_macro,
        mesh_space,
        mesh,
        x_value
    )
    
    micro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        solution_micro,
        mesh_space,
        mesh,
        x_value
    )
    return macro_velocity_on_fixed_x, micro_velocity_on_fixed_x


def solve_cgls_velocity_patch(
    mesh, 
    mesh_space, 
    x_value,
    quadrilateral=True,
    p_degree=1, 
    v_degree=1,
    mesh_dependent_stabilization=True,
    stabilization_method=StabilizationMethods.CGLS,
):
    primal_family = "CG"
    flux_family = "CG"
    U = VectorFunctionSpace(mesh, flux_family, v_degree)
    V = FunctionSpace(mesh, primal_family, p_degree)
    W = U * V * U * V

    # Trial and test functions
    DPP_solution = Function(W)
    u1, p1, u2, p2 = TrialFunctions(W)
    v1, q1, v2, q2 = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    #################################################
    # *** Model parameters
    #################################################
    kSpace = FunctionSpace(mesh, "DG", 0)

    mu0 = Constant(1.0)
    k_macro = make_k_macro(mesh)
    k_micro = make_k_micro(mesh)
    k1 = interpolate(k_macro, kSpace)
    k2 = interpolate(k_micro, kSpace)


    def alpha1():
        return mu0 / k1


    def invalpha1():
        return 1.0 / alpha1()


    def alpha2():
        return mu0 / k2


    def invalpha2():
        return 1.0 / alpha2()


    #################################################
    #################################################
    #################################################

    #  Flux BCs
    un1_1 = -k1 / mu0
    un2_1 = -k2 / mu0
    un1_2 = k1 / mu0
    un2_2 = k2 / mu0

    # Source term
    rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
    f = Constant(0.0)

    # Stabilizing parameter
    delta_0, delta_1, delta_2, delta_3 = get_mixed_method_stabilization_parameters(stabilization_method)

    if mesh_dependent_stabilization:
        delta_1 = delta_1 * h * h
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h
        
    # Nitsche's penalty
    eta = Constant(1e8)

    # Mixed classical terms
    a = (dot(alpha1() * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
    a += (dot(alpha2() * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
    a += delta_0 * q1 * (invalpha1() / k1) * (p1 - p2) * dx
    a += delta_0 * q2 * (invalpha2() / k2) * (p2 - p1) * dx
    L = -delta_0 * dot(rhob1, v1) * dx
    L += -delta_0 * dot(rhob2, v2) * dx
    # Stabilizing terms
    ###
    a += (
        delta_1
        * inner(invalpha1() * (alpha1() * u1 + grad(p1)), delta_0 * alpha1() * v1 + grad(q1))
        * dx
    )
    a += (
        delta_1
        * inner(invalpha2() * (alpha2() * u2 + grad(p2)), delta_0 * alpha2() * v2 + grad(q2))
        * dx
    )
    ###
    a += delta_2 * alpha1() * div(u1) * div(v1) * dx
    a += delta_2 * alpha2() * div(u2) * div(v2) * dx
    a += -delta_2 * alpha1() * (invalpha1() / k1) * (p1 - p2) * div(v1) * dx
    a += -delta_2 * alpha2() * (invalpha2() / k2) * (p2 - p1) * div(v2) * dx
    ###
    a += delta_3 * inner(invalpha1() * curl(alpha1() * u1), curl(alpha1() * v1)) * dx
    a += delta_3 * inner(invalpha2() * curl(alpha2() * u2), curl(alpha2() * v2)) * dx
    # Weakly imposed BC by Nitsche's method
    a += dot(v1, n) * p1 * ds + dot(v2, n) * p2 * ds - q1 * dot(u1, n) * ds - q2 * dot(u2, n) * ds
    L += -q1 * un1_1 * ds(1) - q2 * un2_1 * ds(1) - q1 * un1_2 * ds(2) - q2 * un2_2 * ds(2)
    a += eta / h * inner(dot(v1, n), dot(u1, n)) * ds + eta / h * inner(dot(v2, n), dot(u2, n)) * ds
    L += (
        eta / h * dot(v1, n) * un1_1 * ds(1)
        + eta / h * dot(v2, n) * un2_1 * ds(1)
        + eta / h * dot(v1, n) * un1_2 * ds(2)
        + eta / h * dot(v2, n) * un2_2 * ds(2)
    )

    PETSc.Sys.Print("*******************************************\nSolving...\n")
    solver_parameters = {
        "ksp_monitor": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem_flow = LinearVariationalProblem(a, L, DPP_solution, bcs=[])
    solver_flow = LinearVariationalSolver(
        problem_flow, options_prefix="flow_cgls", solver_parameters=solver_parameters
    )
    solver_flow.solve()
    PETSc.Sys.Print("Solver finished.\n")

    plot_scalar_field(DPP_solution.sub(1), name='macro_p_cgls', xlim=[0, 5], ylim=[0, 4])
    
    macro_velocity_solution_x_component_function = DPP_solution.sub(0).sub(0)
    macro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        macro_velocity_solution_x_component_function,
        mesh_space,
        mesh,
        x_value
    )
    
    micro_velocity_solution_x_component_function = DPP_solution.sub(2).sub(0)
    micro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        micro_velocity_solution_x_component_function,
        mesh_space,
        mesh,
        x_value
    )
    
    return macro_velocity_on_fixed_x, micro_velocity_on_fixed_x


def solve_dvms_velocity_patch(mesh, mesh_space, x_value, quadrilateral=True, p_degree=1, v_degree=1):
    primal_family = "DQ" if quadrilateral else "DG"
    flux_family = "DQ" if quadrilateral else "DG"
    velSpace = VectorFunctionSpace(mesh, flux_family, v_degree)
    pSpace = FunctionSpace(mesh, primal_family, p_degree)
    wSpace = MixedFunctionSpace([velSpace, pSpace, velSpace, pSpace])

    kSpace = FunctionSpace(mesh, primal_family, 0)

    mu0 = Constant(1.0)
    k_macro = make_k_macro(mesh)
    k_micro = make_k_micro(mesh)
    k1 = interpolate(k_macro, kSpace)
    k2 = interpolate(k_micro, kSpace)


    def alpha1():
        return mu0 / k1


    def invalpha1():
        return 1.0 / alpha1()


    def alpha2():
        return mu0 / k2


    def invalpha2():
        return 1.0 / alpha2()


    un1_1 = -k1 / mu0
    un2_1 = -k2 / mu0
    un1_2 = k1 / mu0
    un2_2 = k2 / mu0

    (v1, p1, v2, p2) = TrialFunctions(wSpace)
    (w1, q1, w2, q2) = TestFunctions(wSpace)
    DPP_solution = Function(wSpace)

    rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
    f = Constant(0.0)

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2.0

    eta_p, eta_u = Constant(50.0), Constant(500.0)

    aDPP = (
        dot(w1, alpha1() * v1) * dx
        + dot(w2, alpha2() * v2) * dx
        - div(w1) * p1 * dx
        - div(w2) * p2 * dx
        + q1 * div(v1) * dx
        + q2 * div(v2) * dx
        + q1 * (invalpha1() / k1) * (p1 - p2) * dx
        - q2 * (invalpha2() / k2) * (p1 - p2) * dx
        + jump(w1, n) * avg(p1) * dS
        + jump(w2, n) * avg(p2) * dS
        - avg(q1) * jump(v1, n) * dS
        - avg(q2) * jump(v2, n) * dS
        + dot(w1, n) * p1 * ds
        + dot(w2, n) * p2 * ds
        - q1 * dot(v1, n) * ds
        - q2 * dot(v2, n) * ds
        - 0.5 * dot(alpha1() * w1 - grad(q1), invalpha1() * (alpha1() * v1 + grad(p1))) * dx
        - 0.5 * dot(alpha2() * w2 - grad(q2), invalpha2() * (alpha2() * v2 + grad(p2))) * dx
        + (eta_u * h_avg) * avg(alpha1()) * (jump(v1, n) * jump(w1, n)) * dS
        + (eta_u * h_avg) * avg(alpha2()) * (jump(v2, n) * jump(w2, n)) * dS
        + (eta_p / h_avg) * avg(1.0 / alpha1()) * dot(jump(q1, n), jump(p1, n)) * dS
        + (eta_p / h_avg) * avg(1.0 / alpha2()) * dot(jump(q2, n), jump(p2, n)) * dS
    )

    LDPP = (
        dot(w1, rhob1) * dx
        + dot(w2, rhob2) * dx
        - q1 * un1_1 * ds(1)
        - q2 * un2_1 * ds(1)
        - q1 * un1_2 * ds(2)
        - q2 * un2_2 * ds(2)
        - 0.5 * dot(alpha1() * w1 - grad(q1), invalpha1() * rhob1) * dx
        - 0.5 * dot(alpha2() * w2 - grad(q2), invalpha2() * rhob2) * dx
    )

    PETSc.Sys.Print("*******************************************\nSolving...\n")
    solver_parameters = {
        "ksp_monitor": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem_flow = LinearVariationalProblem(aDPP, LDPP, DPP_solution, bcs=[])
    solver_flow = LinearVariationalSolver(
        problem_flow, options_prefix="flow_dvms", solver_parameters=solver_parameters
    )
    solver_flow.solve()
    PETSc.Sys.Print("Solver finished.\n")
    
    plot_scalar_field(DPP_solution.sub(1), name='macro_p_dvms', xlim=[0, 5], ylim=[0, 4])
    
    macro_velocity_solution_x_component_function = DPP_solution.sub(0).sub(0)
    macro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        macro_velocity_solution_x_component_function,
        mesh_space,
        mesh,
        x_value
    )
    
    micro_velocity_solution_x_component_function = DPP_solution.sub(2).sub(0)
    micro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        micro_velocity_solution_x_component_function,
        mesh_space,
        mesh,
        x_value
    )
    
    return macro_velocity_on_fixed_x, micro_velocity_on_fixed_x
    

def solve_sdhm_velocity_patch(mesh, mesh_space, x_value, quadrilateral=True, p_degree=1, v_degree=1):
    primal_family = "DQ" if quadrilateral else "DG"
    flux_family = "DQ" if quadrilateral else "DG"
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, flux_family, v_degree + 1)
    V = FunctionSpace(mesh, primal_family, p_degree)
    T = FunctionSpace(mesh, trace_family, p_degree)
    W = U * V * T * U * V * T

    # Trial and test functions
    DPP_solution = Function(W)
    u1, p1, lambda1, u2, p2, lambda2 = split(DPP_solution)
    v1, q1, mu1, v2, q2, mu2 = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)

    #################################################
    # *** Model parameters
    #################################################
    kSpace = FunctionSpace(mesh, "DG", 0)

    mu0 = Constant(1.0)
    k_macro = make_k_macro(mesh)
    k_micro = make_k_micro(mesh)
    k1 = interpolate(k_macro, kSpace)
    k2 = interpolate(k_micro, kSpace)


    def alpha1():
        return mu0 / k1


    def invalpha1():
        return 1.0 / alpha1()


    def alpha2():
        return mu0 / k2


    def invalpha2():
        return 1.0 / alpha2()


    #################################################
    #################################################
    #################################################

    # Flux BCs
    un1_1 = -k1 / mu0
    un2_1 = -k2 / mu0
    un1_2 = k1 / mu0
    un2_2 = k2 / mu0

    # Source term
    rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
    f = Constant(0.0)

    # Stabilizing parameter
    beta_0 = Constant(1.0)
    beta = beta_0 / h
    beta_avg = beta_0 / h("+")
    delta_0 = Constant(1.0)
    delta_1 = Constant(-0.5)
    delta_2 = Constant(0.5)
    delta_3 = Constant(0.5)

    # Mixed classical terms
    a = (dot(alpha1() * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
    a += (dot(alpha2() * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
    a += delta_0 * q1 * (invalpha1() / k1) * (p1 - p2) * dx
    a += delta_0 * q2 * (invalpha2() / k2) * (p2 - p1) * dx
    L = -delta_0 * dot(rhob1, v1) * dx
    L += -delta_0 * dot(rhob2, v2) * dx
    # Stabilizing terms
    ###
    a += (
        delta_1
        * inner(invalpha1() * (alpha1() * u1 + grad(p1)), delta_0 * alpha1() * v1 + grad(q1))
        * dx
    )
    a += (
        delta_1
        * inner(invalpha2() * (alpha2() * u2 + grad(p2)), delta_0 * alpha2() * v2 + grad(q2))
        * dx
    )
    ###
    a += delta_2 * alpha1() * div(u1) * div(v1) * dx
    a += delta_2 * alpha2() * div(u2) * div(v2) * dx
    L += delta_2 * alpha1() * (invalpha1() / k1) * (p1 - p2) * div(v1) * dx
    L += delta_2 * alpha2() * (invalpha2() / k2) * (p2 - p1) * div(v2) * dx
    ###
    a += delta_3 * inner(invalpha1() * curl(alpha1() * u1), curl(alpha1() * v1)) * dx
    a += delta_3 * inner(invalpha2() * curl(alpha2() * u2), curl(alpha2() * v2)) * dx
    # Hybridization terms
    ###
    a += lambda1("+") * jump(v1, n) * dS + mu1("+") * jump(u1, n) * dS
    a += lambda2("+") * jump(v2, n) * dS + mu2("+") * jump(u2, n) * dS
    ###
    a += beta_avg * invalpha1()("+") * (lambda1("+") - p1("+")) * (mu1("+") - q1("+")) * dS
    a += beta_avg * invalpha2()("+") * (lambda2("+") - p2("+")) * (mu2("+") - q2("+")) * dS
    # Weakly imposed BC from hybridization
    a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n) - un1_1)) * ds(1)
    a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n) - un2_1)) * ds(1)
    a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n) - un1_2)) * ds(2)
    a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n) - un2_2)) * ds(2)
    a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n))) * (ds(3) + ds(4))
    a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n))) * (ds(3) + ds(4))

    F = a - L

    #  Solving SC below
    PETSc.Sys.Print("*******************************************\nSolving using static condensation.\n")
    solver_parameters = {
        "ksp_monitor": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    # solver_parameters = {
    #     "ksp_monitor": None,
    #     # "snes_monitor": None,
    #     "snes_type": "ksponly",
    #     "pmat_type": "matfree",
    #     "ksp_type": "lgmres",
    #     # "ksp_monitor_true_residual": None,
    #     "ksp_rtol": 1.0e-5,
    #     "ksp_atol": 1.0e-5,
    #     "pc_type": "fieldsplit",
    #     "pc_fieldsplit_0_fields": "0,1,2",
    #     "pc_fieldsplit_1_fields": "3,4,5",
    #     "fieldsplit_0": {
    #         "pmat_type": "matfree",
    #         "ksp_type": "preonly",
    #         "pc_type": "python",
    #         "pc_python_type": "firedrake.SCPC",
    #         "pc_sc_eliminate_fields": "0, 1",
    #         "condensed_field": {
    #             "ksp_type": "preonly",
    #             "pc_type": "lu",
    #             "pc_factor_mat_solver_type": "mumps",
    #         },
    #     },
    #     "fieldsplit_1": {
    #         "pmat_type": "matfree",
    #         "ksp_type": "preonly",
    #         "pc_type": "python",
    #         "pc_python_type": "firedrake.SCPC",
    #         "pc_sc_eliminate_fields": "0, 1",
    #         "condensed_field": {
    #             "ksp_type": "preonly",
    #             "pc_type": "lu",
    #             "pc_factor_mat_solver_type": "mumps",
    #         },
    #     },
    # }
    problem_flow = NonlinearVariationalProblem(F, DPP_solution, bcs=[])
    solver_flow = NonlinearVariationalSolver(
        problem_flow, options_prefix="flow_sdhm", solver_parameters=solver_parameters
    )
    solver_flow.solve()
    PETSc.Sys.Print("Solver finished.\n")
    
    plot_scalar_field(DPP_solution.sub(1), name='macro_p_sdhm', xlim=[0, 5], ylim=[0, 4])
    
    macro_velocity_solution_x_component_function = DPP_solution.sub(0).sub(0)
    macro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        macro_velocity_solution_x_component_function,
        mesh_space,
        mesh,
        x_value
    )
    
    micro_velocity_solution_x_component_function = DPP_solution.sub(3).sub(0)
    micro_velocity_on_fixed_x = retrieve_solution_on_line_fixed_x(
        micro_velocity_solution_x_component_function,
        mesh_space,
        mesh,
        x_value
    )
    
    return macro_velocity_on_fixed_x, micro_velocity_on_fixed_x    


nx, ny = 50, 40
Lx, Ly = 5.0, 4.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)
lagrange_family = "DQ" if quadrilateral else "DG"
mesh_space = FunctionSpace(mesh, lagrange_family, 1)

x_points, y_points = get_xy_coordinate_points(mesh_space, mesh)
x_mid_point = (x_points.min() + x_points.max()) / 2

solve_cgls_div_velocity_patch = lambda mesh, mesh_space, x_mid_point: solve_cgls_velocity_patch(
    mesh, mesh_space, x_mid_point, stabilization_method=StabilizationMethods.CGLS_DIV
)
solve_vms_velocity_patch = lambda mesh, mesh_space, x_mid_point: solve_cgls_velocity_patch(
    mesh, mesh_space, x_mid_point, stabilization_method=StabilizationMethods.VMS
)
solve_vms_div_velocity_patch = lambda mesh, mesh_space, x_mid_point: solve_cgls_velocity_patch(
    mesh, mesh_space, x_mid_point, stabilization_method=StabilizationMethods.VMS_DIV
)
solve_evms_velocity_patch = lambda mesh, mesh_space, x_mid_point: solve_cgls_velocity_patch(
    mesh, mesh_space, x_mid_point, stabilization_method=StabilizationMethods.eVMS
)
solve_mgls_velocity_patch = lambda mesh, mesh_space, x_mid_point: solve_cgls_velocity_patch(
    mesh, mesh_space, x_mid_point, stabilization_method=StabilizationMethods.MGLS
)
methods_to_solve = {
    # "CGLS": solve_cgls_velocity_patch,
    # "CGLS(Div)": solve_cgls_div_velocity_patch,
    # "MGLS": solve_mgls_velocity_patch,
    # "VMS": solve_vms_velocity_patch,
    # "VMS(Div)": solve_vms_div_velocity_patch,
    # "eVMS": solve_evms_velocity_patch,
    # "MDG": solve_dvms_velocity_patch,
    # "SDHM": solve_sdhm_velocity_patch,
    "Exact": calculate_velocity_patch_exact_solution,
}
solutions_macro_velocity = {}
solutions_micro_velocity = {}
for method, solver_driver in methods_to_solve.items():
    PETSc.Sys.Print(f"\n*** Solving for {method} ***\n")
    macro_velocity, micro_velocity = solver_driver(mesh, mesh_space, x_mid_point)
    solutions_macro_velocity[method] = macro_velocity
    solutions_micro_velocity[method] = micro_velocity

SAVE_RESULTS = False
DIR_RESULTS = Path("./output-dpp-all-methods-multilayer/")
if not os.path.exists(DIR_RESULTS):
    os.mkdir(DIR_RESULTS)
    
###############################################################
#################### PLOTTING #################################
###############################################################

PETSc.Sys.Print("*******************************************\nPlotting...\n")

# *** Macro velocity ***

PETSc.Sys.Print("*** Macro velocities ***\n")

plt.figure(figsize=(10, 9))
markers = itertools.cycle(('v', '+', '^', '.', 'o', '*', '1'))
plt.plot([], [], lw=0, label="Methods")
for method in methods_to_solve:
    macro_velocity = solutions_macro_velocity[method]
    if method == "MDG":
        plt.plot(y_points, macro_velocity, lw=4, c='k', label=method)
    elif method == "SDHM":
        plt.plot(y_points, macro_velocity, "--", lw=4, c='cyan', label="HM")
    elif method == "Exact":
        plt.plot(y_points, macro_velocity, "-.", lw=4, c='magenta', label=method)
    else:
        plt.plot(y_points, macro_velocity, lw=2.5, ms=10, marker=next(markers), label=method)

plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel("Macro velocity (component x)")
plt.xlim([0, 4])
# plt.ylim([0, 24])
if SAVE_RESULTS:
    plt.savefig(f"{DIR_RESULTS}/macro_velocities.png")

# *** Micro velocity ***

PETSc.Sys.Print("*** Micro velocities ***\n")

plt.figure(figsize=(10, 9))
markers = itertools.cycle(('v', '+', '^', '.', 'o', '*', '1'))
plt.plot([], [], lw=0, label="Methods")
for method in methods_to_solve:
    micro_velocity = solutions_micro_velocity[method]
    if method == "MDG":
        plt.plot(y_points, micro_velocity, lw=4, c='k', label=method)
    elif method == "SDHM":
        plt.plot(y_points, micro_velocity, "--", lw=4, c='cyan', label="HM")
    elif method == "Exact":
        plt.plot(y_points, micro_velocity, "-.", lw=4, c='magenta', label=method)
    else:
        plt.plot(y_points, micro_velocity, lw=2.5, ms=10, marker=next(markers), label=method)

plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel("Micro velocity (component x)")
plt.xlim([0, 4])
# plt.ylim([0, 7])
if SAVE_RESULTS:
    plt.savefig(f"{DIR_RESULTS}/micro_velocities.png")

Citations.print_at_exit()