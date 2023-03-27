from enum import Enum, auto
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from firedrake.petsc import PETSc
import numpy as np

try:
    import matplotlib.pyplot as plt
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
    GLS = auto()
    GLS_DIV = auto()
    MGLS = auto()   
    eMGLS = auto()
    VMS = auto()
    VMS_DIV = auto()
    eVMS = auto()


def get_mixed_method_stabilization_parameters(stabilization_method=StabilizationMethods.GLS):
    if stabilization_method is StabilizationMethods.GLS:
        delta_0 = Constant(1.0)
        delta_1 = Constant(-0.5)
        delta_2 = Constant(0.5)
        delta_3 = Constant(0.5)
    elif stabilization_method is StabilizationMethods.GLS_DIV:
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


def plot_scalar_field(
    scalar_field_function, 
    filename="field_func", 
    output_dir=None, 
    xlim=None, 
    ylim=None,
    title='',
    label='',
    clim=None
):
    fig, axes = plt.subplots()
    if clim is not None:
        assert len(ylim) == 2
        collection = tripcolor(scalar_field_function, axes=axes, cmap='plasma', vmin=clim[0], vmax=clim[1])
    else:
        collection = tripcolor(scalar_field_function, axes=axes, cmap='plasma')
        
    fig.colorbar(collection, orientation='horizontal', label=label)
    axes.set_aspect("equal")
    
    if xlim is not None:
        assert len(xlim) == 2
        axes.set_xlim(xlim)
        
    if ylim is not None:
        assert len(ylim) == 2
        axes.set_ylim(ylim)
        
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    
    if output_dir is None:
        plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=600)
    else:
        plt.savefig(output_dir / f"{filename}.png", bbox_inches='tight', dpi=600)
        
    plt.close()


def get_xy_coordinate_points(function_space, mesh):
    x, y = SpatialCoordinate(mesh)
    
    xfunc = Function(function_space).interpolate(x)
    x_points = np.unique(np.array(xfunc.dat.data))
    
    yfunc = Function(function_space).interpolate(y)
    y_points = np.unique(np.array(yfunc.dat.data))
    
    return x_points, y_points


def calculate_domain_lenght_x(mesh, mesh_space):
    x_points, _ = get_xy_coordinate_points(mesh_space, mesh)
    len_x = np.abs(x_points.max() - x_points.min())
    return len_x


def make_c0_function(mesh, mesh_space):
    x, _ = SpatialCoordinate(mesh)
    Lx = calculate_domain_lenght_x(mesh, mesh_space)
    rng = np.random.default_rng(12345)
    c_0 = conditional(x < 0.010 * Lx, abs(0.1 * exp(-x * x) * rng.random()), 0.0)
    return c_0


def make_k_field(
    mesh, k_layer_1=Constant(1.1), k_layer_2=Constant(0.9), boundary_layer_position=0.2
):
    _, y = SpatialCoordinate(mesh)
    k_macro = conditional(
        y < boundary_layer_position,
        k_layer_1,
        k_layer_2
    )
    return k_macro


def get_permeability_values():
    k1_0 = Constant(1.1)
    k1_1 = Constant(0.9)
    k2_0 = 0.01 * k1_0
    k2_1 = 0.01 * k1_1
    return k1_0, k1_1, k2_0, k2_1


def get_rock_fluid_properties():
    mu0, Rc, D = Constant(1e-3), Constant(3.0), Constant(2e-6)
    return mu0, Rc, D


def get_time_evolution_parameters():
    t_total = 1.5e-3
    dt = 5e-5
    return dt, t_total


def compute_limit_values_over_all_solutions(solutions):
    solutions_values = np.array([])
    for _, solution in solutions.items():
        solution_array = np.array(solution.dat.data)
        solutions_values = np.concatenate([solutions_values, solution_array])
    min_value = solutions_values.min()
    max_value = solutions_values.max()
    return min_value, max_value


def solve_dpp_tracer_with_dgls(
    mesh, 
    mesh_space, 
    quadrilateral=True,
    mesh_dependent_stabilization=True,
    stabilization_method=StabilizationMethods.GLS,
):
    degree = 1
    primal_family = "DQ" if quadrilateral else "DG"
    flux_family = "DQ" if quadrilateral else "DG"
    U = VectorFunctionSpace(mesh, flux_family, degree)
    V = FunctionSpace(mesh, primal_family, degree)
    W = U * V * U * V

    # Trial and test functions
    DPP_solution = Function(W)
    u1, p1, u2, p2 = TrialFunctions(W)
    v1, q1, v2, q2 = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)

    #################################################
    # *** Model parameters
    #################################################
    
    uSpace = FunctionSpace(mesh, "CG", 1)
    kSpace = FunctionSpace(mesh, primal_family, 0)

    # Rock/fluid/flow properties
    mu0, Rc, D = get_rock_fluid_properties()

    # Permeability for the layers and scales. (1) Macro and (2) Micro.
    k1_0, k1_1, k2_0, k2_1 = get_permeability_values()

    _, y_points = get_xy_coordinate_points(mesh_space, mesh)
    Ly = np.abs(y_points.max() - y_points.min())
    k_macro = make_k_field(
        mesh, k_layer_1=k1_0, k_layer_2=k1_1, boundary_layer_position=Ly / 2.0
    )
    k1 = interpolate(k_macro, kSpace)

    k_micro = make_k_field(
        mesh, k_layer_1=k2_0, k_layer_2=k2_1, boundary_layer_position=Ly / 2.0
    )
    k2 = interpolate(k_micro, kSpace)


    def alpha1(c):
        return mu0 * exp(Rc * (1.0 - c)) / k1


    def invalpha1(c):
        return 1.0 / alpha1(c)


    def alpha2(c):
        return mu0 * exp(Rc * (1.0 - c)) / k2


    def invalpha2(c):
        return 1.0 / alpha2(c)


    # Initial condition for tracer
    c1 = TrialFunction(uSpace)
    c_0 = make_c0_function(mesh, mesh_space)
    u = TestFunction(uSpace)
    conc = Function(uSpace)
    conc_k = interpolate(c_0, uSpace)
    
    # BCs
    p_L = Constant(10.0)
    p_R = Constant(1.0)
    bcDPP = []
    c_inj = Constant(1.0)
    bcleft_c = DirichletBC(uSpace, c_inj, 1)
    bcAD = [bcleft_c]
    eta_u = Constant(50)
    eta_p = 10 * eta_u

    # Time parameters
    dt, t_total = get_time_evolution_parameters()

    # Source term
    rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
    f = Constant(0.0)

    # Stabilizing parameter
    h_avg = (h("+") + h("-")) / 2.0
    delta_0, delta_1, delta_2, delta_3 = get_mixed_method_stabilization_parameters(stabilization_method)
    if mesh_dependent_stabilization:
        # delta_1 = delta_1 * h * h
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h

    # Mixed classical terms
    a = (dot(alpha1(conc_k) * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
    a += (dot(alpha2(conc_k) * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
    a += delta_0 * q1 * (invalpha1(conc_k) / k1) * (p2 - p1) * dx
    a += delta_0 * q2 * (invalpha2(conc_k) / k2) * (p1 - p2) * dx
    L = -delta_0 * dot(rhob1, v1) * dx
    L += -delta_0 * dot(rhob2, v2) * dx
    # DG terms
    a += (
        jump(v1, n) * avg(p1) * dS
        + jump(v2, n) * avg(p2) * dS
        - avg(q1) * jump(u1, n) * dS
        - avg(q2) * jump(u2, n) * dS
    )
    # Edge stabilizing terms
    a += (
        (eta_u * h_avg) * avg(alpha1(conc_k)) * (jump(u1, n) * jump(v1, n)) * dS
        + (eta_u * h_avg) * avg(alpha2(conc_k)) * (jump(u2, n) * jump(v2, n)) * dS
        + (eta_p / h_avg) * avg(1.0 / alpha1(conc_k)) * dot(jump(q1, n), jump(p1, n)) * dS
        + (eta_p / h_avg) * avg(1.0 / alpha2(conc_k)) * dot(jump(q2, n), jump(p2, n)) * dS
    )
    # Stabilizing terms
    ###
    a += (
        delta_1
        * inner(
            invalpha1(conc_k) * (alpha1(conc_k) * u1 + grad(p1)),
            delta_0 * alpha1(conc_k) * v1 + grad(q1),
        )
        * dx
    )
    a += (
        delta_1
        * inner(
            invalpha2(conc_k) * (alpha2(conc_k) * u2 + grad(p2)),
            delta_0 * alpha2(conc_k) * v2 + grad(q2),
        )
        * dx
    )
    ###
    a += delta_2 * alpha1(conc_k) * div(u1) * div(v1) * dx
    a += delta_2 * alpha2(conc_k) * div(u2) * div(v2) * dx
    a += -delta_2 * alpha1(conc_k) * (invalpha1(conc_k) / k1) * (p2 - p1) * div(v1) * dx
    a += -delta_2 * alpha2(conc_k) * (invalpha2(conc_k) / k2) * (p1 - p2) * div(v2) * dx
    ###
    a += delta_3 * inner(invalpha1(conc_k) * curl(alpha1(conc_k) * u1), curl(alpha1(conc_k) * v1)) * dx
    a += delta_3 * inner(invalpha2(conc_k) * curl(alpha2(conc_k) * u2), curl(alpha2(conc_k) * v2)) * dx
    # Weakly imposed BC
    a += (
        dot(v1, n) * p1 * ds(3)
        + dot(v2, n) * p2 * ds(3)
        - q1 * dot(u1, n) * ds(3)
        - q2 * dot(u2, n) * ds(3)
        + dot(v1, n) * p1 * ds(4)
        + dot(v2, n) * p2 * ds(4)
        - q1 * dot(u1, n) * ds(4)
        - q2 * dot(u2, n) * ds(4)
    )
    L += (
        -dot(v1, n) * p_L * ds(1)
        - dot(v2, n) * p_L * ds(1)
        - dot(v1, n) * p_R * ds(2)
        - dot(v2, n) * p_R * ds(2)
    )
    a += eta_u / h * inner(dot(v1, n), dot(u1, n)) * (ds(3) + ds(4)) + eta_u / h * inner(
        dot(v2, n), dot(u2, n)
    ) * (ds(3) + ds(4))

    # *** Transport problem
    vnorm = sqrt(
        dot((DPP_solution.sub(0) + DPP_solution.sub(2)), (DPP_solution.sub(0) + DPP_solution.sub(2)))
    )

    taw = h / (2.0 * vnorm) * dot((DPP_solution.sub(0) + DPP_solution.sub(2)), grad(u))

    a_r = (
        taw
        * (c1 + dt * (dot((DPP_solution.sub(0) + DPP_solution.sub(2)), grad(c1)) - div(D * grad(c1))))
        * dx
    )

    L_r = taw * (conc_k + dt * f) * dx

    aAD = (
        a_r
        + u * c1 * dx
        + dt
        * (
            u * dot((DPP_solution.sub(0) + DPP_solution.sub(2)), grad(c1)) * dx
            + dot(grad(u), D * grad(c1)) * dx
        )
    )
    LAD = L_r + u * conc_k * dx + dt * u * f * dx

    solver_parameters = {
        "ksp_monitor": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem_flow = LinearVariationalProblem(
        a, L, DPP_solution, bcs=bcDPP
    )
    solver_flow = LinearVariationalSolver(
        problem_flow, options_prefix="flow", solver_parameters=solver_parameters
    )

    problem_tracer = LinearVariationalProblem(aAD, LAD, conc, bcs=bcAD)
    solver_tracer = LinearVariationalSolver(problem_tracer, options_prefix='tracer', solver_parameters=solver_parameters)
    
    # Storing concentration solution over time
    concentration_solutions = {}
    concentration_solutions[0.0] = conc_k.copy(deepcopy=True)
    t = dt
    t_count = 0
    while t <= t_total:
        print('============================')
        print('\ttime =', t)
        print('============================')
        t_count += 1

        solver_flow.solve()
        solver_tracer.solve()
        conc_k.assign(conc)
        concentration_solutions[t] = conc_k.copy(deepcopy=True)

        t += dt

    print("total time = ", t)
    return concentration_solutions


def solve_dpp_tracer_with_sdhm(
    mesh, 
    mesh_space, 
    quadrilateral=True,
    mesh_dependent_stabilization=True,
    stabilization_method=StabilizationMethods.GLS,
):
    degree = 1
    primal_family = "DQ" if quadrilateral else "DG"
    flux_family = "DQ" if quadrilateral else "DG"
    trace_family = "HDiv Trace"
    U1 = VectorFunctionSpace(mesh, flux_family, degree)
    U2 = VectorFunctionSpace(mesh, flux_family, degree)
    V = FunctionSpace(mesh, primal_family, degree)
    T = FunctionSpace(mesh, trace_family, degree)
    W = U1 * V * T * U2 * V * T

    # Trial and test functions
    DPP_solution = Function(W)
    u1, p1, lambda1, u2, p2, lambda2 = split(DPP_solution)
    v1, q1, mu1, v2, q2, mu2 = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    #################################################
    # *** Model parameters
    #################################################
    
    uSpace = FunctionSpace(mesh, "CG", 1)
    kSpace = FunctionSpace(mesh, primal_family, 0)

    # Rock/fluid/flow properties
    mu0, Rc, D = get_rock_fluid_properties()

    # Permeability for the layers and scales. (1) Macro and (2) Micro.
    k1_0, k1_1, k2_0, k2_1 = get_permeability_values()

    _, y_points = get_xy_coordinate_points(mesh_space, mesh)
    Ly = np.abs(y_points.max() - y_points.min())
    k_macro = make_k_field(
        mesh, k_layer_1=k1_0, k_layer_2=k1_1, boundary_layer_position=Ly / 2.0
    )
    k1 = interpolate(k_macro, kSpace)

    k_micro = make_k_field(
        mesh, k_layer_1=k2_0, k_layer_2=k2_1, boundary_layer_position=Ly / 2.0
    )
    k2 = interpolate(k_micro, kSpace)


    def alpha1(c):
        return mu0 * exp(Rc * (1.0 - c)) / k1


    def alpha1_avg(c):
        return mu0 * exp(Rc * (1.0 - c)) / k1("+")


    def invalpha1(c):
        return 1.0 / alpha1(c)


    def invalpha1_avg(c):
        return 1.0 / alpha1_avg(c)


    def alpha2(c):
        return mu0 * exp(Rc * (1.0 - c)) / k2


    def alpha2_avg(c):
        return mu0 * exp(Rc * (1.0 - c)) / k2("+")


    def invalpha2(c):
        return 1.0 / alpha2(c)


    def invalpha2_avg(c):
        return 1.0 / alpha2_avg(c)


    # Initial condition for tracer
    c1 = TrialFunction(uSpace)
    c_0 = make_c0_function(mesh, mesh_space)
    u = TestFunction(uSpace)
    conc = Function(uSpace)
    conc_k = interpolate(c_0, uSpace)
    
    # BCs
    v_topbottom = Constant(0.0)
    p_L = Constant(10.0)
    p_R = Constant(1.0)
    bcDPP = []
    c_inj = Constant(1.0)
    bcleft_c = DirichletBC(uSpace, c_inj, 1)
    bcAD = [bcleft_c]

    # Time parameters
    dt, t_total = get_time_evolution_parameters()

    # Source term
    rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
    f = Constant(0.0)

    # Stabilizing parameter
    beta_0 = Constant(1.0e-15)
    beta = beta_0 / h
    beta_avg = beta_0 / h("+")
    delta_0, delta_1, delta_2, delta_3 = get_mixed_method_stabilization_parameters(stabilization_method)
    if mesh_dependent_stabilization:
        # delta_1 = delta_1 * h * h
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h

    # Mixed classical terms
    a = (dot(alpha1(conc_k) * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
    a += (dot(alpha2(conc_k) * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
    a += delta_0 * q1 * (invalpha1(conc_k) / k1) * (p2 - p1) * dx
    a += delta_0 * q2 * (invalpha2(conc_k) / k2) * (p1 - p2) * dx
    L = -delta_0 * dot(rhob1, v1) * dx
    L += -delta_0 * dot(rhob2, v2) * dx
    
    # Stabilizing terms
    ###
    a += (
        delta_1
        * inner(
            invalpha1(conc_k) * (alpha1(conc_k) * u1 + grad(p1)),
            delta_0 * alpha1(conc_k) * v1 + grad(q1),
        )
        * dx
    )
    a += (
        delta_1
        * inner(
            invalpha2(conc_k) * (alpha2(conc_k) * u2 + grad(p2)),
            delta_0 * alpha2(conc_k) * v2 + grad(q2),
        )
        * dx
    )
    ###
    a += delta_2 * alpha1(conc_k) * div(u1) * div(v1) * dx
    a += delta_2 * alpha2(conc_k) * div(u2) * div(v2) * dx
    L += delta_2 * alpha1(conc_k) * (invalpha1(conc_k) / k1) * (p2 - p1) * div(v1) * dx
    L += delta_2 * alpha2(conc_k) * (invalpha2(conc_k) / k2) * (p1 - p2) * div(v2) * dx
    ###
    a += delta_3 * inner(invalpha1(conc_k) * curl(alpha1(conc_k) * u1), curl(alpha1(conc_k) * v1)) * dx
    a += delta_3 * inner(invalpha2(conc_k) * curl(alpha2(conc_k) * u2), curl(alpha2(conc_k) * v2)) * dx
    
    # Hybridization terms
    ###
    a += lambda1("+") * jump(v1, n) * dS + mu1("+") * jump(u1, n) * dS
    a += lambda2("+") * jump(v2, n) * dS + mu2("+") * jump(u2, n) * dS
    ###
    a += beta_avg * invalpha1_avg(conc_k("+")) * (lambda1("+") - p1("+")) * (mu1("+") - q1("+")) * dS
    a += beta_avg * invalpha2_avg(conc_k("+")) * (lambda2("+") - p2("+")) * (mu2("+") - q2("+")) * dS

    # Weakly imposed BC from hybridization
    a += (p_L * dot(v1, n) + mu1 * dot(u1, n)) * ds(1)
    a += (p_L * dot(v2, n) + mu2 * dot(u2, n)) * ds(1)
    a += (p_R * dot(v1, n) + mu1 * dot(u1, n)) * ds(2)
    a += (p_R * dot(v2, n) + mu2 * dot(u2, n)) * ds(2)
    a += (lambda1 * dot(v1, n) + mu1 * dot(u1, n)) * (ds(3) + ds(4))
    a += (lambda2 * dot(v2, n) + mu2 * dot(u2, n)) * (ds(3) + ds(4))
    ###
    a += beta * invalpha1(conc_k) * lambda1 * mu1 * (ds(3) + ds(4))
    a += beta * invalpha2(conc_k) * lambda2 * mu2 * (ds(3) + ds(4))
    a += beta * invalpha1(conc_k) * (lambda1 - p_L) * mu1 * ds(1)
    a += beta * invalpha2(conc_k) * (lambda2 - p_L) * mu2 * ds(1)
    a += beta * invalpha1(conc_k) * (lambda1 - p_R) * mu1 * ds(2)
    a += beta * invalpha2(conc_k) * (lambda2 - p_R) * mu2 * ds(2)

    F = a - L

    # *** Transport problem
    total_velocity = DPP_solution.sub(0) + DPP_solution.sub(3)
    vnorm = sqrt(
        dot(total_velocity, total_velocity)
    )

    taw = h / (2.0 * vnorm) * dot(total_velocity, grad(u))

    a_r = (
        taw
        * (c1 + dt * (dot(total_velocity, grad(c1)) - div(D * grad(c1))))
        * dx
    )

    L_r = taw * (conc_k + dt * f) * dx

    aAD = (
        a_r
        + u * c1 * dx
        + dt
        * (
            u * dot(total_velocity, grad(c1)) * dx
            + dot(grad(u), D * grad(c1)) * dx
        )
    )
    LAD = L_r + u * conc_k * dx + dt * u * f * dx

    PETSc.Sys.Print("*******************************************\nSolving using static condensation.\n")
    solver_parameters = {
        "snes_type": "ksponly",
        "pmat_type": "matfree",
        "ksp_type": "lgmres",
        "ksp_rtol": 1.0e-12,
        "ksp_atol": 1.0e-12,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_0_fields": "0,1,2",
        "pc_fieldsplit_1_fields": "3,4,5",
        "fieldsplit_0": {
            "pmat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
            "condensed_field": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "fieldsplit_1": {
            "pmat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
            "condensed_field": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
    }
    problem_flow = NonlinearVariationalProblem(F, DPP_solution, bcs=bcDPP)
    solver_flow = NonlinearVariationalSolver(problem_flow, options_prefix="flow", solver_parameters=solver_parameters)

    solver_parameters = {
        "ksp_monitor": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem_tracer = LinearVariationalProblem(aAD, LAD, conc, bcs=bcAD)
    solver_tracer = LinearVariationalSolver(problem_tracer, options_prefix='tracer', solver_parameters=solver_parameters)
    
    # Storing concentration solution over time
    concentration_solutions = {}
    concentration_solutions[0.0] = conc_k.copy(deepcopy=True)
    t = dt
    t_count = 0
    while t <= t_total:
        print('============================')
        print('\ttime =', t)
        print('============================')
        t_count += 1

        solver_flow.solve()
        solver_tracer.solve()
        conc_k.assign(conc)
        concentration_solutions[t] = conc_k.copy(deepcopy=True)

        t += dt

    print("total time = ", t)
    return concentration_solutions


nx, ny = 100, 40
Lx, Ly = 1.0, 0.4
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)
lagrange_family = "DQ" if quadrilateral else "DG"
mesh_space = FunctionSpace(mesh, lagrange_family, 1)

solve_dpp_tracer_with_dvms_div = lambda mesh, mesh_space, quadrilateral: solve_dpp_tracer_with_dgls(
    mesh, mesh_space, quadrilateral, stabilization_method=StabilizationMethods.VMS_DIV
)
solve_dpp_tracer_with_edvms = lambda mesh, mesh_space, quadrilateral: solve_dpp_tracer_with_dgls(
    mesh, mesh_space, quadrilateral, stabilization_method=StabilizationMethods.eVMS
)
solve_dpp_tracer_with_dgls_div = lambda mesh, mesh_space, quadrilateral: solve_dpp_tracer_with_dgls(
    mesh, mesh_space, quadrilateral, stabilization_method=StabilizationMethods.GLS_DIV
)
solve_dpp_tracer_with_hvms_div = lambda mesh, mesh_space, quadrilateral: solve_dpp_tracer_with_sdhm(
    mesh, mesh_space, quadrilateral, stabilization_method=StabilizationMethods.VMS_DIV
)
solve_dpp_tracer_with_ehvms = lambda mesh, mesh_space, quadrilateral: solve_dpp_tracer_with_sdhm(
    mesh, mesh_space, quadrilateral, stabilization_method=StabilizationMethods.eVMS
)
solve_dpp_tracer_with_sdhm_div = lambda mesh, mesh_space, quadrilateral: solve_dpp_tracer_with_sdhm(
    mesh, mesh_space, quadrilateral, stabilization_method=StabilizationMethods.GLS_DIV
)

methods_to_solve = {
    "DVMS(Div)": solve_dpp_tracer_with_dvms_div,
    "eDVMS": solve_dpp_tracer_with_edvms,
    "DGLS": solve_dpp_tracer_with_dgls,
    "DGLS(Div)": solve_dpp_tracer_with_dgls_div,
    "HVMS(Div)": solve_dpp_tracer_with_hvms_div,
    "eHVMS": solve_dpp_tracer_with_ehvms,
    "SDHM(Div)": solve_dpp_tracer_with_sdhm_div,
    "SDHM": solve_dpp_tracer_with_sdhm,
}
solutions_by_method = {}
for method, solver_driver in methods_to_solve.items():
    PETSc.Sys.Print(f"\n*** Solving for {method} ***\n")
    solutions_by_method[method] = solver_driver(mesh, mesh_space, quadrilateral)
    
###############################################################
#################### PLOTTING #################################
###############################################################

PETSc.Sys.Print("*******************************************\nPlotting...\n")

for method_name, solutions in solutions_by_method.items():
    DIR_RESULTS = Path(f"./output-dpp-tracer-all-methods/{method_name}")
    if not os.path.exists(DIR_RESULTS):
        os.mkdir(DIR_RESULTS)
        
    PETSc.Sys.Print(f"\n*** Plotting {method_name} ***\n")
    t_count = 0
    colorbar_min_value, colorbar_max_value = compute_limit_values_over_all_solutions(solutions)
    for time_value, solution in solutions.items():
        PETSc.Sys.Print(f"* Time: {time_value}")
        plot_scalar_field(
            solution, 
            filename=f"{method_name}_conc_field_00{t_count}", 
            output_dir=DIR_RESULTS, 
            xlim=[0.0, 1.0], 
            ylim=[0.0, 0.4],
            title=f't = {time_value:.3e}',
            label='Concentration',
            clim=(colorbar_min_value, colorbar_max_value)
        )
        t_count += 1
