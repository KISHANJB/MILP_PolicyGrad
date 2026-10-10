import sys
import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB
#Import brute-force functions to calculate values
sigma = float(sys.argv[6]) if len(sys.argv) > 5 else 0.01

def integrality_loss(x, p):
    """Calculates f(x) = sum( (min(xi, 1 - xi))^p ) using NumPy."""
    x = np.asarray(x)

    # 1. Compute min{xi, 1 - xi} element-wise
    min_terms = np.minimum(x, 1 - x)

    # 2. Raise to the power of p element-wise
    powered_terms = np.power(min_terms, p)

    # 3. Sum all the terms together
    return np.sum(powered_terms)



def gaussian_mixture_score(x, sigma):
    """
    Normalized Gaussian mixture score.

    Parameters
    ----------
    x : array_like
        n-dimensional vector.

    sigma : float, optional
        Standard deviation of the Gaussians.
        Smaller values produce sharper peaks at 0 and 1.

    Returns
    -------
    float
        Average normalized score in [0, 1].
    """
    #print(f"\n sigma = {sigma}")
    x = np.asarray(x, dtype=np.float64)

    # Gaussian centred at 0
    g0 = np.exp(-(x**2) / (2.0 * sigma**2))

    # Gaussian centred at 1
    g1 = np.exp(-((x - 1.0)**2) / (2.0 * sigma**2))

    # Maximum possible value of one coordinate
    normalizer = 1.0 + np.exp(-1.0 / (2.0 * sigma**2))

    return np.mean((g0 + g1) / normalizer)


def extract_sparse_gurobipy_matrices(model):
    """
    Extracts the entire .mps file into SciPy CSR sparse matrices ONCE at startup.
    """
    
    all_vars = model.getVars()
    all_constrs = model.getConstrs()
    
    num_vars = len(all_vars)
    num_constrs = len(all_constrs)
    
    # 1. Coordinate collection lists for efficient sparse building
    row_indices = []
    col_indices = []
    coefficients = []
    
    b_full = np.zeros(num_constrs)
    senses = np.zeros(num_constrs, dtype=object)
    
    # 2. Extract structural data via Gurobi's fast variable index mapping
    for c_idx, constr in enumerate(all_constrs):
        lhs_expr = model.getRow(constr)
        for i in range(lhs_expr.size()):
            v_idx = lhs_expr.getVar(i).index
            
            row_indices.append(c_idx)
            col_indices.append(v_idx)
            coefficients.append(lhs_expr.getCoeff(i))
        
        b_full[c_idx] = constr.RHS
        senses[c_idx] = constr.Sense

    # 3. Create the initial sparse matrix using the collected coordinates
    A_sparse = sp.coo_matrix((coefficients, (row_indices, col_indices)), 
                              shape=(num_constrs, num_vars)).tocsr()

    # 4. Compute DFP Scale Factors (Max value per row across coefficients AND RHS)
    # To get the max absolute coefficient per row in a CSR matrix safely and quickly:
    #max_coeff_per_row = np.zeros(num_constrs)
    row_scales_vector = np.zeros(num_constrs)
    for i in range(num_constrs):
        row_data = A_sparse.getrow(i).data
        row_data = np.append(row_data, b_full[i])  # Include RHS in max calculation
        #if row_data.size > 0:
        #    max_coeff_per_row[i] = np.max(np.abs(row_data))

        # 4. Apply the normalization denominator
        l1_norm = np.linalg.norm(row_data, ord=1)
        l2_norm = np.linalg.norm(row_data, ord=2)
        linf_norm = np.linalg.norm(row_data, ord=np.inf)
        lmax_norm = max(l1_norm, l2_norm, linf_norm)
        row_scales_vector[i] = l2_norm 

    row_scales_vector = np.array(row_scales_vector, dtype=float)
    row_scales_vector[row_scales_vector < 1e-9] = 1.0  # Protect against division by zero
    #inv_multipliers = 1.0 / row_scales_vector
    
    # Create a sparse diagonal matrix out of the inverse multipliers
    #D_scaling_matrix = sp.diags(inv_multipliers)
    
    # Matrix-Matrix multiplication scales rows simultaneously
    #A_scaled = D_scaling_matrix @ A_sparse
    
    # Standard element-wise array multiplication for the RHS vector
    #b_scaled = b_full * inv_multipliers
    # 6. Create boolean masks for fast vectorized filtering during training
    le_mask = (senses == '<')
    ge_mask = (senses == '>')
    eq_mask = (senses == '=')
    
    return A_sparse, b_full, le_mask, ge_mask, eq_mask





def infeasibility_cause(model):
    """
    Computes the IIS of an infeasible Gurobi model.

    Returns
    -------
    iis_constraints : list[str]
        Names of constraints in the IIS.

    iis_lb : list[tuple]
        (variable name, lower bound)

    iis_ub : list[tuple]
        (variable name, upper bound)
    """

    if model.Status != GRB.INFEASIBLE:
        print(f"Model status is {model.Status}. IIS can only be computed for an infeasible model.")
        return [], [], []

    print("\nComputing IIS...")
    model.computeIIS()

    iis_constraints = []
    iis_lb = []
    iis_ub = []

    # Structural constraints
    for c in model.getConstrs():
        if c.IISConstr:
            iis_constraints.append(c.ConstrName)

    # Variable lower/upper bounds
    for v in model.getVars():

        if v.IISLB:
            iis_lb.append((v.VarName, v.LB))

        if v.IISUB:
            iis_ub.append((v.VarName, v.UB))

    '''print("\n================ IIS REPORT ================\n")
    print("\n=== CONSTRAINTS IN THE CONFLICT SET (IIS) ===")
    for constr in model.getConstrs():
        if constr.IISConstr:
            print(f"Constraint {constr.ConstrName}: {model.getRow(constr)} {constr.Sense} {constr.RHS}")

    print("\n=== VARIABLE BOUNDS IN THE CONFLICT SET (IIS) ===")
    for var in model.getVars():
        if var.IISLB:
            print(f"Lower bound forced in IIS: {var.VarName} >= {var.LB}")
        if var.IISUB:
            print(f"Upper bound forced in IIS: {var.VarName} <= {var.UB}")'''

    '''print(f"Structural Constraints ({len(iis_constraints)}):")
    for name in iis_constraints:
        print(f"  {name}")
    print("\n================ Bounds_Violations ================\n")
    print(f"\nVariable Lower Bounds ({len(iis_lb)}):")
    for name, lb in iis_lb:
        print(f"  {name} >= {lb}")

    print(f"\nVariable Upper Bounds ({len(iis_ub)}):")
    for name, ub in iis_ub:
        print(f"  {name} <= {ub}")'''

    print("\n============================================")

    return iis_constraints, iis_lb, iis_ub



def load_model(prob):
    gmodel = gp.read(prob)
    gmodel.Params.OutputFlag = 0          # ✅ reads .mps natively, no conversion needed
    return gmodel



def print_bounds(model):
    print("=== ALL VARIABLE BOUNDS IN RELAXED MODEL ===")
    for v in model.getVars():
        print(f"{v.VarName}: [{v.LB}, {v.UB}]" if v.LB == v.UB else f"{v.VarName}: [{v.LB}, {v.UB}]")

# ─────────────────────────────────────────
# (Solving the BIP by adding the bounds of the variables from vec so that we get the solution corresponding to each of the states)
# ─────────────────────────────────────────
def solve_bip_gurobi(gmodel, vec, relax):
    # Relaxation
    if relax == True:
        gmodel = gmodel.relax()
    # Apply bounds if provided
    var_list = gmodel.getVars()
    N = len(var_list)
    if vec is not None:
        if len(vec) != 2 * N:
            raise ValueError(f"vec must have length 2*N={2*N}, got {len(vec)}")
        for i, var in enumerate(var_list):
            var.LB = vec[i]          # lower bound
            var.UB = vec[i + N]      # upper bound
        gmodel.update()               # ✅ must call update() after changing bounds



    # Solve
    gmodel.optimize()

    # Status

    status_map = {
        gp.GRB.LOADED:          'LOADED',           # 1  - Model loaded, not yet solved
        gp.GRB.OPTIMAL:         'OPTIMAL',           # 2  - Optimal solution found
        gp.GRB.INFEASIBLE:      'INFEASIBLE',        # 3  - Model is infeasible
        gp.GRB.INF_OR_UNBD:     'INF_OR_UNBD',       # 4  - Infeasible or unbounded
        gp.GRB.UNBOUNDED:       'UNBOUNDED',         # 5  - Model is unbounded
        gp.GRB.CUTOFF:          'CUTOFF',            # 6  - Objective worse than cutoff
        gp.GRB.ITERATION_LIMIT: 'ITERATION_LIMIT',   # 7  - Iteration limit reached
        gp.GRB.NODE_LIMIT:      'NODE_LIMIT',        # 8  - Node limit reached (MIP)
        gp.GRB.TIME_LIMIT:      'TIME_LIMIT',        # 9  - Time limit reached
        gp.GRB.SOLUTION_LIMIT:  'SOLUTION_LIMIT',    # 10 - Solution limit reached
        gp.GRB.INTERRUPTED:     'INTERRUPTED',       # 11 - Solve interrupted by user
        gp.GRB.NUMERIC:         'NUMERIC',           # 12 - Numerical issues
        gp.GRB.SUBOPTIMAL:      'SUBOPTIMAL',        # 13 - Suboptimal solution available
        gp.GRB.INPROGRESS:      'INPROGRESS',        # 14 - Solve currently in progress
        gp.GRB.USER_OBJ_LIMIT:  'USER_OBJ_LIMIT',    # 15 - User objective limit reached
        gp.GRB.WORK_LIMIT:      'WORK_LIMIT',        # 16 - Work limit reached
        gp.GRB.MEM_LIMIT:       'MEM_LIMIT',         # 17 - Memory limit reached
    }

    # Safe lookup with fallback
    tc = status_map.get(gmodel.Status, f'UNKNOWN({gmodel.Status})')
    #print(f"Solver status: {tc}")

    SOLUTION_EXISTS = {'OPTIMAL', 'SUBOPTIMAL'}

    LIMIT_REACHED = {
        'TIME_LIMIT', 'NODE_LIMIT', 'ITERATION_LIMIT',
        'SOLUTION_LIMIT', 'WORK_LIMIT', 'MEM_LIMIT', 'USER_OBJ_LIMIT'
    }

    NO_SOLUTION = {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}

    '''total_normalized_loss = 0.0

    for constr in gmodel.getConstrs():
       
        # 2. Grab the native slack value calculated by Gurobi
        native_slack = constr.Slack
        extracted_vector = extract_constraint_vectors(gmodel, constr)

        # 4. Apply the normalization denominator
        l1_norm = np.linalg.norm(extracted_vector, ord=1)
        l2_norm = np.linalg.norm(extracted_vector, ord=2)
        linf_norm = np.linalg.norm(extracted_vector, ord=np.inf)
        lmax_norm = max(l1_norm, l2_norm, linf_norm)
        normalized_violation = native_slack / l1_norm if l1_norm > 0 else 0.0
        total_normalized_loss += normalized_violation
        num_constrs = len(gmodel.getConstrs())
        
    avg_normalized_loss = total_normalized_loss / num_constrs if num_constrs > 0 else 0.0'''

    
    
    #print(f"Aggregated Scaled Loss: {avg_normalized_loss:.6f}")

    if tc in NO_SOLUTION:
        x_val = None
        obj_val = None
        return tc, obj_val, x_val
    else:
        x_val = [var.X for var in gmodel.getVars()]
        obj_val = gmodel.ObjVal
        return tc, obj_val, x_val


def solve_bip_gurobiws(
    gmodel,
    vec=None,
    relax=False,
    warm_start=None,
    vbasis=None,
    cbasis=None
):
    # ---------------------------------------------------------
    # 1. Relaxation
    # ---------------------------------------------------------
    if relax:
        gmodel = gmodel.relax()

    # ---------------------------------------------------------
    # 2. Apply bounds if provided
    # ---------------------------------------------------------
    var_list = gmodel.getVars()
    N = len(var_list)

    if vec is not None:
        if len(vec) != N:
            raise ValueError(
                f"vec must have length N={N}, got {len(vec)}"
            )

        for i, var in enumerate(var_list):
            if vec[i] in {0,1}:
                var.LB = vec[i]
                var.UB = vec[i]

        gmodel.update()

    # ---------------------------------------------------------
    # 3. Integer solution warm start
    # ---------------------------------------------------------
    if warm_start is not None:

        if len(warm_start) != N:
            raise ValueError(
                f"warm_start must have length N={N}, "
                f"got {len(warm_start)}"
            )

        for var, value in zip(gmodel.getVars(), warm_start):
            var.Start = value

    # ---------------------------------------------------------
    # 4. Basis warm start
    #
    # Only applicable when solving an LP / LP relaxation.
    # ---------------------------------------------------------
    if vbasis is not None and cbasis is not None:

        if len(vbasis) != len(gmodel.getVars()):
            raise ValueError(
                f"vbasis must have length {len(gmodel.getVars())}, "
                f"got {len(vbasis)}"
            )

        if len(cbasis) != len(gmodel.getConstrs()):
            raise ValueError(
                f"cbasis must have length {len(gmodel.getConstrs())}, "
                f"got {len(cbasis)}"
            )

        # Tell Gurobi to use the supplied LP warm start
        gmodel.Params.LPWarmStart = 2

        for var, value in zip(gmodel.getVars(), vbasis):
            var.VBasis = value

        for constr, value in zip(gmodel.getConstrs(), cbasis):
            constr.CBasis = value

    # ---------------------------------------------------------
    # 5. Solve
    # ---------------------------------------------------------
    gmodel.optimize()

    # ---------------------------------------------------------
    # 6. Status
    # ---------------------------------------------------------
    status_map = {
        gp.GRB.LOADED: 'LOADED',
        gp.GRB.OPTIMAL: 'OPTIMAL',
        gp.GRB.INFEASIBLE: 'INFEASIBLE',
        gp.GRB.INF_OR_UNBD: 'INF_OR_UNBD',
        gp.GRB.UNBOUNDED: 'UNBOUNDED',
        gp.GRB.CUTOFF: 'CUTOFF',
        gp.GRB.ITERATION_LIMIT: 'ITERATION_LIMIT',
        gp.GRB.NODE_LIMIT: 'NODE_LIMIT',
        gp.GRB.TIME_LIMIT: 'TIME_LIMIT',
        gp.GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
        gp.GRB.INTERRUPTED: 'INTERRUPTED',
        gp.GRB.NUMERIC: 'NUMERIC',
        gp.GRB.SUBOPTIMAL: 'SUBOPTIMAL',
        gp.GRB.INPROGRESS: 'INPROGRESS',
        gp.GRB.USER_OBJ_LIMIT: 'USER_OBJ_LIMIT',
        gp.GRB.WORK_LIMIT: 'WORK_LIMIT',
        gp.GRB.MEM_LIMIT: 'MEM_LIMIT',
    }

    tc = status_map.get(
        gmodel.Status,
        f'UNKNOWN({gmodel.Status})'
    )

    NO_SOLUTION = {
        'INFEASIBLE',
        'INF_OR_UNBD',
        'UNBOUNDED'
    }

    # ---------------------------------------------------------
    # 7. No solution
    # ---------------------------------------------------------
    if tc in NO_SOLUTION:
        return tc, None, None, None, None

    # ---------------------------------------------------------
    # 8. Extract solution
    # ---------------------------------------------------------
    x_val = [
        var.X for var in gmodel.getVars()
    ]

    obj_val = gmodel.ObjVal

    # ---------------------------------------------------------
    # 9. Extract optimal/current basis
    #
    # This is meaningful for an LP / LP relaxation.
    # ---------------------------------------------------------
    

    new_vbasis = [
        var.VBasis for var in gmodel.getVars()
    ]

    new_cbasis = [
        constr.CBasis for constr in gmodel.getConstrs()
    ]

    # ---------------------------------------------------------
    # 10. Return everything
    # ---------------------------------------------------------
    return tc, obj_val, x_val, new_vbasis, new_cbasis
    