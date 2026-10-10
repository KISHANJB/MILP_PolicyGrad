import uuid

from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_STATUS, SCIP_BASESTAT, SCIP_PARAMSETTING


class SinglePassStrongBrancher(Branchrule):
    def __init__(self, model, warm_start_x=None, vbasis=None, cbasis=None, tol=1e-5):
        super().__init__()
        self.model = model
        self.tol = tol
        
        # Warm start inputs
        self.warm_start_x = warm_start_x
        self.vbasis_in = vbasis
        self.cbasis_in = cbasis
        
        # Output attributes captured during execution
        self.x_val = None                  # Primal solution vector (for warm start)
        self.obj_val = None                # LP Objective value
        self.solve_status = None           # Solver status string
        self.chosen_var_idx = None         # Selected branching variable index
        self.next_branch_direction = None  # Branch direction ('DOWN' or 'UP')
        self.out_vbasis = None             # Variable basis statuses
        self.out_cbasis = None             # Constraint basis statuses

    def branchexeclp(self, allowaddcons):
        var_list = self.model.getVars()
        cons_list = self.model.getConss()

        # 1. Apply Primal Warm Start if provided
        if self.warm_start_x is not None and len(self.warm_start_x) == len(var_list):
            sol = self.model.createSol()
            for var, val in zip(var_list, self.warm_start_x):
                self.model.setSolVal(sol, var, val)
            self.model.addSol(sol)

        # 2. Extract Primal Solution Vector (x_val) and Objective Value
        self.x_val = [self.model.getVal(var) for var in var_list]
        self.obj_val = self.model.getObjVal()

        # 3. Extract Solver Status
        status_map = {
            SCIP_STATUS.OPTIMAL: 'OPTIMAL',
            SCIP_STATUS.INFEASIBLE: 'INFEASIBLE',
            SCIP_STATUS.UNBOUNDED: 'UNBOUNDED',
            SCIP_STATUS.INFORUNBD: 'INF_OR_UNBD',
            SCIP_STATUS.NODELIMIT: 'NODE_LIMIT',
            SCIP_STATUS.TIMELIMIT: 'TIME_LIMIT',
            SCIP_STATUS.MEMLIMIT: 'MEM_LIMIT',
            SCIP_STATUS.USERINTERRUPT: 'INTERRUPTED',
        }
        raw_status = self.model.getStatus()
        self.solve_status = status_map.get(raw_status, None)

        # 4. Save Basis States for Variables and Constraints
        try:
            self.out_vbasis = [var.getBasisStatus() for var in var_list]
            self.out_cbasis = [cons.getBasisStatus() for cons in cons_list]
        except Exception:
            self.out_vbasis = None
            self.out_cbasis = None

        # 5. Filter fractional discrete variables
        fractional_vars = [
            var for var, val in zip(var_list, self.x_val)
            if var.vtype() in ["BINARY", "INTEGER"] and abs(val - round(val)) > self.tol
        ]

        if not fractional_vars:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # 6. Run Native Strong Branching
        result, fallback = self.model.executeBranchrule("allfullstrong")

        if result == SCIP_RESULT.BRANCHED:
            branch_vars, _, _ = self.model.getBranchingPoint()
            if branch_vars:
                var = branch_vars[0]
                self.chosen_var_idx = var.getIndex()

                # 7. Extract child node direction preference
                children = self.model.getChildren()
                if children:
                    next_child = (
                        self.model.getBestChild()
                        if hasattr(self.model, "getBestChild")
                        else children[0]
                    )

                    child_ub = self.model.getNodeUb(next_child, var)
                    parent_ub = var.getUbLocal()

                    if child_ub < parent_ub - 1e-6:
                        self.next_branch_direction = 'L'  # DOWN
                    else:
                        self.next_branch_direction = 'R'  # UP

        return {"result": result}


def scip_sb(
    model: Model,
    warm_start: list = None,
    vbasis: list = None,
    cbasis: list = None,
    tol: float = 1e-5
):
    """
    Solves the LP relaxation, executes strong branching, and returns 
    all execution metrics (solution vector, obj_val, status, decisions, bases).

    Returns:
        (solve_status, obj_val, x_val, chosen_var_idx, next_branch_direction, out_vbasis, out_cbasis)
    """


    model.setParam("presolving/maxrounds", 0)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setParam("limits/nodes", 1)  # Stop after root node
    model.setParam("branching/allfullstrong/priority", 999999)

    extractor = SinglePassStrongBrancher(
        model, 
        warm_start_x=warm_start, 
        vbasis=vbasis, 
        cbasis=cbasis, 
        tol=tol
    )
    unique_brancher_name = f"SinglePassStrongBrancher_{uuid.uuid4().hex[:8]}"
    model.includeBranchrule(
            extractor,
            name=unique_brancher_name,
            desc="Extracts LP metrics and evaluates strong branching",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1.0,
        )

    # Execute single optimization pass
    model.optimize()

    # Fallback status check returning None if callback was never reached or unmapped
    status = extractor.solve_status
    if status is None:
        status_map = {
            SCIP_STATUS.OPTIMAL: 'OPTIMAL',
            SCIP_STATUS.INFEASIBLE: 'INFEASIBLE',
            SCIP_STATUS.UNBOUNDED: 'UNBOUNDED',
            SCIP_STATUS.INFORUNBD: 'INF_OR_UNBD',
            SCIP_STATUS.NODELIMIT: 'NODE_LIMIT',
            SCIP_STATUS.TIMELIMIT: 'TIME_LIMIT',
            SCIP_STATUS.MEMLIMIT: 'MEM_LIMIT',
            SCIP_STATUS.USERINTERRUPT: 'INTERRUPTED',
        }
        status = status_map.get(model.getStatus(), None)

    obj_val = extractor.obj_val if extractor.obj_val is not None else (
        model.getObjVal() if model.getNSols() > 0 else None
    )
    x_val =extractor.x_val
    br_id = extractor.chosen_var_idx
    br_dir = extractor.next_branch_direction
    vbasis = extractor.out_vbasis
    cbasis = extractor.out_cbasis
    model.freeTransform()
    
    return (
        status,                        # Returns None if solve was unmapped or skipped
        obj_val,                       # Returns None if no solution exists
        x_val,
        br_id,
        br_dir,
        vbasis,
        cbasis
    )