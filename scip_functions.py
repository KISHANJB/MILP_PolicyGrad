import random
import pyscipopt

from pyscipopt import Model, Branchrule, Nodesel, SCIP_RESULT,SCIP_PARAMSETTING

from pyscipopt import Model, Branchrule, SCIP_RESULT


class BranchingAndNextChildExtractor(Branchrule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.chosen_var_name = None
        self.next_branch_direction = None  # Will store 'DOWN' or 'UP'

    def branchexeclp(self, allowaddcons):
        # 1. Execute SCIP's native Full Strong Branching
        result, fallback = self.model.executeBranchrule("allfullstrong")

        if result == SCIP_RESULT.BRANCHED:
            # 2. Get the variable chosen by the strong brancher
            branch_vars, branch_bounds, branch_types = self.model.getBranchingPoint()
            if branch_vars:
                var = branch_vars[0]
                self.chosen_var_name = var.name

                # 3. Inspect the newly created child nodes before SCIP solves them
                children = self.model.getChildren()

                if children:
                    # SCIP's node selector picks the next child to process
                    next_child = (
                        self.model.getBestChild()
                        if hasattr(self.model, "getBestChild")
                        else children[0]
                    )

                    # 4. Compare Upper Bounds to determine DOWN vs UP
                    child_ub = self.model.getNodeUb(next_child, var)
                    parent_ub = var.getUbLocal()

                    if child_ub < parent_ub - 1e-6:
                        self.next_branch_direction = "DOWN"
                    else:
                        self.next_branch_direction = "UP"

        return {"result": result}


def scip_solver(smodel, state_vec = None, relax=False, time_limit_hours=None):
    """
    Solves the root node relaxation and returns:
      1. status_str     : Termination status (e.g., 'OPTIMAL', 'INFEASIBLE', 'UNBOUNDED')
      2. obj_val        : Objective value (or None if no valid solution exists)
      3. x_val          : Solution vector as a list (or None if no valid solution exists)
      4. chosen_var     : Name of the variable selected by Strong Branching (or None)
      5. next_direction : Next child branch direction selected ('DOWN' or 'UP', or None)
    """

    # 1. Set Time Limit (Convert hours to seconds for SCIP)
    if time_limit_hours is not None:
            time_limit_seconds = time_limit_hours * 3600.0
            smodel.setParam("limits/time", time_limit_seconds)
    
    # 2. Relaxation (In-place modification)
    if relax:
            for var in smodel.getVars():
                if var.vtype() in ["BINARY", "INTEGER"]:
                    smodel.chgVarType(var, "CONTINUOUS")
    
    # 3. Apply bounds if provided
    var_list = smodel.getVars()
    N = len(var_list)
    if state_vec is not None:
        if len(state_vec) != 2 * N:
            raise ValueError(f"state_vec must have length 2*N={2*N}, got {len(state_vec)}")

        for i, var in enumerate(var_list):
            smodel.chgVarLb(var, state_vec[i])
            smodel.chgVarUb(var, state_vec[i + N])
                
    # 4. Disable presolving and heuristics to analyze the pure LP relaxation
    smodel.setParam("presolving/maxrounds", 0)
    smodel.setParam("heuristics/emphasis", 0)

    # 5. Force SCIP's native full strong branching
    smodel.setParam("branching/allfullstrong/priority", 999999)

    # 6. Attach the extractor plugin
    extractor = BranchingAndNextChildExtractor(smodel)
    smodel.includeBranchrule(
        extractor,
        name="BranchingAndNextChildExtractor",
        desc="Extracts strong branching candidate and next child direction",
        priority=999999,
        maxdepth=-1,
        maxbounddist=1.0,
    )

    # 7. Solve the model
    smodel.optimize()

    # 8. Extract termination status, objective value, and solution vector
    status_str = smodel.getStatus().upper()
    var_list = smodel.getVars()
    has_sol = smodel.getNSols() > 0

    NO_SOLUTION_STATUSES = {"INFEASIBLE", "UNBOUNDED", "INFORUNBD"}

    if status_str in NO_SOLUTION_STATUSES or not has_sol:
        obj_val = None
        x_val = None
    else:
        obj_val = smodel.getObjVal()
        x_val = [smodel.getVal(var) for var in var_list]

    return status_str, obj_val, x_val, extractor.chosen_var_name, extractor.next_branch_direction,
    

class UniformRandomBrancher(Branchrule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def branchexeclp(self, allowaddcons):
        cands, _, _, _, npriocands = self.model.getLPBranchCands()
        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Pick a fractional candidate uniformly at random
        chosen_var = random.choice(cands[:npriocands])
        self.model.branchVar(chosen_var)
        return {"result": SCIP_RESULT.BRANCHED}


class RandomChildNodeSelector(Nodesel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def nodeselevent(self):
        child = self.model.getPrioChild()
        sibling = self.model.getPrioSibling()

        candidates = []
        if child is not None:
            candidates.append(child)
        if sibling is not None:
            candidates.append(sibling)

        # Select randomly between local children/siblings if they exist
        if candidates:
            selected_node = random.choice(candidates)
            return {"selnode": selected_node}

        # Fallback to the global tree if the current branch was pruned/solved
        return {"selnode": self.model.getBestNode()}

    def nodeselcomp(self, node1, node2):
        return 0




def constraint_slacks(model, x_val):
    """
    Computes and prints the activity and slack for every constraint.
    """
    # 1. Map your solution vector to SCIP variables
    sol = model.createSol(None)
    for var, val in zip(model.getVars(), x_val):
        model.setSolVal(sol, var, val)

    print(f"{'Constraint Name':<20} | {'LHS':<10} | {'Activity':<10} | {'RHS':<10} | {'Slack':<10}")
    print("-" * 65)
    prod_slacks = 0
  
    # 2. Extract values and display
    for cons in model.getConss():
        activity = model.getActivity(cons, sol)
        lhs = model.getLhs(cons)
        rhs = model.getRhs(cons)
        
        # Calculate the closest distance to a binding bound
        # (Filtering out SCIP's internal infinity values ~1e20)
        slacks = []
        curr_slack = 0
        if lhs is not None and lhs > -1e10:
            slacks.append(activity - lhs)
            curr_slack = activity - lhs
            result = 1 if curr_slack > 0 else 0

        if rhs is not None and rhs < 1e10:
            slacks.append(rhs - activity)
            curr_slack = rhs - activity
            result = 1 if curr_slack > 0 else 0
        
        prod_slacks *= result
        if prod_slacks == 0:
            break
            return prod_slacks
        

    return prod_slacks

def get_violated_constraints(model: Model, x_val: list):
    """
    Checks an external solution vector against a SCIP model and returns
    a list of names of all constraints that are violated.
    
    Parameters:
    - model: The PySCIPOpt Model object
    - x_val: A list of floats representing the values of model.getVars()
    """
    var_list = model.getVars()
    if len(x_val) != len(var_list):
        raise ValueError(f"Solution vector length ({len(x_val)}) must match variable count ({len(var_list)})")

    # 1. Create an empty standalone Solution object linked to this model
    sol = model.createSol(None)

    # 2. Populate the solution object with your values
    for var, val in zip(var_list, x_val):
        model.setSolVal(sol, var, val)

    # 3. Ask SCIP to check the solution. 
    # Setting printreason=True forces SCIP to print detailed violation 
    # messages directly to stdout (e.g., "violation in constraint <cons_name>: ...")
    is_feasible = model.checkSol(sol, printreason=True, completely=True)
    
    violated_names = []
    num_violated = 0
    
    if is_feasible:
        print("The solution is perfectly feasible! No constraints violated.")
        return violated_names, num_violated

    # 4. Programmatically identify which ones are violated
    # We iterate over all constraints and check if their activity/slack violates bounds
    for cons in model.getConss():
        # SCIP provides a native method to check a constraint's row violation in a solution
        violation = model.getConsViolation(cons, sol)
        
        # If the violation is greater than zero (or SCIP's basic tolerance), it's violated
        if violation > 1e-6:
            violated_names.append(cons.name)
            num_violated += 1

    return violated_names, num_violated

def solve_bip_scip(smodel: Model, vec=None, relax=False, use_custom_plugins=True, time_limit_hours=None):
    """
    Migrated BIP solver function using SCIP with custom branching,
    local child node selection, and an optional time limit in hours.
    """
    # 1. Set Time Limit (Convert hours to seconds for SCIP)
    if time_limit_hours is not None:
        time_limit_seconds = time_limit_hours * 3600.0
        smodel.setParam("limits/time", time_limit_seconds)

    # 2. Relaxation (In-place modification)
    if relax:
        for var in smodel.getVars():
            if var.vtype() in ["BINARY", "INTEGER"]:
                smodel.chgVarType(var, "CONTINUOUS")

    # 3. Apply bounds if provided
    var_list = smodel.getVars()
    N = len(var_list)
    if vec is not None:
        if len(vec) != 2 * N:
            raise ValueError(f"vec must have length 2*N={2*N}, got {len(vec)}")

        for i, var in enumerate(var_list):
            smodel.chgVarLb(var, vec[i])
            smodel.chgVarUb(var, vec[i + N])
            
    smodel.setHeuristics(SCIP_PARAMSETTING.OFF)

    # 4. Include Custom Brancher and Node Selector
    if use_custom_plugins and not relax:
        my_brancher = UniformRandomBrancher(smodel)
        my_selector = RandomChildNodeSelector(smodel)

        smodel.includeBranchrule(
            my_brancher,
            name="UniformRandom",
            desc="Branches on a fractional variable completely at random",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1.0
        )

        smodel.includeNodesel(
            my_selector,
            name="RandomChildOnly",
            desc="Selects randomly between local children/siblings before looking globally",
            stdpriority=999999,
            mempriority=999999
        )

    # 5. Solve
    smodel.optimize()

    # 6. Status Mapping & Best Solution Extraction
    status_str = smodel.getStatus()
    NO_SOLUTION = {'infeasible', 'unbounded', 'inforunbd'}

    # Check if any valid solution was found before or during the time limit
    has_sol = smodel.getNSols() > 0

    if status_str in NO_SOLUTION or (not has_sol and status_str != 'optimal'):
        x_val = None
        obj_val = None
        return status_str.upper(), obj_val, x_val
    else:
        # If status_str is 'timelimit' but has_sol is True,
        # this safely extracts the best incumbent solution found so far.
        x_val = [smodel.getVal(var) for var in var_list]
        obj_val = smodel.getObjVal()
        return status_str.upper(), obj_val, x_val