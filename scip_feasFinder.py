from scip_functions import solve_bip_scip, UniformRandomBrancher, RandomChildNodeSelector
import pandas as pd
import multiprocessing
import concurrent.futures
import os
from pyscipopt import Model,SCIP_PARAMSETTING
import datetime
import time

def scip_solver(file, vec=None, relax=False, use_custom_plugins=True, time_limit_hours=None):
    """
    Migrated BIP solver function using SCIP with custom branching,
    local child node selection, and an optional time limit in hours.
    """
    filename  = os.path.basename(file)         # "2club200v15p5scn.mps"
    name_only = os.path.splitext(filename)[0] 
    print(f"\n\n{'#'*10}\nProcessing file: {name_only}\n{'#'*10}")

    smodel = Model()
    smodel.hideOutput()

    # Read the model from a file (e.g., .mps, .lp, or .zpl)
    smodel.readProblem(file)
    num_binary = smodel.getNBinVars()
    num_integer = smodel.getNIntVars()
    num_continuous = smodel.getNContVars()
    num_implicit = smodel.getNImplVars()  # Implicit integer variables

    # 2. Fetch total variables and constraints
    total_vars = smodel.getNVars()
    total_conss = smodel.getNConss()

    N = num_binary
    M = total_conss
    sense = smodel.getObjectiveSense()
    print(f"\n\nproblem Name: {name_only}, Num_Vars: {N}, Num_Constrs: {M}")
    print(f"The problem is a {sense} problem.")



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
            memsavepriority=999999
        )

    # 5. Solve
    time1 = time.perf_counter()
    smodel.optimize()
    time2 = time.perf_counter()
    # 6. Status Mapping & Best Solution Extraction
    status_str = smodel.getStatus()
    NO_SOLUTION = {'infeasible', 'unbounded', 'inforunbd'}

    # Check if any valid solution was found before or during the time limit
    has_sol = smodel.getNSols() > 0

    if status_str in NO_SOLUTION:
        x_val = None
        obj_val = None
    else:
        # If status_str is 'timelimit' but has_sol is True,
        # this safely extracts the best incumbent solution found so far.
        x_val = [smodel.getVal(var) for var in var_list]
        obj_val = smodel.getObjVal()
    return name_only,N, M, obj_val, x_val, time2-time1
if __name__ == "__main__":
    results = {}
    HOURS_LIMIT = 0.5  # Set the time limit in hours for each SCIP run
    num_processes = multiprocessing.cpu_count()
    folder_path = r"C:\Users\KISHAN\Downloads\smoke_test"
    #mps_files = [f for f in os.listdir(folder_path) if f.endswith('.mps') or f.endswith('.mps.gz')]
    # Get and sort all files in the folder
    all_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])



    with concurrent.futures.ProcessPoolExecutor(max_workers= 4) as executor:
        future_to_file = {
            executor.submit(scip_solver, os.path.join(folder_path, file_name), None, False, True, HOURS_LIMIT): file_name 
            for file_name in all_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            # We look up what file name this future started with just to handle errors safely
            start_file_name = future_to_file[future] 
            
            try:
                # Unpack the returned tuple (filename and the bundle)
                name_only, N, M, obj_val, x_val, sol_time = future.result()
                results[name_only] = (N, M, obj_val, sol_time)


                
                print(f"Successfully mapped results for {start_file_name}")
                
            except Exception as e:
                print(f"Error occurred while processing {start_file_name}: {e}")

    df=pd.DataFrame(results)

    # 2. Transpose so Model names become Rows for now, metrics become Columns
    df = df.T
    '''columns_hierarchy = [
    ("Num_Vars", ""),
    ("Num_Constrs",""),
    ("SCIP_Obj", ""),
    ("SCIP_Time", "")
    ]

  # 3. Sort columns so the "Performance" categories group together completely
    df = df.sort_index(axis=1)'''
    df.columns = ['Num_Vars', 'Num_Constrs', 'SCIP_Obj', 'SCIP_Time']

    df.to_csv("scipSmoketest_res.csv", encoding="utf-8", index= True)

    print(df)
   