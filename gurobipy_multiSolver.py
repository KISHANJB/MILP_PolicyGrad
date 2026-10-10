import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import time
import datetime
import openpyxl
# --- Define Your Max Time Limit in HOURS here ---
HOURS_LIMIT = 0.5  # Example: 2 and a half hours
TIMEOUT_SECONDS = HOURS_LIMIT * 3600.0  # Automatically converts to seconds for Gurobi

def solve(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder path '{folder_path}' does not exist.")
        return

    mps_files = [f for f in os.listdir(folder_path) if f.endswith('.mps') or f.endswith('.mps.gz')]
    
    if not mps_files:
        print("No .mps files found in the specified directory.")
        return


    #with gp.Env(empty=True) as env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()  # Internet handshake happens here exactly once

    results = {}

    for file_name in mps_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"\nProcessing File: {file_name} (Max Time Alloc: {HOURS_LIMIT} Hours)")
        results[file_name] = {}

        # -------------------------------------------------------------
        # MODE 1: Feasible Mode
        # -------------------------------------------------------------
        model_feas = gp.read(file_path, env)
        N = model_feas.NumIntVars
        M = model_feas.NumConstrs
        results[file_name]['Num_Vars'] = N
        results[file_name]['Num_Constrs'] = M
        model_feas.Params.SolutionLimit = 1  
        model_feas.Params.TimeLimit = TIMEOUT_SECONDS  # Applied here
        time1 = time.perf_counter()
        model_feas.optimize()
        time2 = time.perf_counter()
                
        if model_feas.SolCount > 0:
            results[file_name]['Feasible_Obj'] = model_feas.ObjVal
            #results[file_name]['Feasible_Status'] = "Feasible Found Early" if model_feas.Status != GRB.TIME_LIMIT else f"Feasible (Hit {HOURS_LIMIT}h Limit)"
        else:
            status_str = f"Limit Expired ({HOURS_LIMIT}h) - No Sol" if model_feas.Status == GRB.TIME_LIMIT else "Infeasible/Failed"
            results[file_name]['Feasible_Obj'] = None
            #results[file_name]['Feasible_Status'] = status_str
        results[file_name]['Feasible_Time'] = str(datetime.timedelta(seconds=time2-time1))
        model_feas.dispose()  # Free memory after feasible mode
        # -------------------------------------------------------------
        # MODE 2: Optimal Mode
        # -------------------------------------------------------------
        model_opt = gp.read(file_path, env)
        model_opt.Params.MIPGap = 0.0  
        model_opt.Params.TimeLimit = TIMEOUT_SECONDS  # Applied here
        time3 = time.perf_counter()
        model_opt.optimize()
        time4 = time.perf_counter()

        if model_opt.Status == GRB.OPTIMAL:
            results[file_name]['Optimal_Obj'] = model_opt.ObjVal
            #results[file_name]['Optimal_Status'] = "Proven Optimal Early"
        elif model_opt.SolCount > 0:
            results[file_name]['Optimal_Obj'] = model_opt.ObjVal
            #results[file_name]['Optimal_Status'] = f"Feasible (Hit {HOURS_LIMIT}h Limit, Gap: {model_opt.MIPGap:.4f})"
        else:
            status_str = f"Limit Expired ({HOURS_LIMIT}h) - No Sol" if model_opt.Status == GRB.TIME_LIMIT else f"Status Code: {model_opt.Status}"
            results[file_name]['Optimal_Obj'] = None
            #results[file_name]['Optimal_Status'] = status_str
        results[file_name]['Optimal_Time'] = str(datetime.timedelta(seconds=time4-time3))
        model_opt.dispose()  # Free memory after optimal mode
        # -------------------------------------------------------------
        # MODE 3: Default Mode
        # -------------------------------------------------------------
        model_def = gp.read(file_path, env)
        model_def.Params.TimeLimit = TIMEOUT_SECONDS  # Applied here
        time5 = time.perf_counter()
        model_def.optimize()
        time6 = time.perf_counter()

        if model_def.Status == GRB.OPTIMAL:
            results[file_name]['Default_Obj'] = model_def.ObjVal
            #results[file_name]['Default_Status'] = "Proven Optimal Early"
        elif model_def.SolCount > 0:
            results[file_name]['Default_Obj'] = model_def.ObjVal
            #results[file_name]['Default_Status'] = f"Stopped at {HOURS_LIMIT}h Limit (Gap: {model_def.MIPGap:.4f})"
        else:
            status_str = f"Limit Expired ({HOURS_LIMIT}h) - No Sol" if model_def.Status == GRB.TIME_LIMIT else f"Status Code: {model_def.Status}"
            results[file_name]['Default_Obj'] = None
            #results[file_name]['Default_Status'] = status_str
        results[file_name]['Default_Time'] = str(datetime.timedelta(seconds=time6-time5))
        model_def.dispose()  # Free memory after default mode

    return results

   

if __name__ == "__main__":
    results = solve(r"C:\Users\KISHAN\Downloads\smokeTest")
    # 1. Convert dictionary to DataFrame
    df = pd.DataFrame(results)

    # 2. Transpose so Model names become Rows for now, metrics become Columns
    df = df.T
    columns_hierarchy = [
    ("Num_Vars", "", ""),
    ("Num_Constrs", "", ""),
    ("Gurobipy", "Feasible", "Obj"),
    ("Gurobipy", "Feasible", "Time"),

    ("Gurobipy", "Optimal", "Obj"),
    ("Gurobipy", "Optimal", "Time"),

    ("Gurobipy", "Default", "Obj"),
    ("Gurobipy", "Default", "Time")]
    df.columns = pd.MultiIndex.from_tuples(
    columns_hierarchy, 
    names=["Solver", "Mode", "Metrics"])
    df.columns = pd.MultiIndex.from_tuples(columns_hierarchy)

# 3. Sort columns so the "Performance" categories group together completely
    df = df.sort_index(axis=1)
    #df.to_csv("gurobi_results.csv", index= True)
    df.to_csv("gurobi_results.csv", index= True, float_format='%.4f')
    print(df)




    