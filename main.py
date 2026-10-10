# Final Act of Courage
import csv
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
# Force stdout and stderr to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# Reconfigure the existing streams safely
#sys.stdout.reconfigure(encoding='utf-8')
#sys.stderr.reconfigure(encoding='utf-8')

import multiprocessing
import concurrent.futures
import os
from pathlib import Path
import numpy as np
import time
import random
from IPython.display import display
import pandas as pd
import datetime
import matplotlib
import scipy.sparse as sp
#matplotlib.use("Agg")   # <<< CRITICAL for server (no display)
import json
import gurobipy as gp
import pyscipopt
from pyscipopt import Model, Branchrule, Nodesel, SCIP_RESULT,SCIP_PARAMSETTING , SCIP_STATUS, SCIP_BASESTAT
matplotlib.rcParams['figure.max_open_warning'] = 0
from IPython.display import clear_output
clear_output(wait=True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gurobipy_functions import solve_bip_gurobi,extract_sparse_gurobipy_matrices,sigma
from scip_strongBrancher import SinglePassStrongBrancher, scip_sb
from Learn import actor_critic_nn, twoCritic, learning_hours
#from scip_learn import actor_critic_nn, twoCritic
from Test_Display import plot_any_returns, run_policy_nn, run_uniform, category, load_checkpoint, save_checkpoint,plot_splitAxes, plot_obj_change
from scip_sb_TreeEp_environ import scip_MyMDPEnv
from environ_mdp3 import MyMDPEnv
from environ_TreeEpisode import TreeEp_MyMDPEnv
from env_TreeEp2 import TreeEp2_MyMDPEnv
from scip_TreeEpsisode import solve_bip_scipws

#Time_Limit = 1800
CHECKPOINT_FILE = "processed_files.json"
start_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
end_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
alpha = 1
#hash_ep = int(sys.argv[1]) if len(sys.argv) > 0 else 2
N_runs    = 1
results = []

master_file_path = "smoke_test_Ga300_mdp1.txt"
master_csv_path = "smoke_test_Ga300_mdp1.csv"
folder_end = "smoke_test_Ga300_mdp1" 

'''master_file_path = f"{category[0].upper()}_LftOvr_{end_ep}_tr+100ε1G_TreeEp.txt"
master_csv_path = f"{category[0].upper()}_LftOvr_{end_ep}_tr+100ε1G_TreeEp.csv"
folder_end = f"{category[0].upper()}_LftOvr_{end_ep}_tr+100ε1G_TreeEp" '''

'''master_file_path = f"2club200v15p5scn_tr2{end_ep}.txt"
master_csv_path = f"2club200v15p5scn_tr2{end_ep}.csv"
folder_end = f"tr2_{end_ep}" '''

'''master_file_path = f"2club200v15p5scn_tr2_scip_sb.txt"
master_csv_path = f"2club200v15p5scn_tr2_scip_sb.csv"
folder_end = f"2club200v15p5scn_tr2_scip_sb" '''



def set_seed(seed: int = 42):
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. NumPy random seed
    np.random.seed(seed)
    
    # 3. PyTorch CPU & GPU seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 4. PyTorch Deterministic Operations (ensures exact CPU/GPU operations)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def experiment(file):
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
        try:
          csv_vals =[]
          gp_env = gp.Env(empty=True)
          gp_env.setParam("OutputFlag", 0)
          gp_env.setParam("FeasibilityTol", 1e-9)
          gp_env.start()

          #from Learn import episodes
          filename  = os.path.basename(file)         # "2club200v15p5scn.mps"
          name_only = os.path.splitext(filename)[0]
          print(f"\n\n{'#'*10}\nProcessing file: {filename}\n{'#'*10}")
          gmodel = gp.read(file, gp_env)
          A_scaled, b_scaled, le_mask, ge_mask, eq_mask = extract_sparse_gurobipy_matrices(gmodel)
          N = gmodel.NumIntVars
          M = gmodel.NumConstrs
          sense = gmodel.ModelSense
          #n_steps = N + 50
          print(f"\n\nproblem Name: {name_only}, Num_Vars: {N}, Num_Constrs: {M}")
          if sense == gp.GRB.MINIMIZE:  # or sense == 1
            print("The problem is a MINIMIZATION problem.")
          elif sense == gp.GRB.MAXIMIZE:  # or sense == -1
            print("The problem is a MAXIMIZATION problem.")
          #print("\nObjective Function Expression:")
          #print(gmodel.getObjective())
          gmodel = gmodel.relax()  # Relax the model to get fractional solutions
          time1 = time.perf_counter()
          tc, obj_val, x_val = solve_bip_gurobi(gmodel, vec=None, relax=False)
          time2 = time.perf_counter()
          print(f"Time taken by solver:", datetime.timedelta(seconds=time2-time1))
          #print("Number of binary variables:", N)
          #print("Number of constraints:", M)
          print("Solution of the Relaxation:------------")
          #print(f"\nX: {np.round(x_val,2)}, Obj = {np.round(obj_val,2)}")
          print(f"\nObj = {np.round(obj_val,2)}")

          time3 = time.perf_counter()
          env = MyMDPEnv(N,gmodel,A_scaled, b_scaled, le_mask, ge_mask, eq_mask) 

          for episode in range(start_ep, end_ep + 1, 200):
          #for alpha in np.linspace(0.0, 1.0, 11):
            #alpha = round(alpha, 1)
            all_returns = []
            '''all_vio_constr =[]
            all_vio_bounds = []'''
            #state, _ = env.reset()
            #print(f"\nInitial state: {state}")            
            #print(f"\nIs terminal: {env.is_terminal(state)[0]}")
            print(f"\n{'='*50}")
            print(f"Running for {episode} episodes")
            #print(f"Running for alpha = {alpha}")
            print(f"{'='*50}")
            
            actor, critic, episode_returns, vio_constr, vio_bounds, success, obj_change_list = actor_critic_nn(
                      prob       = name_only,
                      env        = env,
                      alpha      = alpha,
                      Time_Limit = learning_hours,
                      n_episodes = episode,  # <<< CHANGE >>> number of episodes to train
                      gamma      = 1.0,                 # <<< CHANGE >>> no discounting for episodic tasks
                      alpha_w    = 1e-3,
                      alpha_theta= 1e-4,
                      tau        = 0.1,
                      hidden_dim = 128,                 # <<< CHANGE >>> increase for complex problems
                      device= device,
                      entropy_coeff_start = 0.9,    # <<< high exploration at start
                      entropy_coeff_end = 0.01,
                      epsilon = 0.2)
            print(episode_returns)
            all_returns.append(episode_returns)   # Storing episodic returns of all the episodes in each run
            #all_vio_constr.append(vio_constr)
            #all_vio_bounds.append(vio_bounds)
            time4 = time.perf_counter()
            print("\nTraining complete.")
            #print(f"Total time alpha {alpha}:", datetime.timedelta(seconds=time4-time3))
            print(f"Total time in {episode} episodes:", datetime.timedelta(seconds=time4-time3))
            #mean_returns = np.mean(all_returns, axis=0)   # shape: (m_episodes,) — avg of ep_i across N runs
            #std_returns  = np.std(all_returns,  axis=0)   # shape: (m_episodes,) — std  of ep_i across N runs
            
            #plot_averaged_returns(name_only,all_returns, episode)
            #plot_violn(file, all_vio_constr, all_vio_bounds, episode)
            #plot_obj_change(name_only, obj_change_list, episode)
            print(f"Running the learnt policy to solve the {name_only}:**************************************************")
            time5 = time.perf_counter()
            trajectory, total_return, obj_final, x_final, reason, entropy,max_entropy = run_policy_nn(env, actor, alpha, device=device, hours=2)   # <<< CHANGE >>>
            time6 = time.perf_counter()
            #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
            print(f"Time taken by {episode} episode_policy :", datetime.timedelta(seconds=time6-time5))
            print(f"Trajectory length: {len(trajectory)}")
            print(f"Total return: {total_return:.2f}")

            # Inspect policy at any state — e.g. the first state in the trajectory
            '''state0, _, _ = trajectory[0]
            valid = env.compute_feasible_actions(state0)
            print("\nTrajectory carved by the policy:---")
            trajectory = print_trajectory(trajectory, env)
            #traj_dir = f"Experiment_trajectories\\hard_instances\\cubed\\{episodes}ep_a{alpha}b{beta}g{gamma}s{sigma}"  # <<< CHANGE >>> specify your desired folder here
            traj_dir = f"Experiment_trajectories\\smoke_test\\cubed\\{episodes}ep_a{alpha}b{beta}g{gamma}s{sigma}"  # <<< CHANGE >>> specify your desired folder here

            os.makedirs(traj_dir, exist_ok=True)
            trajectory.to_csv(f"{traj_dir}/{name_only}_trajectory.csv", index= True)'''

            #print(f"\nSolution in Final State: {x_final}")
            print(f"\nObjective in Final State: {obj_final}")
            print(f"\nTotal Return: {total_return:.2f} over {len(trajectory)-1} steps")
            print(f"\nReason for termination: {reason}")
            csv_vals.append({
            "Problem": name_only,
            "#Vars": N,
            "#Cons": M,
            "#Episodes": episode,
            "RL_Obj": f"{obj_final:.2f}" if obj_final is not None else obj_final,
            "#Success": success,
            "Vio_Constr": f"{(np.sum(vio_constr)/episode):.2f}",
            "Vio_Bounds": f"{(np.sum(vio_bounds)/episode):.2f}",
            "Ent": f"{entropy:.2f}",
            "Max_Ent": f"{max_entropy:.2f}"})
          '''csv_vals.append({
          "Problem": "----",
          "#Vars": "----",
          "#Cons": "----",
          "#Episodes":"----" ,
          "RL_Obj": "----",
          "#Success":"----",
          "Vio_Constr": "----",
          "Vio_Bounds": "----",
          "AvgEp_Time": "----",
          "Ent": "----",
          "Max_Ent": "----",
          "Train_Time": "----",
          "Policy_Time": "----"})'''
          #processed_files.add(name_only)
          save_checkpoint(CHECKPOINT_FILE, name_only)
        except Exception as e:
            # Captures internal exception tracebacks inside the log buffer
            print(f"\n[ERROR] Function failed for {file}: {e}")
            traceback.print_exc()
        return name_only, csv_vals, log_buffer.getvalue()  # Return the log buffer content for this file

def extract_sparse_scip_matrices(model):
    """
    Extracts the entire .mps file into SciPy CSR sparse matrices ONCE at startup.
    """

    all_vars = model.getVars()
    all_constrs = model.getConss()

    num_vars = len(all_vars)
    num_constrs = len(all_constrs)

    # 1. Coordinate collection lists for efficient sparse building
    row_indices = []
    col_indices = []
    coefficients = []

    b_full = np.zeros(num_constrs)
    senses = np.zeros(num_constrs, dtype=object)

    # Map variable string names to their list position/index for fast lookup
    var_name_to_idx = {var.name: i for i, var in enumerate(all_vars)}

    # 2. Extract structural data via PySCIPOpt
    for c_idx, constr in enumerate(all_constrs):
        # PySCIPOpt returns a dictionary: {var_name_string: coefficient}
        vals_linear = model.getValsLinear(constr)
        
        for var_name, coeff in vals_linear.items():
            v_idx = var_name_to_idx[var_name]
            
            row_indices.append(c_idx)
            col_indices.append(v_idx)
            coefficients.append(coeff)
        
        lhs = model.getLhs(constr)
        rhs = model.getRhs(constr)

        # Reconstruct standard constraint senses and RHS values
        # Case 1: Equality constraint (LHS == RHS)
        if lhs is not None and rhs is not None and abs(lhs - rhs) < 1e-9:
            b_full[c_idx] = rhs
            senses[c_idx] = '='
        # Case 2: Upper bound constraint (Ax <= RHS)
        elif rhs is not None and rhs < 1e10:
            b_full[c_idx] = rhs
            senses[c_idx] = '<'
        # Case 3: Lower bound constraint (Ax >= LHS)
        elif lhs is not None and lhs > -1e10:
            b_full[c_idx] = lhs
            senses[c_idx] = '>'

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

def sameSeed_experiment(file):
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
          set_seed(42)  # Set a fixed seed for reproducibility
          gp_env = gp.Env(empty=True)
          gp_env.setParam("OutputFlag", 0)
          gp_env.setParam("FeasibilityTol", 1e-9)
          gp_env.start()

          #from Learn import episodes
          filename  = os.path.basename(file)         # "2club200v15p5scn.mps"
          name_only = os.path.splitext(filename)[0] 
          print(f"\n\n{'#'*10}\nProcessing file: {filename}\n{'#'*10}")

          gmodel = gp.read(file, gp_env)
          A_scaled, b_scaled, le_mask, ge_mask, eq_mask = extract_sparse_gurobipy_matrices(gmodel)
          N = gmodel.NumIntVars
          M = gmodel.NumConstrs
          sense = gmodel.ModelSense
          #n_steps = N + 50
          print(f"\n\nproblem Name: {name_only}, Num_Vars: {N}, Num_Constrs: {M}")
          if sense == gp.GRB.MINIMIZE:  # or sense == 1
            print("The problem is a MINIMIZATION problem.")
          elif sense == gp.GRB.MAXIMIZE:  # or sense == -1
            print("The problem is a MAXIMIZATION problem.")
          #print("\nObjective Function Expression:")
          #print(gmodel.getObjective())
          gmodel = gmodel.relax()  # Relax the model to get fractional solutions
          time1 = time.perf_counter()
          tc, obj_val, x_val = solve_bip_gurobi(gmodel, vec=None, relax=False)
          time2 = time.perf_counter()
          print(f"Time taken by solver:", datetime.timedelta(seconds=time2-time1))
          #print("Number of binary variables:", N)
          #print("Number of constraints:", M)
          print("Solution of the Relaxation:------------")
          #print(f"\nX: {np.round(x_val,2)}, Obj = {np.round(obj_val,2)}")
          print(f"\nObj = {np.round(obj_val,2)}")

          time3 = time.perf_counter()
          env = TreeEp2_MyMDPEnv(N, M, gmodel,A_scaled, b_scaled, le_mask, ge_mask, eq_mask) 
          for run in range(N_runs):
                for episode in range(start_ep, end_ep + 1, 200):
                #for alpha in np.linspace(0.0, 1.0, 11):
                    #alpha = round(alpha, 1)
                    all_returns = []
                    all_obj_change = []
                    all_qs0 = []
                    all_critic_losses = []
                    all_critic1_losses = []
                    all_critic2_losses = []
                    all_actor_losses = []
                    '''all_vio_constr =[]
                    all_vio_bounds = []'''
                    #state, _ = env.reset()
                    #print(f"\nInitial state: {state}")            
                    #print(f"\nIs terminal: {env.is_terminal(state)[0]}")
                    print(f"\n{'='*50}")
                    print(f"Running for {episode} episodes")
                    #print(f"Running for alpha = {alpha}")
                    print(f"{'='*50}")
                    
                    actor, episode_returns, qs0, critic1_losses,critic2_losses,critic_losses, actorGradNorm, obj_change_list, csv_vals = twoCritic(
                            vars       = N,
                            cons       = M,
                            prob       = name_only,
                            env        = env,
                            alpha      = alpha,
                            Time_Limit = learning_hours,
                            n_episodes = episode,  # <<< CHANGE >>> number of episodes to train
                            gamma      = 1.0,                 # <<< CHANGE >>> no discounting for episodic tasks
                            alpha_w    = 1e-3,
                            alpha_theta= 1e-4,
                            tau        = 0.005,
                            hidden_dim = 128,                 # <<< CHANGE >>> increase for complex problems
                            device= device,
                            entropy_coeff_start = 0.9,    # <<< high exploration at start
                            entropy_coeff_end = 0.01,
                            epsilon1 = 1.0
                            )
                    #print(episode_returns)
                    #dir = f"experiment_plots\\Training_Curves\\{category}\\qs0_{end_ep}Ep_mdp3"  # <<< CHANGE >>> specify your desired folder here
                    all_returns.append(episode_returns)   # Storing episodic returns of all the episodes in each run
                    all_qs0.append(qs0)  # Storing Q-values of the initial state in each run
                    #all_critic_losses.append(critic_losses)  # Storing critic losses of all the episodes in each run
                    all_critic1_losses.append(critic1_losses)  # Storing critic1 losses of all the episodes in each run
                    #all_critic2_losses.append(critic2_losses)  # Storing critic2 losses of all the episodes in each run
                    all_actor_losses.append(actorGradNorm)  # Storing actor gradient norms of all the episodes in each run
                    flot_list = [float(x) for x in obj_change_list]
                    print(f"\n List of mean Obj_Change per episode for all episodes: {flot_list}")
                    all_obj_change.append(obj_change_list)  # Storing objective value changes
                    #all_vio_constr.append(vio_constr)
                    #all_vio_bounds.append(vio_bounds)
                    time4 = time.perf_counter()
                    print("\nTraining complete.")
                    #print(f"Total time alpha {alpha}:", datetime.timedelta(seconds=time4-time3))
                    print(f"Total time in {episode} episodes:", datetime.timedelta(seconds=time4-time3))
                    #plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{name_only}_{folder_end}\\critLoss_{end_ep}Ep_mdp3","(Critic1 + Critic2) Loss",all_critic_losses, N, M)
                    plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\actorLoss_{end_ep}Ep_{folder_end}","avg_policyGradNorm",all_actor_losses, N, M)
                    plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\V(S0)_{end_ep}Ep_{folder_end}","V(s0)",all_qs0, N, M)
                    plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\critic1Loss_{end_ep}Ep_{folder_end}","Critic1 Loss",all_critic1_losses, N, M)
                    #plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{name_only}_{folder_end}\\critic2Loss_{end_ep}Ep_{folder_end}","Critic2 Loss",all_critic2_losses, N, M)
                    plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\returns_{end_ep}Ep_{folder_end}","TotReturns",all_returns, N, M)
                    #plot_violn(file, all_vio_constr, all_vio_bounds, episode)
                    #plot_obj_change(name_only, all_obj_change, episode, N, M)
                    print(f"Running the learnt policy to solve the {name_only} finally:**************************************************")
                    time5 = time.perf_counter()
                    trajectory, total_return, obj_final, x_final, reason, entropy,max_entropy = run_policy_nn(env, actor, device=device, hours= 0.1)   # <<< CHANGE >>>
                    time6 = time.perf_counter()
                    #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
                    print(f"Time taken by {episode} episode_policy :", datetime.timedelta(seconds=time6-time5))
                    print(f"Trajectory length: {len(trajectory)}")
                    print(f"Total return: {total_return:.2f}")
                    print(f"\nObjective in Final State: {obj_final}")
                    print(f"\nTotal Return: {total_return:.2f} over {len(trajectory)-1} steps")
                    print(f"\nReason for termination: {reason}")

          os.makedirs(rf'D:\MINLPRL\policy_files\{category}\{name_only}_{folder_end}\{end_ep}Ep_mdp3', exist_ok=True)
          torch.save(actor.state_dict(), rf'D:\MINLPRL\policy_files\{category}\{name_only}_{folder_end}\{end_ep}Ep_mdp3\{name_only}.pt')
          csv_vals.append({
          "Problem": "----",
          "#Vars": "----",
          "#Cons": "----",
          "#Episodes":"----" ,
          "RL_Obj": "----",
          "Mean_Rwd": "----",
          "V(S0)": "----",
          "#Success":"----",
          #"Vio_Constr": "----",
          #"Vio_Bounds": "----",
          "Obj_Change": "----",
          "AvgEp_Time": "----",
          "Ent": "----",
          "Max_Ent": "----",
          "critic1_loss": "----",
          #"critic2_loss": "----",
          "actor_loss": "----",
          "Train_Time": "----",
          "Policy_Time": "----"})
          #processed_files.add(name_only)
          save_checkpoint(CHECKPOINT_FILE, name_only)

          return name_only, csv_vals, log_buffer.getvalue()  # Return the log buffer content for this file


def scip_sameSeed_exp(file):
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            set_seed(42)
            filename  = os.path.basename(file)         # "2club200v15p5scn.mps"
            name_only = os.path.splitext(filename)[0] 
            print(f"\n\n{'#'*10}\nProcessing file: {filename}\n{'#'*10}")

            # 1. Initialize PySCIPOpt Model
            smodel = Model()
            #smodel.includeDefaultPlugins()
            #smodel.setLogfile("scip_solver_log.txt")



            smodel.hideOutput()  # Equivalent to gp_env.setParam("OutputFlag", 0)
            # Alternatively, set specific quiet parameters
            smodel.setParam("display/verblevel", 0)

            # 2. Read MPS file
            smodel.readProblem(file)

            # 3. Extract matrix data via PySCIPOpt function
            A_scaled, b_scaled, le_mask, ge_mask, eq_mask = extract_sparse_scip_matrices(smodel)

            # 4. Count variables and constraints
            # Filter integer/binary variables for N (matches Gurobi's NumIntVars)
            N = sum(1 for v in smodel.getVars() if v.vtype() in ["BINARY", "INTEGER"])
            M = smodel.getNConss()
            sense = smodel.getObjectiveSense()

            print(f"\n\nproblem Name: {name_only}, Num_Vars: {N}, Num_Constrs: {M}")

            if sense == "minimize":
                print("The problem is a MINIMIZATION problem.")
            elif sense == "maximize":
                print("The problem is a MAXIMIZATION problem.")

            # 5. Relax the model (relax integer variables to continuous)
            smodel.relax()
            env = MyMDPEnv(N, M, smodel, A_scaled, b_scaled, le_mask, ge_mask, eq_mask) 
            time1 = time.perf_counter()
            tc, obj_val, x_val, _, _ = solve_bip_scipws(smodel, vec=None, relax=False, warm_start= None, vbasis= None, cbasis = None)
            #tc, obj_val, x_val,br_id, br_dir, _, _ = scip_sb(smodel,warm_start= None,vbasis= None, cbasis= None) 
            time2 = time.perf_counter()
            print(f"Time taken by solver:", datetime.timedelta(seconds=time2-time1))
            print("Solution of the Relaxation:------------")
            print(f"\nObj = {np.round(obj_val,2)}")

            time3 = time.perf_counter()


            for episode in range(start_ep, end_ep + 1, 200):
            #for alpha in np.linspace(0.0, 1.0, 11):
                #alpha = round(alpha, 1)
                all_returns = []
                all_obj_change = []
                all_qs0 = []
                all_critic_losses = []
                all_critic1_losses = []
                all_critic2_losses = []
                all_actor_losses = []
                '''all_vio_constr =[]
                all_vio_bounds = []'''
                #state, _ = env.reset()
                #print(f"\nInitial state: {state}")            
                #print(f"\nIs terminal: {env.state_info(state)[0]}")
                print(f"\n{'='*50}")
                print(f"Running for {episode} episodes")
                #print(f"Running for alpha = {alpha}")
                print(f"{'='*50}")
                
                actor, episode_returns, qs0, critic1_losses,critic2_losses,critic_losses, actorGradNorm, csv_vals = twoCritic(
                                vars       = N,
                                cons       = M,
                                prob       = name_only,
                                env        = env,
                                alpha      = alpha,
                                Time_Limit = learning_hours,
                                n_episodes = episode,  # <<< CHANGE >>> number of episodes to train
                                gamma      = 1.0,                 # <<< CHANGE >>> no discounting for episodic tasks
                                alpha_w    = 1e-3,
                                alpha_theta= 1e-4,
                                tau        = 0.005,
                                hidden_dim = 128,                 # <<< CHANGE >>> increase for complex problems
                                device= device,
                                entropy_coeff_start = 0.9,    # <<< high exploration at start
                                entropy_coeff_end = 0.01,
                                #epsilon = 1.0
                                )
                #print(episode_returns)
                all_returns.append(episode_returns)   # Storing episodic returns of all the episodes in each run
                all_qs0.append(qs0)
                all_critic1_losses.append(critic1_losses)
                all_actor_losses.append(actorGradNorm)
                #flot_list = [float(x) for x in obj_change_list]
                #print(f"\n List of mean Obj_Change per episode for all episodes: {flot_list}")
                #all_obj_change.append(obj_change_list)  # Storing objective value changes
                #all_vio_constr.append(vio_constr)
                #all_vio_bounds.append(vio_bounds)
                time4 = time.perf_counter()
                print("\nTraining complete.")
                #print(f"Total time alpha {alpha}:", datetime.timedelta(seconds=time4-time3))
                print(f"Total time in {episode} episodes:", datetime.timedelta(seconds=time4-time3))
                #plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{name_only}_{folder_end}\\critLoss_{end_ep}Ep_mdp3","(Critic1 + Critic2) Loss",all_critic_losses, N, M)
                plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\actorLoss_{end_ep}Ep_{folder_end}","avg_policyGradNorm",all_actor_losses, N, M)
                plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\V(S0)_{end_ep}Ep_{folder_end}","V(s0)",all_qs0, N, M)
                plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\critic1Loss_{end_ep}Ep_{folder_end}","Critic1 Loss",all_critic1_losses, N, M)
                #plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{name_only}_{folder_end}\\critic2Loss_{end_ep}Ep_{folder_end}","Critic2 Loss",all_critic2_losses, N, M)
                plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\returns_{end_ep}Ep_{folder_end}","TotReturns",all_returns, N, M)
            
                print(f"Running the learnt policy to solve the {name_only} finally:**************************************************")
                time5 = time.perf_counter()
                trajectory, total_return, obj_final, x_final, reason, entropy,max_entropy = run_policy_nn(env, actor, device=device, hours=0.1)   # <<< CHANGE >>>
                time6 = time.perf_counter()
                #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
                print(f"Time taken by {episode} episode_policy :", datetime.timedelta(seconds=time6-time5))
                print(f"Trajectory length: {len(trajectory)}")
                print(f"Total return: {total_return:.2f}")
                print(f"\nObjective in Final State: {obj_final}")
                print(f"\nTotal Return: {total_return:.2f} over {len(trajectory)-1} steps")
                print(f"\nReason for termination: {reason}")

            os.makedirs(rf'D:\MINLPRL\policy_files\{category}\{name_only}_{folder_end}\{end_ep}Ep_mdp3', exist_ok=True)
            torch.save(actor.state_dict(), rf'D:\MINLPRL\policy_files\{category}\{name_only}_{folder_end}\{end_ep}Ep_mdp3\{name_only}.pt')
            csv_vals.append({
            "Problem": "----",
            "#Vars": "----",
            "#Cons": "----",
            "#Episodes":"----" ,
            "RL_Obj": "----",
            "Mean_Rwd": "----",
            "V(S0)": "----",
            "#Success":"----",
            #"Vio_Constr": "----",
            #"Vio_Bounds": "----",
            #"Obj_Change": "----",
            "AvgEp_Time": "----",
            "Ent": "----",
            "Max_Ent": "----",
            "critic1_loss": "----",
            #"critic2_loss": "----",
            "actor_loss": "----",
            "Train_Time": "----",
            "Policy_Time": "----"})
            #processed_files.add(name_only)
            save_checkpoint(CHECKPOINT_FILE, name_only)

            return name_only, csv_vals, log_buffer.getvalue()  # Return the log buffer content for this file
               
            
def uniform_experiment(file):

        csv_vals =[]
        gp_env = gp.Env(empty=True)
        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("FeasibilityTol", 1e-9)
        gp_env.start()

        #from Learn import episodes
        filename  = os.path.basename(file)         # "2club200v15p5scn.mps"
        name_only = os.path.splitext(filename)[0]
        print(f"\n\n{'#'*10}\nProcessing file: {filename}\n{'#'*10}")


        gmodel = gp.read(file, gp_env)
        
        
        A_scaled, b_scaled, le_mask, ge_mask, eq_mask = extract_sparse_gurobipy_matrices(gmodel)
        N = gmodel.NumIntVars
        M = gmodel.NumConstrs
        sense = gmodel.ModelSense
        #n_steps = N + 50
        print(f"\n\nproblem Name: {name_only}, Num_Vars: {N}, Num_Constrs: {M}")
        if sense == gp.GRB.MINIMIZE:  # or sense == 1
            print("The problem is a MINIMIZATION problem.")
        elif sense == gp.GRB.MAXIMIZE:  # or sense == -1
            print("The problem is a MAXIMIZATION problem.")
        #print("\nObjective Function Expression:")
        #print(gmodel.getObjective())
        gmodel = gmodel.relax()  # Relax the model to get fractional solutions
        time1 = time.perf_counter()
        tc, obj_val, x_val = solve_bip_gurobi(gmodel, vec=None, relax=False)
        time2 = time.perf_counter()
        print(f"Time taken by solver:", datetime.timedelta(seconds=time2-time1))
        #print("Number of binary variables:", N)
        #print("Number of constraints:", M)
        print("Solution of the Relaxation:------------")
        #print(f"\nX: {np.round(x_val,2)}, Obj = {np.round(obj_val,2)}")
        print(f"\nObj = {np.round(obj_val,2)}")

        time3 = time.perf_counter()
        env = MyMDPEnv(N,gmodel,A_scaled, b_scaled, le_mask, ge_mask, eq_mask) 

        print("\n@@@@@@@@@@@@@@@@@@@@@@ Running Uniform Action Selector @@@@@@@@@@@@@@@@@@@@@@")          
        time5 = time.perf_counter()
        trajectory, total_return, obj_final, x_final, reason = run_uniform(env, alpha, device=None, hours=2)   # <<< CHANGE >>>
        time6 = time.perf_counter()
        #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
        print(f"Time taken by uniform_policy :", datetime.timedelta(seconds=time6-time5))
        print(f"Trajectory length: {len(trajectory)}")
        print(f"Total return: {total_return:.2f}")


        #print(f"\nSolution in Final State: {x_final}")
        print(f"\nObjective in Final State: {obj_final}")
        print(f"\nTotal Return: {total_return:.2f} over {len(trajectory)-1} steps")
        print(f"\nReason for termination: {reason}")
        csv_vals.append({
        "Problem": name_only,
        "#Vars": N,
        "#Cons": M,
        "Uni_Obj": f"{obj_final:.2f}" if obj_final is not None else obj_final})

        #processed_files.add(name_only)
        save_checkpoint(CHECKPOINT_FILE, name_only)

        return name_only, csv_vals 
# ═══════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════

if __name__ == "__main__":
    
    num_processes = multiprocessing.cpu_count() 
    csv_header_written = False
    folder_path = rf"C:\Users\KISHAN\Downloads\{category}"
    #folder_path = r"C:\Users\KISHAN\Downloads\smoke_test"
    #mps_files = [f for f in os.listdir(folder_path) if f.endswith('.mps') or f.endswith('.mps.gz')]
    # Get and sort all files in the folder
    #all_files = sorted([os.path.splitext(os.path.basename(f))[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    all_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    processed_files = load_checkpoint(CHECKPOINT_FILE)
    time_exp_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers= num_processes) as executor:
            with open(master_file_path, "w", encoding="utf-8") as master_file, \
             open(master_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            #with open(master_file_path, "w", encoding="utf-8") as master_file:
                csv_writer = csv.writer(csv_file)
     
                '''future_to_file = {
                    executor.submit(experiment, os.path.join(folder_path, file_name)): file_name 
                    for file_name in all_files if file_name not in processed_files
                }'''
                '''future_to_file = {
                    executor.submit(uniform_experiment, os.path.join(folder_path, file_name)): file_name 
                    for file_name in all_files if file_name not in processed_files
                }'''
                future_to_file = {
                    executor.submit(sameSeed_experiment, os.path.join(folder_path, file_name)): file_name 
                    for file_name in all_files if file_name not in processed_files 
                }  
                for future in concurrent.futures.as_completed(future_to_file):
                    # We look up what file name this future started with just to handle errors safely
                    start_file_name = future_to_file[future] 
                    
                    #try:
                    # Unpack the returned tuple (filename and the bundle)
                    returned_file_name,results_bundle,log_content = future.result()
                    
                    # Map the bundle to the key in your master dictionary
                    results.extend(results_bundle)

                    #print(f"\nprinting dictionary at {returned_file_name}:\n {results[returned_file_name]}")
                    
                    print(f"Successfully mapped results for {returned_file_name}")
                    # Write in-memory logs directly to master file on the fly
                    master_file.write(f"=== Output for {returned_file_name} ===\n")
                    master_file.write(log_content)
                    master_file.write("\n\n")
                    master_file.flush()  # Forces immediate disk write

                    # 3. Preserve results_bundle structure in CSV
                    if results_bundle:
                        first_item = results_bundle[0]

                        # Case A: results_bundle is a list of dictionaries with key/value pairs
                        if isinstance(first_item, dict):
                            keys = list(first_item.keys())
                            
                            # Dynamically write CSV headers on the very first completion
                            if not csv_header_written:
                                csv_writer.writerow(keys)
                                csv_header_written = True

                            # Write each dictionary row matching the extracted keys
                            for row_dict in results_bundle:
                                csv_writer.writerow([row_dict.get(k, "") for k in keys])

                        # Case B: results_bundle is a list of structured rows (lists/tuples)
                        elif isinstance(first_item, (list, tuple)):
                            for row in results_bundle:
                                csv_writer.writerow(list(row))

                        csv_file.flush()  # Force write to disk
                        
                    '''except Exception as e:
                        error_msg= f"Error occurred while processing {start_file_name}: {e}"
                        print(error_msg)
                        traceback.print_exc()
                        master_file.write(f"=== Output for {start_file_name} (CRASHED) ===\n")
                        master_file.write(error_msg)
                        master_file.write("\n\n")
                        master_file.flush()

                        # Record the failure row in the CSV
                        csv_writer.writerow([error_msg])
                        csv_file.flush()'''
    #print(f"\nPrinting the results dictionary{results}")
    
    time_exp_end = time.perf_counter()
    total_exp_time = time_exp_end - time_exp_start
    print(f"\nTotal Parallel Experiment Time = {total_exp_time:.2f} Seconds")
    #print(f"\nTotal Parallel uniform_experiment Time = {total_exp_time:.2f} Seconds")
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.2f}'.format
    #df = pd.read_csv(f"{category[0].upper()}{end_ep}_mdp3.csv", encoding = 'utf-8')
    df = pd.read_csv(f"{master_csv_path}", encoding = 'utf-8')

    print(df)
    with open("processed_files.json", "w") as f:
        json.dump([], f, indent=4)




  
