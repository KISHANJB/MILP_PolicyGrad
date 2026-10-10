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
from Test_Display import plot_any_returns, run_policy_nn, run_uniform, category, load_checkpoint, save_checkpoint,plot_ma_returns
from scip_sb_TreeEp_environ import scip_MyMDPEnv
from environ_mdp3 import MyMDPEnv
from environ_TreeEpisode import TreeEp_MyMDPEnv
from env_TreeEp2 import TreeEp2_MyMDPEnv
from inst_gen import generate_indset, generate_setcover
from Test_Display import load_checkpoint, save_checkpoint
category = sys.argv[3] if len(sys.argv) > 3 else 'setcover'


#Time_Limit = 1800
CHECKPOINT_FILE = "processed_files.json"
start_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
end_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
alpha = 1
#hash_ep = int(sys.argv[1]) if len(sys.argv) > 0 else 2
N_runs    = 1
results = []

master_file_path = f"D:\\Logs & CSVs\\logs{category}_ma_Mdp3_{end_ep}Ep.txt"
master_csv_path = f"D:\\Logs & CSVs\\csv{category}_ma_Mdp3_{end_ep}Ep.csv"
folder_end = f"SynInst_ma_Mdp3_{end_ep}Ep" 

'''master_file_path = f"{category[0].upper()}_LftOvr_{end_ep}_tr+100ε1G_TreeEp.txt"
master_csv_path = f"{category[0].upper()}_LftOvr_{end_ep}_tr+100ε1G_TreeEp.csv"
folder_end = f"{category[0].upper()}_LftOvr_{end_ep}_tr+100ε1G_TreeEp" '''

'''master_file_path = f"2club200v15p5scn_tr2{end_ep}.txt"
master_csv_path = f"2club200v15p5scn_tr2{end_ep}.csv"
folder_end = f"tr2_{end_ep}" '''

'''master_file_path = f"2club200v15p5scn_tr2_scip_sb.txt"
master_csv_path = f"2club200v15p5scn_tr2_scip_sb.csv"
folder_end = f"2club200v15p5scn_tr2_scip_sb" '''

'''master_file_path = f"S_000_stPen_Tr2.txt"
master_csv_path = f"S_000_stPen_Tr2.csv"
folder_end = f"S_000_stPen_Tr2" '''

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


def syntheticInst_experiment(gmodel, name_only):
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
        set_seed(42)  # Set a fixed seed for reproducibility
        gp_env = gp.Env(empty=True)
        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("FeasibilityTol", 1e-9)
        gp_env.start()
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
        env = MyMDPEnv(N, M, gmodel,A_scaled, b_scaled, le_mask, ge_mask, eq_mask) 
        all_returns = []
        all_obj_change = []
        all_qs0 = []
        all_critic_losses = []
        all_critic1_losses = []
        all_critic2_losses = []
        all_actor_losses = []
        csv_vals = []
        for run in range(N_runs):
                for episode in range(start_ep, end_ep + 1, 200):
                #for alpha in np.linspace(0.0, 1.0, 11):
                    #alpha = round(alpha, 1)

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
                            run        = run,
                            csv_vals   = csv_vals,
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
                    #all_obj_change.append(obj_change_list)  # Storing objective value changes
                    #all_vio_constr.append(vio_constr)
                    #all_vio_bounds.append(vio_bounds)
                    time4 = time.perf_counter()
                    print("\nTraining complete.")
                    #print(f"Total time alpha {alpha}:", datetime.timedelta(seconds=time4-time3))
                    print(f"Total time in {episode} episodes:", datetime.timedelta(seconds=time4-time3))
                    #plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{name_only}_{folder_end}\\critLoss_{end_ep}Ep_mdp3","(Critic1 + Critic2) Loss",all_critic_losses, N, M)
        plot_ma_returns(name_only,f"D:\\experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\actorLoss_{end_ep}Ep_{folder_end}","avg_policyGradNorm",all_actor_losses, N, M, window_size=10)
        plot_ma_returns(name_only,f"D:\\experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\V(S0)_{end_ep}Ep_{folder_end}","V(s0)",all_qs0, N, M, window_size=10)
        plot_ma_returns(name_only,f"D:\\experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\critic1Loss_{end_ep}Ep_{folder_end}","Critic1 Loss",all_critic1_losses, N, M, window_size=10)
        #plot_any_returns(name_only,f"experiment_plots\\Training_Curves\\{category}\\{name_only}_{folder_end}\\critic2Loss_{end_ep}Ep_{folder_end}","Critic2 Loss",all_critic2_losses, N, M)
        plot_ma_returns(name_only,f"D:\\experiment_plots\\Training_Curves\\{category}\\{folder_end}\\{name_only}_{folder_end}\\returns_{end_ep}Ep_{folder_end}","TotReturns",all_returns, N, M, window_size=10)
        #plot_violn(file, all_vio_constr, all_vio_bounds, episode)
        #plot_obj_change(name_only, all_obj_change, episode, N, M)
        print(f"Running the learnt policy to solve the {name_only} finally:**************************************************")
        time5 = time.perf_counter()
        trajectory, total_return, obj_final, x_final, reason, entropy,max_entropy = run_policy_nn(env, actor, device=device, hours= 0.1)   # <<< CHANGE >>>
        time6 = time.perf_counter()
        #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
        print(f"Time taken by {episode} episode_policy in {run}th run :", datetime.timedelta(seconds=time6-time5))
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
        "Uni_Obj": "----",
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

def generate_instance(category, seed):
    if category == "setcover":
        model = generate_setcover(nrows=500, ncols=1000, density=0.05,
                                         max_coef=100, seed=seed)
    elif category == "indset":
        model = generate_indset(n_nodes=500, affinity=4, seed=seed)
    else:
        raise ValueError(f"Unknown category: {category}")

    model.setParam("Threads", 1)
    return model


def run_generated_instance(instance_id, category, seed):
    model = generate_instance(category, seed)
    instance_name = f"{category[0].upper()}_{instance_id:03d}"

    returned_name, results_bundle, log_content = syntheticInst_experiment(
        model, instance_name
    )

    '''for row in results_bundle:
        if isinstance(row, dict):
            row["instance_id"] = instance_id
            row["seed"] = seed
            row["instance_name"] = instance_name'''

    return returned_name, results_bundle, log_content

# ═══════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════

if __name__ == "__main__":
    
    NUM_INSTANCES = 10
    BASE_SEED = 10000

    num_processes = multiprocessing.cpu_count()
    csv_header_written = False

    processed_files = load_checkpoint(CHECKPOINT_FILE)
    

    time_exp_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        with open(master_file_path, "w", encoding="utf-8") as master_file, \
            open(master_csv_path, "w", newline="", encoding="utf-8") as csv_file:

            csv_writer = csv.writer(csv_file)

            future_to_instance = {
                executor.submit(
                    run_generated_instance,
                    instance_id,
                    category,
                    BASE_SEED + instance_id
                ): instance_id
                for instance_id in range(NUM_INSTANCES)
                if instance_id not in processed_files
            }

            for future in concurrent.futures.as_completed(future_to_instance):
                instance_id = future_to_instance[future]

                try:
                    returned_name, results_bundle, log_content = future.result()

                    results.extend(results_bundle)

                    print(f"Successfully completed {returned_name}")

                    master_file.write(f"=== Output for {returned_name} ===\n")
                    master_file.write(log_content)
                    master_file.write("\n\n")
                    master_file.flush()

                    if results_bundle:
                        first_item = results_bundle[0]

                        if isinstance(first_item, dict):
                            keys = list(first_item.keys())

                            if not csv_header_written:
                                csv_writer.writerow(keys)
                                csv_header_written = True

                            for row_dict in results_bundle:
                                csv_writer.writerow([row_dict.get(k, "") for k in keys])

                        elif isinstance(first_item, (list, tuple)):
                            for row in results_bundle:
                                csv_writer.writerow(list(row))

                        csv_file.flush()

                    #processed_files.add(instance_id)
                    #save_checkpoint(CHECKPOINT_FILE, processed_files)

                except Exception as e:
                    print(f"ERROR in instance {instance_id}: {e}")
    
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




  
