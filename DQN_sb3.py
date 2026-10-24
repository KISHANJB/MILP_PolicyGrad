
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import sys

from networkx import display
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import multiprocessing
import concurrent.futures
import os
from random import random
import datetime
import time
import gurobipy as gp
import pandas as pd
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from IPython.display import display
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import torch
import random
import math
# ============================================================
# 1. USER CONFIGURATION
# ============================================================

# <<< FILL IN: import your existing Gymnasium MDP class >>>
from environ_mdp3_dqn import MyMDPEnv
from gurobipy_functions import extract_sparse_gurobipy_matrices, solve_bip_gurobiws
from main2 import generate_instance, run_generated_instance
from inst_gen import generate_indset, generate_setcover
from Test_Display import load_checkpoint, save_checkpoint
# Choose: "DQN" or "SAC"

TOTAL_TIMESTEPS = 1_000
LEARNING_RATE = 1e-4
BUFFER_SIZE = 500
BATCH_SIZE = 32
LEARNING_STARTS = 100
GAMMA = 1
HIDDEN_DIM = 128
DEVICE = "cuda"       # Use "cuda" if available
SEED = 42

csv_vals = []
RUN_ALGORITHM = "DQN"
CHECKPOINT_FILE = "processed_files.json"
category = sys.argv[1] if len(sys.argv) > 1 else "setcover"  # Default to "setcover" if no argument is provided
SAVE_PATH = f"D:\\policy_files\\{RUN_ALGORITHM}\\{category}\\"
# DQN supports Discrete actions; SAC supports Box actions.
# Do not change the MDP action space just to satisfy an algorithm.

'''
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE = 1e-4
BUFFER_SIZE = 10_000
BATCH_SIZE = 128
LEARNING_STARTS = 2_000
GAMMA = 1
HIDDEN_DIM = 128
DEVICE = "cuda"       # Use "cuda" if available
SEED = 42 '''





class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

log_file = open(f"D:\\Logs & CSVs\\logs\\{category}_DQN_stPen_Mdp3.txt", "w", encoding="utf-8",buffering=1)
log_buffer = Tee(sys.stdout, log_file)
master_csv_path = f"D:\\Logs & CSVs\\csv\\{category}_DQN_stPen_Mdp3.csv"

# ============================================================
# 2. OPTIONAL ACTION-SAFETY WRAPPER
# ============================================================

class ActionSafetyWrapper(gym.Wrapper):
    """
    Prevents execution of invalid actions by redirecting them
    to a random feasible action.

    IMPORTANT:
    This is an execution safeguard, NOT genuine DQN action
    masking during exploration and Bellman-target calculation.
    """

    def __init__(self, env):
        super().__init__(env)
        #self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs).copy()
        return obs, info

    def step(self, action):
        # <<< FILL IN if your method uses a different signature >>>
        valid = self.env.compute_feasible_actions(self.env.state)

        valid = np.asarray(valid, dtype=int).reshape(-1)

        if len(valid) == 0:
            raise RuntimeError(
                "No feasible actions in a nonterminal state. "
                "Check your MDP termination and feasibility logic."
            )
        #action_tuple = self.actionTuple_to_id[int(action)]
        #action_int = int(np.asarray(action_tuple).item())

        if action not in valid:
            #action = self.env.np_random.choice(valid)
            action = int(np.random.choice(valid))
        #action_tuple = self.var_id_to_actionTuple[int(action)]
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        #self._last_obs = np.asarray(obs).copy()

        return obs, reward, terminated, truncated, info 


# ============================================================
# 3. EPISODE-RETURN LOGGER
# ============================================================

class RewardLoggerCallback(BaseCallback):
    """Records the cumulative reward of each completed episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_rewards = None

    def _on_training_start(self):
        # One accumulator per environment.
        self._current_rewards = np.zeros(
            self.training_env.num_envs, dtype=float
        )

    def _on_step(self):
        rewards = np.asarray(self.locals["rewards"]).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)

        self._current_rewards += rewards

        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(
                    float(self._current_rewards[i])
                )
                self._current_rewards[i] = 0.0

        return True


# ============================================================
# 4. ENVIRONMENT SETUP
# ============================================================

def make_env(name_only, gmodel=None):
    # <<< FILL IN constructor arguments required by your MDP >>>
        '''gp_env = gp.Env(empty=True)
        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("FeasibilityTol", 1e-9)
        gp_env.start()'''
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
        tc, rel_obj_val, rel_x_val, _, _ = solve_bip_gurobiws(gmodel, vec=None, relax=False, warm_start=None, vbasis=None, cbasis=None)
        time2 = time.perf_counter()
        print(f"Time taken by solver:", datetime.timedelta(seconds=time2-time1))
        #print("Number of binary variables:", N)
        #print("Number of constraints:", M)
        print("Solution of the Relaxation:------------")
        #print(f"\nX: {np.round(rel_x_val,2)}, Obj = {np.round(rel_obj_val,2)}")
        print(f"\nObj = {np.round(rel_obj_val,2)}")

        time3 = time.perf_counter()
        env = MyMDPEnv(N, M, gmodel,A_scaled, b_scaled, le_mask, ge_mask, eq_mask) 
        env.initial_config()

    # Keep your existing reward and transition implementation.
        return env, rel_obj_val, rel_x_val


def validate_env(env):
    check_env(env, warn=True)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)


# ============================================================
# 5. TRAIN DQN OR SAC
# ============================================================

def train_agent(ins_name=None, gmodel=None):
    base_env, rel_obj_val, rel_x_val = make_env(ins_name, gmodel)
    validate_env(base_env)

    if RUN_ALGORITHM not in ("DQN", "SAC"):
        raise ValueError("RUN_ALGORITHM must be 'DQN' or 'SAC'.")

    # Enforce algorithm/action-space compatibility.
    if RUN_ALGORITHM == "DQN":
        if not isinstance(base_env.action_space, gym.spaces.Discrete):
            raise TypeError(
                "SB3 DQN requires gymnasium.spaces.Discrete."
            )

        if not hasattr(base_env, "compute_feasible_actions"):
            raise AttributeError(
                "Your MDP must implement compute_feasible_actions(state) "
                "if you use the action-safety wrapper."
            )

    elif RUN_ALGORITHM == "SAC":
        if not isinstance(base_env.action_space, gym.spaces.Box):
            raise TypeError(
                "Standard SB3 SAC requires a continuous Box action space. "
                "It cannot train directly on your discrete branching actions. "
                "Use a compatible discrete-SAC implementation or a "
                "different SB3 algorithm."
            )

        print(
            "SAC selected: this script does not implement "
            "state-dependent action masking for continuous actions."
        )

    # Apply the wrapper only to DQN in this example.

    env = base_env
    env = ActionSafetyWrapper(env)
    env = Monitor(env)
    start_time = time.perf_counter()
    # These configure SB3's networks. They do not reuse custom
    # ActorNet/CriticNet classes automatically.
    if RUN_ALGORITHM == "DQN":
        policy_kwargs = dict(
            net_arch=[HIDDEN_DIM, HIDDEN_DIM]
        )

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=10_000,
            policy_kwargs=policy_kwargs,
            seed=SEED,
            device=DEVICE,
            verbose=0,
        )


    else:
        policy_kwargs = dict(
            net_arch=dict(
                pi=[HIDDEN_DIM, HIDDEN_DIM],
                qf=[HIDDEN_DIM, HIDDEN_DIM],
            )
        )

        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            seed=SEED,
            device=DEVICE,
            verbose=0,
        )

    print("\nSB3 policy architecture:")
    print(model.policy)

    reward_callback = RewardLoggerCallback()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=reward_callback,
        progress_bar= False,
    )
    end_time = time.perf_counter()
    directory = os.path.dirname(SAVE_PATH)
    os.makedirs(directory, exist_ok=True)
    filename = f"{ins_name}_{RUN_ALGORITHM}_pol"
    full_path = os.path.join(directory, filename)
    model.save(full_path)
    print(f"Saved model: {full_path}.zip")
    env = env.unwrapped
    print("\nRunning the trained policy deterministically on the environment...")
    trajectory, total_return, pol_obj, pol_x, elapsed = run_sb3_policy(env, model, time_limit_hours=1)
    uni_trajectory, uni_total_return, uni_obj, uni_x, reason = run_uniform_policy(env, hours=None)
    #print_trajectory(trajectory, env)
    csv_vals.append({
    "Problem": ins_name,
    "#Vars": env.num_vars,
    "#Cons": env.num_constraints,
    #"#Episodes":"----" ,
    "rel_Obj": rel_obj_val,
    "Uni_Obj": uni_obj,
    "Pol_Obj": pol_obj,
    #"Mean_Rwd": "----",
    #"V(S0)": "----",
    "#Success":env.success_count,
    #"Vio_Constr": "----",
    #"Vio_Bounds": "----",
    #"Obj_Change": "----",
    #"AvgEp_Time": "----",
    #"Ent": "----",
    #"Max_Ent": "----",
    #"critic1_loss": "----",
    #"critic2_loss": "----",
    #"actor_loss": "----",
    "Train_Time": end_time - start_time,
    "Policy_Time": elapsed})    
    env.close()
    return model, reward_callback.episode_rewards, csv_vals


def format_state(state):
    N           = len(state) // 2
    line1 = "  ".join(f"{int(v)}" for v in state[:N])
    line2 = "  ".join(f"{int(v)}" for v in state[N:])
    return f"{line1}\n{line2}"


def print_trajectory(trajectory, env):
    returns_to_go = []
    cumulative    = 0.0
    for _, _, reward in reversed(trajectory):
        cumulative += reward
        returns_to_go.insert(0, cumulative)

    rows = []
    for i, (state, action, reward) in enumerate(trajectory):
        action_str = (str(env.var_id_to_actionTuple[action])
                      if action is not None else "TERMINAL")
        rows.append({
            "Step"             : i + 1,
            "State"            : format_state(state),
            "Action"           : action_str,
            "Return from here" : f"{returns_to_go[i]:.3f}",
        })

    df = pd.DataFrame(rows).set_index("Step")
    pd.set_option("display.max_colwidth", None)
    styled = (
        df.style
        .set_properties(**{
            "white-space": "pre",
            "text-align" : "left",
            "font-family": "monospace",
            "font-size"  : "13px",
            "padding"    : "6px 12px",
        })
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#4A90D9"),
                ("color", "white"),
                ("font-size", "13px"),
                ("padding", "6px 12px"),
            ]
        }])
    )
    display(styled)

def run_sb3_policy(env, model, time_limit_hours=1):    # <<< CHANGE >>>
    """
    Runs the learnt SB3 policy deterministically.
    Works for both DQN and SAC models.
    """
    import time
    state, _           = env.reset()
    state              = np.array(state, dtype=np.float32)
    total_return       = 0.0
    trajectory         = []
    time_limit_seconds = time_limit_hours * 3600
    start_time         = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_limit_seconds:
            print(f"Time limit of {time_limit_hours}h reached "
                  f"after {len(trajectory)} steps.")
            break

        # SB3 predict — deterministic=True for greedy execution
        action, _ = model.predict(state, deterministic=True)
        action    = int(action)

        # Enforce action masking
        valid = env.compute_feasible_actions(state)
        if action not in valid:
            action = int(np.random.choice(valid))

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state    = np.array(next_state, dtype=np.float32)
        total_return += reward

        trajectory.append((state, action, reward))
        state = next_state
        NO_SOLUTION = {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}

        if terminated or truncated:
            tc, obj, x, _, _ = solve_bip_gurobiws(env.gmodel, vec = None, relax = False, warm_start= None, vbasis= None, cbasis= None)
            if tc in NO_SOLUTION:
                print(f"Terminal state reached. No feasible solution found.")
                pol_obj = None
                pol_x = None
            else:
                print(f"Feasible (Desirable) state reached. Objective value: {obj:.3f}")
                pol_obj = obj
                pol_x = x
            break

    trajectory.append((state, None, 0.0))

    elapsed = time.perf_counter() - start_time
    print(f"Policy execution time: {elapsed:.4f} sec | "
          f"Steps: {len(trajectory)-1} | "
          f"Total return: {total_return:.3f}")

    return trajectory, total_return, pol_obj, pol_x, elapsed

# ============================================================
# 6. PLOT EPISODE RETURNS
# ============================================================

def plot_rewards(inst_name,episode_rewards, save_path=None, window=100,
                 show_moving_average=True):


    rewards = np.asarray(episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.4, label="Episode reward")

    if show_moving_average and len(rewards) > 0:
        window = max(1, min(window, len(rewards)))
        moving_avg = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            episodes[window - 1:],
            moving_avg,
            label=f"Moving average ({window} Ep)"
        )

    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f" Ts: {TOTAL_TIMESTEPS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            filename = f"{inst_name}_Tr2_{TOTAL_TIMESTEPS}Ts.png"
            full_path = os.path.join(directory, filename)

        plt.savefig(full_path, dpi=300, bbox_inches="tight")

    #plt.show()
    plt.close()

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
    with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
        model = generate_instance(category, seed)
        instance_name = f"{category[0].upper()}_{instance_id:03d}"
        print(f"\n\n====================================== Running instance {instance_name} ===============================================")
        model, episode_rewards, csv_vals = train_agent(
            instance_name, model
        )
        plot_rewards(instance_name, episode_rewards, f"D:\\experiment_plots\\Training_Curves\\{RUN_ALGORITHM}\\{category}\\total_rewards", 
                    window=100, show_moving_average=True)
        save_checkpoint(CHECKPOINT_FILE, instance_id)
        return episode_rewards, csv_vals


def run_miplib_instance(file):
        gp_env = gp.Env(empty=True)
        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("FeasibilityTol", 1e-9)
        gp_env.start()

        #from Learn import episodes
        filename  = os.path.basename(file)         # "2club200v15p5scn.mps"
        name_only = os.path.splitext(filename)[0] 
        print(f"\n\n{'#'*10}\nProcessing file: {filename}\n{'#'*10}")

        gmodel = gp.read(file, gp_env)
        
        print(f"\n\n====================================== Training on instance {name_only} ===============================================")
        model, episode_rewards, csv_vals = train_agent(
            name_only, gmodel
        )
        plot_rewards(name_only, episode_rewards, f"D:\\experiment_plots\\Training_Curves\\{RUN_ALGORITHM}\\{category}\\total_rewards", 
                    window=100, show_moving_average=True)
        save_checkpoint(CHECKPOINT_FILE, name_only)
        return episode_rewards, csv_vals



def run_uniform_policy(env, hours=None):
    """
    Run a uniform random policy until a terminal state,
    truncation, or time limit.

    Returns:
        trajectory, total_return, obj_val, x_val, reason
    """

    NO_SOLUTION = {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}
    DESIRABLE = {'ALL FIXED (Desirable)', 'FEASIBLE (Desirable)'}

    trajectory = []
    total_return = 0.0
    obj_val = None
    x_val = None
    reason = None

    # Initialize the environment
    state, info = env.reset()

    if getattr(env, "quit", False):
        return trajectory, total_return, obj_val, x_val, "Infeasible Instance"

    # Optional time limit
    start_time = time.perf_counter()
    time_limit_seconds = hours * 3600 if hours is not None else None

    while True:

        # Check time limit
        if (time_limit_seconds is not None and
                time.perf_counter() - start_time >= time_limit_seconds):
            reason = "TIME_LIMIT"
            break

        # Check whether the current state is terminal, if supported
        if hasattr(env, "is_terminal"):
            terminal_result = env.is_terminal(state)

            # Assumes is_terminal returns (terminal_flag, reason)
            if isinstance(terminal_result, tuple) and terminal_result[0]:
                reason = terminal_result[1]
                break

        # Get feasible actions in the current state
        valid_actions = env.compute_feasible_actions(state)

        if len(valid_actions) == 0:
            reason = "NO_FEASIBLE_ACTIONS"
            break

        # Uniform random selection among feasible actions
        action = random.choice(valid_actions)

        # Take one environment step
        next_state, reward, terminated, truncated, info = env.step(action)

        trajectory.append((state, action, reward))
        total_return += float(reward)

        state = np.asarray(next_state, dtype=np.float32)

        # Extract information returned by the environment
        if isinstance(info, dict):
            reason = info.get("reason", reason)
            obj_val = info.get("obj_val", obj_val)
            x_val = info.get("x_val", x_val)

        # Stop at episode termination
        if terminated:
            if reason in DESIRABLE:
                print("Uniform policy found an integral solution.")
                print(f"Reason: {reason}")
            elif reason in NO_SOLUTION:
                print("Uniform policy terminated without a solution.")
                print(f"Reason: {reason}")

            break

        if truncated:
            #reason = reason or "TRUNCATED"
            print(f"Uniform policy stopped: {reason}")
            break

    return trajectory, total_return, obj_val, x_val, reason

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


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    NUM_INSTANCES = 1
    BASE_SEED = 10000

    set_seed(BASE_SEED)

    num_processes = multiprocessing.cpu_count()
    csv_header_written = False

    processed_files = load_checkpoint(CHECKPOINT_FILE)
    folder_path = rf"C:\Users\KISHAN\Downloads\{category}"
    all_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    time_exp_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            with open(master_csv_path, "w", newline="", encoding="utf-8") as csv_file:
                    csv_writer = csv.writer(csv_file)

                    '''future_to_instance = {
                        executor.submit(
                            run_generated_instance,
                            instance_id,
                            category,
                            BASE_SEED + instance_id
                        ): instance_id
                        for instance_id in range(NUM_INSTANCES)
                        if instance_id not in processed_files
                    }'''
                    future_to_instance = {
                            executor.submit(run_miplib_instance, os.path.join(folder_path, file_name)): file_name 
                            for file_name in all_files if file_name not in processed_files 
                    }  
                    for future, instance_id in future_to_instance.items():
                        #try:
                            episode_rewards, csv_vals = future.result()

                            if csv_vals:
                                first_item = csv_vals[0]

                                if isinstance(first_item, dict):
                                    keys = list(first_item.keys())

                                    if not csv_header_written:
                                        csv_writer.writerow(keys)
                                        csv_header_written = True

                                    for row_dict in csv_vals:
                                        csv_writer.writerow([row_dict.get(k, "") for k in keys])

                                elif isinstance(first_item, (list, tuple)):
                                    for row in csv_vals:
                                        csv_writer.writerow(list(row))

                                csv_file.flush()
                            print(f"Instance {instance_id} completed successfully.")
                        #except Exception as exc:
                            print(f"Instance {instance_id} failed with error")


    
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

