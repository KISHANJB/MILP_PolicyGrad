# Functions to get the results after the learning
import json
import os
import sys
import math
import numpy as np
import time
import random
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.rcParams['figure.max_open_warning'] = 0
from IPython.display import display
from IPython.display import clear_output
#clear_output(wait=True)
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch.distributions import Categorical
matplotlib.use('Agg')
#from environ import alpha, beta, gamma
category = sys.argv[3] if len(sys.argv) > 3 else 'smoke_test'
alpha = beta = gamma = 1
from gurobipy_functions import sigma

# ═══════════════════════════════════════════════════════════════════════
# 4. POLICY EXECUTION  (deterministic)
# ═══════════════════════════════════════════════════════════════════════

def run_policy_nn(env, actor, device=None, hours=None):  # <<< CHANGE >>> time_limit in seconds
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device    = torch.device(device)
    NO_SOLUTION = {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}
    entropies = []
    max_entropies = []
    n_actions = env.action_space.n
    state = env.reset()
    from Learn import greedy_action_nn
    #result = env.is_terminal(state)
    #print(f"Is initial state terminal? {result[0]}")
    #state        = np.array(state, dtype=np.float32)
    total_return = 0.0
    trajectory   = []
    if env.quit == True:
        trajectory.append((state, None, 0.0))    # terminal state
        return trajectory, total_return, None, None, "Infeasible Instance", 1, math.log2(env.num_actions)
    if hours:
        time_limit_seconds = hours * 3600   # convert hours to seconds
    start_time = time.perf_counter()
    while True:   
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_limit_seconds:
            print(f"Time limit of {time_limit_seconds}s reached after {len(trajectory)} steps. Policy Stopping.")
            reason = 'TIME_LIMIT'
            break
        state_t = torch.FloatTensor(state).to(device)
        logits = actor(state_t)
        dist = Categorical(logits=logits)

        # 3. Track entropy
        entropies.append(dist.entropy().mean().detach().cpu()/math.log(2))
        valid   = env.compute_feasible_actions(state)
        #max_entropies.append(math.log(len(valid)))
        action  = greedy_action_nn(actor, state_t, valid, n_actions)
        #next_state, reward, obj_val2, x_val2, terminated, truncated, reason = env.step(state, action) # env_TreeEp2
        next_state, reward, obj_val2, x_val2, terminated, truncated, reason = env.step(action)  # For env_mdp3
        #next_state, reward, obj_val2, x_val2, terminated, truncated, reason = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        #loop = terminated or truncated
        trajectory.append((state, action, reward))
        total_return += reward
        state = next_state
        #result = env.is_terminal(state)


        if truncated == True:
            print(f"Policy stopped after {env.n_steps} steps. Stopping.")
            reason = 'MAX_STEPS'
            obj_val2 = None
            x_val2 = None
            break

        elif terminated == True and reason in ['ALL FIXED (Desirable)', 'FEASIBLE (Desirable)']:
            print(f"Policy found an integral solution")
            print(f"Reason: {reason}")
            break

        elif terminated == True and reason in {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}:
            print(f"Policy stuck at an infeasible state")
            break


    
    ''' trajectory.append((state, action, 0.0))    # terminal state
    ret_coll = [trajectory, total_return, result[2], result[3], reason, torch.stack(entropies).mean(), math.log2(env.num_actions)]
    for index, val in enumerate(ret_coll):
        if val is None:
            print(f"❌ Item at index {index} is None")
        elif isinstance(val, (int, float)) and math.isnan(val):
            print(f"❌ Item at index {index} is NaN")
        elif isinstance(val, (list, dict, str, set, tuple)) and not val:
            print(f"❌ Item at index {index} is an empty {type(val).__name__}") '''

    return trajectory, total_return, obj_val2, x_val2, reason, torch.stack(entropies).mean(), math.log2(env.num_actions)




def run_uniform(env, device=None, hours=None):  # <<< CHANGE >>> time_limit in seconds
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device    = torch.device(device)
    NO_SOLUTION = {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}
    entropies = []
    max_entropies = []
    n_actions = env.action_space.n
    state = env.reset()
    from Learn import greedy_action_nn
    #result = env.is_terminal(state)
    #print(f"Is initial state terminal? {result[0]}")
    #state        = np.array(state, dtype=np.float32)
    total_return = 0.0
    trajectory   = []
    if env.quit == True:
        trajectory.append((state, None, 0.0))    # terminal state
        return trajectory, total_return, None, None, "Infeasible Instance", 1, math.log2(env.num_actions)
    if hours:
        time_limit_seconds = hours * 3600   # convert hours to seconds
    start_time = time.perf_counter()
    while True:   
        elapsed = time.perf_counter() - start_time
        if elapsed >= time_limit_seconds:
            print(f"Time limit of {time_limit_seconds}s reached after {len(trajectory)} steps. Policy Stopping.")
            reason = 'TIME_LIMIT'
            break
        #state_t = torch.FloatTensor(state).to(device)
        #logits = actor(state_t)
        #dist = Categorical(logits=logits)

        # 3. Track entropy
        #entropies.append(dist.entropy().mean().detach().cpu()/math.log(2))
        valid   = env.compute_feasible_actions(state)
        #max_entropies.append(math.log(len(valid)))
        action  = random.choice(valid)  # Uniform random action selection
        #next_state, reward, obj_val2, x_val2, terminated, truncated, reason = env.step(state, action) # env_TreeEp2
        next_state, reward, obj_val2, x_val2, terminated, truncated, reason = env.step(action)  # For env_mdp3
        #next_state, reward, obj_val2, x_val2, terminated, truncated, reason = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        #loop = terminated or truncated
        trajectory.append((state, action, reward))
        total_return += reward
        state = next_state
        #result = env.is_terminal(state)


        if truncated == True:
            print(f"Policy stopped after {env.n_steps} steps. Stopping.")
            reason = 'MAX_STEPS'
            obj_val2 = None
            x_val2 = None
            break

        elif terminated == True and reason in ['ALL FIXED (Desirable)', 'FEASIBLE (Desirable)']:
            print(f"Policy found an integral solution")
            print(f"Reason: {reason}")
            break

        elif terminated == True and reason in {'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'}:
            print(f"Policy stuck at an infeasible state")
            break


    return trajectory, total_return, obj_val2, x_val2, reason

#=======================================================================
# Checkpoint functions
#=======================================================================

def load_checkpoint(CHECKPOINT_FILE):
    """Loads the list of successfully processed files."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()
    return set()

'''def save_checkpoint(CHECKPOINT_FILE, processed_set):
    """Saves the list of successfully processed files."""
    # Add encoding="utf-8" here
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_set), f, indent=4)'''


def save_checkpoint(CHECKPOINT_FILE, prob_name):
    # 1. Load existing data if file exists, else start an empty list
    if os.path.exists(CHECKPOINT_FILE) and os.path.getsize(CHECKPOINT_FILE) > 0:
        with open(CHECKPOINT_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
        
    # 2. Append the new checkpoint
    history.append(prob_name)
    
    # 3. Write back the complete updated list
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(history, f, indent=4)
    
# ═══════════════════════════════════════════════════════════════════════
# 5. PLOT & DISPLAY FUNCTIONS  (server-safe)
# ══════════════════════════════════
'''def plot_policy_at_step_nn(file_path,actor, trajectory, step, env, device=None):
    #from Learn import episodes
    filename  = os.path.basename(file_path)         # "2club200v15p5scn.mps"
    prob_name = os.path.splitext(filename)[0] 
    from Learn import  get_policy
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device    = torch.device(device)
    n_actions = env.action_space.n

    state, action, _ = trajectory[step]
    state_t  = torch.FloatTensor(state).to(device)
    valid    = env.compute_feasible_actions(state)

    with torch.no_grad():
        probs = get_policy(actor, state_t, valid, n_actions).cpu().numpy()

    entropy = -np.sum(probs[valid] * np.log(probs[valid] + 1e-10))
    labels  = [str(env.var_id_to_actionTuple[a]) for a in valid]
    values  = probs[valid]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(valid)), values, tick_label=labels)

    if action is not None:
        taken_idx = list(valid).index(action)
        bars[taken_idx].set_color("orange")

    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_title(f"Step {step+1} | Action: {env.var_id_to_actionTuple[action]} | Entropy={entropy:.3f}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #output_dir = f"experiment_plots\\Policy_Curves\\hard_instances\\{sys.argv[0]}ep_a{sys.argv[1]}b{sys.argv[2]}g{sys.argv[3]}s{sys.argv[4]}"  # <<< CHANGE >>> specify your desired folder here
    output_dir = f"experiment_plots\\Policy_Curves\\hard_instances\\cubed\\{episode}ep_a{alpha}b{beta}g{gamma}s{sigma}"  # <<< CHANGE >>> specify your desired folder here
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prob_name}_policyStep_{step+1}.png"

    # 2. Automatically create the folder if it doesn't exist yet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 3. Combine them safely using os.path.join
    full_path = os.path.join(output_dir, filename)
    # This creates a clean path string like: "experiment_plots/training_loss_epoch_50.png"

    # 4. Save directly to that new folder location
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)'''

def plot_splitAxes(prob_name,dir, y_label,all_returns, vars, cons):
    #from Learn import episodes

    all_returns  = np.array(all_returns) 
    mean_returns = np.mean(all_returns, axis=0)
    std_returns  = np.std(all_returns,  axis=0)
    episode     = np.arange(1, all_returns.shape[1] + 1)

    # --- 1. Create Split Subplots ---
    # sharex=True keeps the X-axis aligned across both plots
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 5), 
        gridspec_kw={'height_ratios': [1, 3]}  # Top plot takes 1/4 space, bottom takes 3/4
    )

    # --- 2. Plot Data on Both Subplots ---
    ax_top.plot(episode, mean_returns, marker='o', color='steelblue', linewidth=2)
    ax_bottom.plot(episode, mean_returns, marker='o', color='steelblue', linewidth=2)

    # --- 3. Calculate Dynamic Y-Limits ---
    # Normal range using percentiles (ignores top outliers)
    y_bottom_min = np.min(mean_returns)
    y_bottom_max = np.percentile(mean_returns, 95)
    margin_bottom = (y_bottom_max - y_bottom_min) * 0.1
    ax_bottom.set_ylim(y_bottom_min - margin_bottom, y_bottom_max + margin_bottom)

    # Spike range (covers maximum outlier values)
    y_top_max = np.max(mean_returns)
    margin_top = (y_top_max - y_bottom_max) * 0.05
    ax_top.set_ylim(y_bottom_max + margin_bottom, y_top_max + margin_top)

    # --- 4. Hide Intersecting Spines ---
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)

    # Remove top subplot's X-ticks so they don't overlap
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # --- 5. Add Diagonal Break Slashes ---
    d = 0.015  # Size of break marks

    # Top plot break marks
    kwargs_top = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs_top)        # bottom-left slash
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)  # bottom-right slash

    # Bottom plot break marks
    kwargs_bottom = dict(transform=ax_bottom.transAxes, color='k', clip_on=False, linewidth=1)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)        # top-left slash
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)  # top-right slash

    # --- 6. Labels and Formatting ---
    ax_bottom.set_xlabel("Episode")
    fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')  # Shared Y-label centered

    # Format tick styles and grid
    ax_top.ticklabel_format(style='plain', useOffset=False, axis='y')
    ax_bottom.ticklabel_format(style='plain', useOffset=False, axis='y')
    ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_top.grid(True, alpha=0.3)
    ax_bottom.grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.08)
    #output_dir = f"experiment_plots\\Training_Curves\\hard_instances\\{sys.argv[0]}ep_a{sys.argv[1]}b{sys.argv[2]}g{sys.argv[3]}s{sys.argv[4]}"  # <<< CHANGE >>> specify your desired folder here
    #output_dir = f"experiment_plots\\Training_Curves\\hard_instances\\{episodes}ep_a{alpha}b{beta}g{gamma}s{sigma}"  # <<< CHANGE >>> specify your desired folder here
    #output_dir = f"experiment_plots\\Training_Curves\\{category}\\{episodes}Ep_mdp3"  # <<< CHANGE >>> specify your desired folder here

    os.makedirs(dir, exist_ok=True)
    filename = f"{prob_name}.png"


    # 3. Combine them safely using os.path.join
    full_path = os.path.join(dir, filename)
    # This creates a clean path string like: "experiment_plots/training_loss_epoch_50.png"

    # 4. Save directly to that new folder location
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_any_returns(prob_name,dir,y_label,all_returns, vars, cons):
    #from Learn import episodes

    all_returns  = np.array(all_returns) 
    #all_returns = all_returns.cpu().detach().numpy()  # Convert tensor to numpy array if needed
    mean_returns = np.mean(all_returns, axis=0)
    std_returns  = np.std(all_returns,  axis=0)
    episode     = np.arange(1, all_returns.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episode, mean_returns, color='steelblue',
            linewidth=2)
    '''ax.plot(episode, color='steelblue',
            linewidth=2)'''

    '''ax.fill_between(episode,
                    mean_returns - std_returns,
                    mean_returns + std_returns,
                    color='steelblue', alpha=0.2,
                    label='± 1 std dev')'''

    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    #ax.set_title(f"Average Return across {N_runs} Runs")
    ax.set_title(f" #Vars = {vars}, #Cons = {cons}")
    plt.ticklabel_format(style='plain', useOffset=False, axis='y')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.get_yaxis().get_major_formatter().set_useoffset(False)
    #ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    #output_dir = f"experiment_plots\\Training_Curves\\hard_instances\\{sys.argv[0]}ep_a{sys.argv[1]}b{sys.argv[2]}g{sys.argv[3]}s{sys.argv[4]}"  # <<< CHANGE >>> specify your desired folder here
    #output_dir = f"experiment_plots\\Training_Curves\\hard_instances\\{episodes}ep_a{alpha}b{beta}g{gamma}s{sigma}"  # <<< CHANGE >>> specify your desired folder here
    #output_dir = f"experiment_plots\\Training_Curves\\{category}\\{episodes}Ep_mdp3"  # <<< CHANGE >>> specify your desired folder here

    os.makedirs(dir, exist_ok=True)
    filename = f"{prob_name}.png"



    # 3. Combine them safely using os.path.join
    full_path = os.path.join(dir, filename)
    # This creates a clean path string like: "experiment_plots/training_loss_epoch_50.png"

    # 4. Save directly to that new folder location
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

from matplotlib.ticker import MaxNLocator

def plot_ma_returns(prob_name, dir, y_label, all_returns, vars, cons, window_size=10):
    """
    Plots the average returns across episodes applying a moving average filter.
    
    Parameters:
    - window_size (int): The size of the rolling window for the moving average.
    """
    all_returns = np.array(all_returns) 
    print (f"All returns shape: {all_returns.shape}")
    #print (f"\nAll returns sample: {all_returns}")
    # Calculate raw mean across runs/experiments
    raw_mean_returns = np.mean(all_returns, axis=0)
    
    # Apply moving average using linear convolution
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        # mode='valid' prevents edge distortion but shrinks the array length by (window_size - 1)
        mean_returns = np.convolve(raw_mean_returns, kernel, mode='valid')
        
        # Adjust the episode X-axis to match the shortened 'valid' array length
        start_ep = window_size
        episode = np.arange(start_ep, all_returns.shape[1] + 1)
    else:
        # Fallback to default behavior if window is 1 or less
        mean_returns = raw_mean_returns
        episode = np.arange(1, all_returns.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the smoothed moving average line
    ax.plot(episode, mean_returns, color='steelblue', linewidth=2, label=f'Moving Avg (w={window_size})')

    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(f" #Vars = {vars}, #Cons = {cons}")
    
    plt.ticklabel_format(style='plain', useOffset=False, axis='y')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    os.makedirs(dir, exist_ok=True)
    filename = f"{prob_name}.png"
    full_path = os.path.join(dir, filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_obj_change(prob_name, all_obj_change, episodes, vars, cons):
    #from Learn import episodes

    all_obj_change = np.array(all_obj_change)
    mean_obj_change = np.mean(all_obj_change, axis=0)
    #std_returns  = np.std(all_returns,  axis=0)
    episode     = np.arange(1, all_obj_change.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episode, mean_obj_change, color='steelblue',
            linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total_Obj_Change")
    #ax.set_title(f"Average Return across {N_runs} Runs")
    ax.set_title(f" #Vars = {vars}, #Cons = {cons}")
    plt.ticklabel_format(style='plain', useOffset=False, axis='y')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.get_yaxis().get_major_formatter().set_useoffset(False)
    #ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    #output_dir = f"experiment_plots\\Obj_Change\\hard_instances\\{sys.argv[0]}ep_a{sys.argv[1]}b{sys.argv[2]}g{sys.argv[3]}s{sys.argv[4]}"  # <<< CHANGE >>> specify your desired folder here
    #output_dir = f"experiment_plots\\Obj_Change\\hard_instances\\{episodes}ep_a{alpha}b{beta}g{gamma}s{sigma}"  # <<< CHANGE >>> specify your desired folder here
    output_dir = f"experiment_plots\\Obj_Change\\{category}\\{episodes}Ep_mdp3"  # <<< CHANGE >>> specify your desired folder here

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prob_name}.png"

    '''# 2. Automatically create the folder if it doesn't exist yet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)'''

    # 3. Combine them safely using os.path.join
    full_path = os.path.join(output_dir, filename)
    # This creates a clean path string like: "experiment_plots/training_loss_epoch_50.png"

    # 4. Save directly to that new folder location
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

'''def plot_violn(file_path, all_returns1, all_returns2, episode, label1="#Constr_violn", label2="#Bounds_violn"):
    #from Learn import episodes
    filename  = os.path.basename(file_path)         
    prob_name = os.path.splitext(filename)[0] 
    
    # Process Dataset 1
    all_returns1 = np.array(all_returns1) 
    mean_returns1 = np.mean(all_returns1, axis=0)
    std_returns1  = np.std(all_returns1,  axis=0)
    episodes1     = np.arange(1, all_returns1.shape[1] + 1)

    # Process Dataset 2
    all_returns2 = np.array(all_returns2) 
    mean_returns2 = np.mean(all_returns2, axis=0)
    std_returns2  = np.std(all_returns2,  axis=0)
    episodes2     = np.arange(1, all_returns2.shape[1] + 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot Dataset 1 (Steel Blue)
    ax.plot(episodes1, mean_returns1, color='steelblue',
            linewidth=2, label=f'{label1} Mean in ({episode} episodes)')
    ax.fill_between(episodes1,
                    mean_returns1 - std_returns1,
                    mean_returns1 + std_returns1,
                    color='steelblue', alpha=0.15,
                    label=f'{label1} ± 1 std dev')
    
    # Plot Dataset 2 (Dark Orange)
    ax.plot(episodes2, mean_returns2, color='darkorange',
            linewidth=2, label=f'{label2} Mean ({episode} runs)')
    ax.fill_between(episodes2,
                    mean_returns2 - std_returns2,
                    mean_returns2 + std_returns2,
                    color='darkorange', alpha=0.15,
                    label=f'{label2} ± 1 std dev')

    # Formatting
    ax.set_xlabel("Episode")
    ax.set_ylabel("#Violations")
    #ax.set_title(f"Average violations across {N_runs} Runs")
    ax.set_title(f" Violations in {episode} Episodes")

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save directory setup
    #output_dir1 = os.path.join("experiment_plots", "Violation_Curves", f"{sys.argv[0]}ep_a{sys.argv[1]}b{sys.argv[2]}g{sys.argv[3]}s{sys.argv[4]}")
    #output_dir1 = os.path.join("experiment_plots", "Violation_Curves", "Hard_instances", "cubed", f"{episodes}ep_a{alpha}b{beta}g{gamma}s{sigma}")
    output_dir1 = os.path.join("experiment_plots", "Violation_Curves", "smoke_test", f"{episode}ep_a{alpha}b{beta}g{gamma}s{sigma}")


    os.makedirs(output_dir1, exist_ok=True)
    
    # Save the file
    filename = f"{prob_name}_violations.png"
    full_path = os.path.join(output_dir1, filename)
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)'''

def format_state(state):
    N           = len(state) // 2
    first_half  = state[:N]
    second_half = state[N:]
    line1 = "  ".join(f"{int(v)}" for v in first_half)
    line2 = "  ".join(f"{int(v)}" for v in second_half)
    return f"{line1}\n{line2}"

def print_trajectory(trajectory, env):
    # Compute return-to-go
    returns_to_go = []
    cumulative    = 0.0
    for _, _, reward in reversed(trajectory):
        cumulative += reward
        returns_to_go.insert(0, cumulative)

    rows = []
    for i, (state, action, reward) in enumerate(trajectory):
        action_str = str(env.var_id_to_actionTuple[action]) if action is not None else "TERMINAL"
        rows.append({
            "Step"             : i + 1,
            #"State"            : format_state(state),
            "Action"           : action_str,
            "Return from here" : f"{returns_to_go[i]:.3f}",
        })

    df = pd.DataFrame(rows).set_index("Step")

    pd.set_option("display.max_colwidth", None)
    styled = (
        df.style
        .set_properties(**{
            "white-space" : "pre",
            "text-align"  : "left",
            "font-family" : "monospace",
            "font-size"   : "13px",
            "padding"     : "6px 12px",
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
    #display(styled)
    return df

