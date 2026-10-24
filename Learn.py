import os
import sys
import numpy as np
import time
import datetime
import matplotlib
import random
from collections import defaultdict
matplotlib.rcParams['figure.max_open_warning'] = 0
from IPython.display import display
from IPython.display import clear_output
clear_output(wait=True)
import torch
import torch.optim as optim
import numpy as np
from Test_Display import run_policy_nn, run_uniform
from gurobipy_functions import solve_bip_gurobiws
#from environ_mdp3 import CriticNet, ActorNet
#from environ_TreeEpisode import CriticNet, ActorNet
from environ_mdp3 import CriticNet, ActorNet
learning_hours = float(sys.argv[4]) if len(sys.argv) > 4 else 1
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# ═══════════════════════════════════════════════════════════════════════
# 1. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def make_hashable(item):
    """Recursively converts nested lists to tuples to guarantee hashability."""
    if isinstance(item, list):
        return tuple(make_hashable(x) for x in item)
    return item


def truncate_vector_str(vec, max_items=10):
    """
    Formats a vector as a string, truncating middle elements if it exceeds max_items.
    Example: [1, 2, 3, 4, 5, 6, 7, 8] -> '[1, 2, 3, ..., 6, 7, 8]'
    """
    if len(vec) <= max_items:
        return str(vec)
    
    half = max_items // 2
    head = ", ".join(map(str, vec[:half]))
    tail = ", ".join(map(str, vec[-half:]))
    return f"[{head}, ..., {tail}]"

def print_vector_table(seen_vectors, max_vec_items=10, col_width=35):
    """
    Prints all collected unique vector-value pairs in a formatted table.
    Truncates large vectors to fit within col_width.
    """
    header = f"{'Vector':<{col_width}} | {'Value':<12}"
    divider = "-" * len(header)
    
    print(header)
    print(divider)
    
    for vec, val in seen_vectors.values():
        vec_str = truncate_vector_str(vec, max_items=max_vec_items)
        print(f"{vec_str:<{col_width}} | {str(val):<12}")




def mask_invalid(logits, valid_actions, n_actions):
    valid_tensor = torch.tensor(valid_actions, dtype=torch.long, device=logits.device)  # ← convert to tensor
    mask = torch.full((n_actions,), float('-inf'), device=logits.device)
    mask[valid_tensor] = logits[valid_tensor]
    return mask


def get_policy(actor, state_t, valid_actions, n_actions):
    """Returns masked softmax probability distribution."""
    logits = actor(state_t)
    masked = mask_invalid(logits, valid_actions, n_actions)
    return torch.softmax(masked, dim=0)



def sample_action_nn(actor, state_t, valid_actions, n_actions):
    with torch.no_grad():
        probs = get_policy(actor, state_t, valid_actions, n_actions)
    #print(f"  valid_actions : {valid_actions}")
    #print(f"  probs         : {probs.cpu().numpy()}")
    action = int(torch.multinomial(probs, 1).item())
    #print(f"  sampled action Id: {action}")
    #print(f"  sampled action   : {env.var_id_to_actionTuple[action]}")
    return action


def epsilon_sample_action_nn(actor,
                     state_t,
                     valid_actions,
                     n_actions,
                     epsilon):

    if random.random() < epsilon:

        # Uniform exploration
        return random.choice(valid_actions)

    with torch.no_grad():
        probs = get_policy(
            actor,
            state_t,
            valid_actions,
            n_actions
        )

    return int(torch.multinomial(probs, 1).item())

def greedy_action_nn(actor, state_t, valid_actions, n_actions):
    with torch.no_grad():
        probs = get_policy(actor, state_t, valid_actions, n_actions)
    #print(f"  valid_actions : {valid_actions}")
    #print(f"  probs         : {probs.cpu().numpy()}")
    action = int(torch.argmax(probs).item())
    #print(f"  greedy action Id: {action}")
    #print(f"  greedy action   : {env.var_id_to_actionTuple[action]}")
    return action


def polyak_update_nn(critic, critic_target, tau=0.005):
    """Soft Polyak update: θ_target ← (1-τ)θ_target + τθ"""
    with torch.no_grad():
        for p, p_tgt in zip(critic.parameters(), critic_target.parameters()):
            p_tgt.data.copy_((1 - tau) * p_tgt.data + tau * p.data)



# ═══════════════════════════════════════════════════════════════════════
# 3. MAIN ACTOR-CRITIC LOOP
# ═══════════════════════════════════════════════════════════════════════

def actor_critic_nn(
    vars,
    cons,
    prob,
    env,
    alpha,
    Time_Limit,
    n_episodes,
    gamma: float       = 1.0,         # Issue 1 — no discounting for episodic tasks
    alpha_w: float      = 1e-3,
    alpha_theta: float  = 1e-5,        # Issue 7 — changed from 1e-4
    tau: float          = 0.005,
    hidden_dim: int     = 128,
    device: str         = None,
    entropy_coeff_start: float = 0.9,
    entropy_coeff_end: float   = 0.01,
    #epsilon: float = 1.0
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    device = torch.device(device)
    n_state   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    critic        = CriticNet(n_state, n_actions, hidden_dim).to(device)
    critic_target = CriticNet(n_state, n_actions, hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor = ActorNet(n_state, n_actions, hidden_dim).to(device)

    critic_opt = optim.Adam(critic.parameters(), lr=alpha_w)
    actor_opt  = optim.Adam(actor.parameters(),  lr=alpha_theta)
    #critic_opt = optim.SGD(critic.parameters(), lr=alpha_w)
    #actor_opt  = optim.SGD(actor.parameters(),  lr=alpha_theta)
    # Instead of: torch.optim.SGD(model.parameters(), lr=0.001)
    #torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)   
    #env.alpha = alpha
    success_count = 0
    success_obj = []
    #successful_trajectories = []
    csv_vals = []
    episode_returns = [0]*n_episodes       # Storing episodic returns for each episode
    vio_constr = [0]*n_episodes
    obj_change_list = [0]*n_episodes
    vio_bounds = [0]*n_episodes
    ep_times = [0]*n_episodes
    qs0 = [0]*n_episodes
    critic_losses = [0]*n_episodes
    actor_losses = [0]*n_episodes
    start_time = time.perf_counter()
    decay_rate = 0.01
    env.initial_config()
    for episode in range(n_episodes):
        time1 = time.perf_counter()
        print(f"Episode {episode}/{n_episodes-1}")
        fraction      = episode / max(1, n_episodes - 1)
        entropy_coeff = entropy_coeff_start + fraction * (entropy_coeff_end - entropy_coeff_start)
        #epsilon = max(0.1, epsilon * (1 - fraction))  # Decay epsilon over episodes, but not below 0.1
        epsilon = max(0.1, 1.0/(1 + decay_rate * episode))  # Decay epsilon over episodes, but not below 0.1
        print(f"\nEpsilon for exploration: {epsilon:.4f} | Entropy coefficient: {entropy_coeff:.4f}")
    
        state = env.reset()
        #tc, obj, x, _, _  = solve_bip_gurobiws(env.gmodel,vec=None,relax=False,warm_start=None,vbasis=None,cbasis=None)
        I            = 1.0
        total_return = 0.0
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_obj_change = 0.0
        obj_change_coll = []
        step_reward = []

        trajectory   = []          # Issue 4 — initialised here
        if episode == 0:
            if env.quit == True:
                trajectory.append((state, None, 0.0))
                episode_returns.append(total_return)
                obj_change_list.append(obj_change)
                qs0.append(None)
                critic_losses.append(None)
                actor_losses.append(None)
                csv_vals.append({
                "Problem": prob,
                "#Vars": vars,
                "#Cons": cons,
                "#Episodes": episode,
                "RL_Obj": None,
                "Mean_Rwd": None,
                "q_s0": None,
                "#Success": success_count,
                "Vio_Constr": env.violated_constraints,
                "Vio_Bounds": env.violator_bounds,
                "Obj_Change": None,
                "AvgEp_Time": None,
                "Ent": None,        
                "Max_Ent": None,
                "critic_loss": None,
                "actor_loss": None,
                "Train_Time": None,
                "Policy_Time": None
                })
                return actor, critic, episode_returns, qs0, critic_losses, actor_losses, success_count, obj_change_list, csv_vals
        #env.obj_val1 = env.init_obj_val
        while True:
            state_t = torch.FloatTensor(state).to(device)
            valid   = env.compute_feasible_actions(state)
            #print(f"\n Valid Actions : {valid}")
            # Filter out overexplored actions if fresh alternatives exist
            '''fresh = [a for a in valid
                    if  0.7*step_action_counts[a] + 0.3*action_counts[a] < threshold]  # <<< CHANGE >>> threshold

            if fresh:
                action = sample_action_nn(env,actor, state_t, fresh, n_actions)
            else:
                action = sample_action_nn(env, actor, state_t, valid, n_actions)'''

            #action = sample_action_nn(actor, state_t, valid, n_actions)
            action = epsilon_sample_action_nn(actor, state_t, valid, n_actions, epsilon)
            #action = (action_root, dir_root = 'L' if dir_root == 'DOWN' else 'R')  # Use the action from the root node solution as the first action in the first episode
            next_state, reward, terminated, truncated, reason, obj_change = env.step(action)
            trajectory.append((state, action, reward))

            print(f"\nenv.step():{env.var_id_to_actionTuple[int(action)]}, {reward:.2f}, {terminated}, {truncated}, {reason}")

            '''assert next_state is not None, \
                f"Initial State is Terminal. Reason = {reason}" '''

            done = terminated or truncated
            
            total_return += reward
            #print(f"\nDataType(obj_change): {type(obj_change)}")
            #print(f"\nDataType(total_obj_change): {type(total_obj_change)}")            
            #total_obj_change = total_obj_change + obj_change
            obj_change_coll.append(obj_change)
            step_reward.append(reward)
            #next_state_t = torch.FloatTensor(next_state).to(device)

            with torch.no_grad():
                if done == True:
                    td_target = torch.tensor(reward, dtype=torch.float32, device=device)
                else:
                    next_state_t = torch.FloatTensor(next_state).to(device)
                    #next_valid    = [a for a in env.compute_feasible_actions(next_state) if a is not None]
                    next_valid    = env.compute_feasible_actions(next_state)
                    #print(f"  Next valid actions: {next_valid}")
                    next_q_vals   = critic_target(next_state_t)
                    next_q_masked = mask_invalid(next_q_vals, next_valid, n_actions)
                    best_q_next   = next_q_masked.max()
                    td_target     = torch.tensor(reward + gamma * best_q_next.item(),
                                                 dtype=torch.float32, device=device)

            q_vals      = critic(state_t)
            q_sa        = q_vals[action]
            critic_loss = (td_target - q_sa) ** 2
            print(f"\n Critic Loss = {critic_loss}")
            total_critic_loss += critic_loss.item()

            critic_opt.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            logits   = actor(state_t)
            masked   = mask_invalid(logits, valid, n_actions)
            probs    = torch.softmax(masked, dim=0)
            log_prob = torch.log_softmax(masked, dim=0)[action]
            entropy  = -torch.sum(probs * torch.log(probs + 1e-10))

            with torch.no_grad():
                q_sa_baseline = critic(state_t)[action]

            actor_loss = -I * log_prob * q_sa_baseline - entropy_coeff * entropy
            print(f"\nActor Loss = {actor_loss}")
            total_actor_loss += actor_loss.item()
            actor_opt.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()
            if done:
                print(f"prob :- {prob}. Episode {episode} ends. Terminated={terminated}, Truncated={truncated}")
                print(f"Reason: {reason}")
                trajectory.append((next_state, None, 0.0))
                if reason in ['ALL FIXED (Desirable)', 'FEASIBLE (Desirable)']:
                    success_count += 1
                    success_obj.append(env.obj_val2)
                    #successful_trajectories.append(trajectory)
                    #successful_trajectories.append("End")
                trajectory = []  # Clear trajectory for next episode
                break

            I *= gamma
            state  = next_state

           

        # Issue 5 — append terminal state
        time2 = time.perf_counter()
        polyak_update_nn(critic, critic_target, tau)
        #episode_returns.append(total_return)
        episode_returns[episode] = total_return
        #episode_returns[episode] = np.mean(step_reward) #calculated per step in this episode i.e. reward per step in this episode      
        #print(f"\nObj_change_coll: {obj_change_coll}")
        clean_coll = [x for x in obj_change_coll if isinstance(x, (int, float))]
        q_initial = critic(torch.FloatTensor(env.init_state).to(device))
        qs0[episode] = torch.max(q_initial).item()  # Q-value of the initial state in this episode
        critic_losses[episode] = total_critic_loss  # Total critic loss in this episode
        actor_losses[episode] = total_actor_loss  # Total actor loss in this episode
        #print(f"Cleaned Obj_change_coll: {clean_coll}")
        obj_change_list[episode] = np.sum(clean_coll) 
        #obj_change_list[episode] = np.mean(clean_coll)  #calculated per step in this episode i.e. ob_change per step in this episode
        vio_constr[episode] = env.violated_constraints  #calculated at the last state of the trajectory in this episode
        vio_bounds[episode] = env.violator_bounds       #calculated at the last state of the trajectory in this episode
        ep_times[episode] = time2 - time1     #calculated after the last state of the trajectory in this episode
        

        # Issue 2 — fixed progress print
        #if episode % 1 == 0:
        if episode % 100 == 0:
            formatted_coll = [f"{x:.2f}" for x in clean_coll]
            print(f"Objectve changes in episode {episode}: {formatted_coll}\n") #In this episode at each step
            print(f"\n Mean obj_change in episode {episode}: {np.mean(clean_coll):.2f}") #In this episode
            print(f"\n\nRunning the learnt policy to solve the {prob} at episode {episode}:**************************************************")
            time3 = time.perf_counter()
            trajectory, total_return, obj_final, x_final, reason, entropy,max_entropy = run_policy_nn(env, actor, device=device, hours=0.5) 
            time4 = time.perf_counter()
            #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
            print(f"Time taken by {episode} episode_policy :", datetime.timedelta(seconds=time4-time3))
            print(f"Trajectory length: {len(trajectory)}")
            print(f"Total return: {total_return:.2f}")

            #print(f"\nSolution in Final State: {x_final}")
            print(f"\nObjective in Final State: {obj_final}")
            print(f"\nTotal Return: {total_return:.2f} over {len(trajectory)-1} steps")
            print(f"\nReason for termination: {reason}")
            csv_vals.append({
            "Problem": prob,
            "#Vars": vars,
            "#Cons": cons,
            "#Episodes": episode,
            "RL_Obj": f"{obj_final:.2f}" if obj_final is not None else obj_final, #returned from the policy at this episode
            "Mean_Rwd": f"{(np.mean(episode_returns)):.2f}" , #Mean reward per step per episode till this episode
            #"Mean_Rwd": f"{(episode_returns[episode]):.2f}" , #Total reward in this episode
            "q_s0": qs0[episode], #Q-value of the initial state in this episode
            "#Success": success_count, #In this episode
            "Vio_Constr": f"{(np.mean(vio_constr)):.2f}", #constr_violations per episode till this episode
            "Vio_Bounds": f"{(np.mean(vio_bounds)):.2f}", #bounds_violations per episode till this episode
            "Obj_Change": f"{(np.mean(obj_change_list)):.2f}", #Obj_change per step per episode till this episode
            "AvgEp_Time": f"{(np.mean(ep_times)):.2f}",        #Time taken per episode till this episode
            "Ent": f"{entropy:.2f}",                      #Entropy of the policy at this episode
            "Max_Ent": f"{max_entropy:.2f}",              #Maximum entropy of the policy at this episode (taking in to account 2N actions)
            "critic_loss": f"{total_critic_loss:.2f}",  #Total critic loss in this episode
            "actor_loss": f"{total_actor_loss:.2f}",        #Total actor loss
            "Train_Time": f"{(np.sum(ep_times)):.2f}",    #Time elapsed in training the policy since the start till this episode
            "Policy_Time": f"{(time4-time3):.2f}"}          #Time taken to run the policy at this episode
            )
        
        # 3. Calculate elapsed time since the loop started
        elapsed_time = time.perf_counter() - start_time
        
        # 4. Check if the time limit has been crossed
        if elapsed_time >= Time_Limit * 3600:  # Convert hours to seconds
            print(f"⏱️ Time limit of {Time_Limit}hrs reached for {prob}! Terminating early at {episode} before {n_episodes} episodes.")
            break
    

    return actor, critic, episode_returns, qs0, critic_losses, actor_losses, vio_constr, vio_bounds, success_count,obj_change_list, csv_vals



def twoCritic(
    vars,
    cons,
    prob,
    env,
    run,
    csv_vals,
    Time_Limit,
    n_episodes,
    gamma: float       = 1.0,         # Issue 1 — no discounting for episodic tasks
    alpha_w: float      = 1e-3,
    alpha_theta: float  = 1e-5,        # Issue 7 — changed from 1e-4
    tau: float          = 0.005,
    hidden_dim: int     = 128,
    device: str         = None,
    entropy_coeff_start: float = 0.9,
    entropy_coeff_end: float   = 0.01,
    epsilon1: float = 1.0,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    device = torch.device(device)
    n_state   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    critic1        = CriticNet(n_state, n_actions, hidden_dim).to(device)
    #critic2        = CriticNet(n_state, n_actions, hidden_dim).to(device)
    critic_target1 = CriticNet(n_state, n_actions, hidden_dim).to(device)
    #critic_target2 = CriticNet(n_state, n_actions, hidden_dim).to(device)
    critic_target1.load_state_dict(critic1.state_dict())
    #critic_target2.load_state_dict(critic2.state_dict())
    for p in critic_target1.parameters():
        p.requires_grad = False
    '''for p in critic_target2.parameters():
        p.requires_grad = False'''

    actor = ActorNet(n_state, n_actions, hidden_dim).to(device)

    critic_opt1 = optim.Adam(critic1.parameters(), lr=alpha_w)
    #critic_opt2 = optim.Adam(critic2.parameters(), lr=alpha_w)
    actor_opt  = optim.Adam(actor.parameters(),  lr=alpha_theta)
    #critic_opt = optim.SGD(critic1.parameters(), lr=alpha_w)
    #actor_opt  = optim.SGD(actor.parameters(),  lr=alpha_theta)
    # Instead of: torch.optim.SGD(model.parameters(), lr=0.001)
    #torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)   
    #env.alpha = alpha
    success_count = 0
    success_obj = []
    successful_X = {}
    #csv_vals = []
    episode_returns = [0]*n_episodes       # Storing episodic returns for each episode
    vio_constr = [0]*n_episodes
    obj_change_list = [0]*n_episodes
    vio_bounds = [0]*n_episodes
    ep_times = [0]*n_episodes
    qs0 = [0]*n_episodes
    critic_losses = [0]*n_episodes
    critic1_losses = [0]*n_episodes
    critic2_losses = [0]*n_episodes
    actorGradNorm = [0]*n_episodes
    start_time = time.perf_counter()
    decay_rate = 0.01
    #env.initial_config()
    for episode in range(n_episodes):
        time1 = time.perf_counter()
        print(f"Episode {episode}/{n_episodes-1}")
        fraction      = episode / max(1, n_episodes - 1)
        entropy_coeff = entropy_coeff_start + fraction * (entropy_coeff_end - entropy_coeff_start)
        epsilon1 = max(0.1, epsilon1 * (1 - fraction))  # Decay epsilon over episodes, but not below 0.1
        #epsilon2 = max(0.1, 1.0/(1 + decay_rate * episode))  # Decay epsilon over episodes, but not below 0.1
        #epsilon = 0
        print(f"\nEpsilon for exploration: {epsilon1:.4f} | Entropy coefficient: {entropy_coeff:.4f}")
    
        #state = env.scip_reset()
        state = env.reset()
        min_length = float('inf')
        #tc, obj, x, _, _  = solve_bip_gurobiws(env.gmodel,vec=None,relax=False,warm_start=None,vbasis=None,cbasis=None)
        I            = 1.0
        total_return = 0.0
        total_critic_loss = 0.0
        total_critic1_loss = 0.0
        total_critic2_loss = 0.0
        total_actor_loss = 0.0
        total_obj_change = 0.0
        #obj_change_coll = []
        step_reward = []
        episode_grad_norms = []  # To store gradient norms for this episode

        trajectory   = []          # Issue 4 — initialised here
        if episode == 0:
            if env.quit == True:
                trajectory.append((state, None, 0.0))
                episode_returns[episode] = total_return
                #obj_change_list.append(obj_change)
                qs0[episode] = 0
                critic1_losses[episode] = 0
                critic2_losses[episode] = 0
                actorGradNorm[episode] = 0
                csv_vals.append({
                "Problem": prob,
                "#Vars": vars,
                "#Cons": cons,
                "#Episodes": episode,
                "RL_Obj": None,
                "Uni_Obj": None,
                "Mean_Rwd": None,
                "V(S0)": None,
                "#Success": success_count,
                #"Vio_Constr": env.violated_constraints,
                #"Vio_Bounds": env.violator_bounds,
                #"Obj_Change": None,
                "AvgEp_Time": None,
                "Ent": None,        
                "Max_Ent": None,
                "critic1_loss": None,
                #"critic2_loss": None,
                "actor_loss": None,
                "Train_Time": None,
                "Policy_Time": None
                })
                return actor,  episode_returns, qs0, critic1_losses, critic2_losses, critic_losses, actorGradNorm, obj_change_list, csv_vals
        #env.obj_val1 = env.init_obj_val
        while True:
            state_t = torch.FloatTensor(state).to(device)
            valid   = env.compute_feasible_actions(state)
            print(f"\n#Valid Actions : {len(valid)}")
            #print(f"\n Valid Actions : {valid}")
            # Filter out overexplored actions if fresh alternatives exist
            '''fresh = [a for a in valid
                    if  0.7*step_action_counts[a] + 0.3*action_counts[a] < threshold]  # <<< CHANGE >>> threshold

            if fresh:
                action = sample_action_nn(env,actor, state_t, fresh, n_actions)
            else:
                action = sample_action_nn(env, actor, state_t, valid, n_actions)'''
            
            '''if episode <= 300:
                action = random.choice(valid)  # Uniform exploration
            else:'''
            #action = greedy_action_nn(actor, state_t, valid, n_actions)
            #action = sample_action_nn(actor, state_t, valid, n_actions)
            action = epsilon_sample_action_nn(actor, state_t, valid, n_actions, epsilon1)
            #action = (action_root, dir_root = 'L' if dir_root == 'DOWN' else 'R')  # Use the action from the root node solution as the first action in the first episode
            #next_state, reward, obj_val, x_val, terminated, truncated, reason = env.step(state, action)  #Fpr env_TreeEp2
            next_state, reward, obj_val, x_val, terminated, truncated, reason = env.step(action)  # For env_mdp3
            trajectory.append((state, action, reward))

            print(f"\nenv.step():{env.var_id_to_actionTuple[int(action)]}, {reward:.2f}, {terminated}, {truncated}, {reason}")

            '''assert next_state is not None, \
                f"Initial State is Terminal. Reason = {reason}" '''

            done = terminated or truncated
            
            total_return += reward
            #print(f"\nDataType(obj_change): {type(obj_change)}")
            #print(f"\nDataType(total_obj_change): {type(total_obj_change)}")            
            #total_obj_change = total_obj_change + obj_change
            #obj_change_coll.append(obj_change)
            step_reward.append(reward)
            #next_state_t = torch.FloatTensor(next_state).to(device)

            with torch.no_grad():
                if done == True:
                    td_target1 = torch.tensor(reward , dtype=torch.float32, device=device)
                    #td_target2 = torch.tensor(reward, dtype=torch.float32, device=device)
                else:
                    next_state_t = torch.FloatTensor(next_state).to(device)
                    #next_valid    = [a for a in env.compute_feasible_actions(next_state) if a is not None]
                    next_valid    = env.compute_feasible_actions(next_state)
                    #print(f"  Next valid actions: {next_valid}")

                    #For the first critic
                    next_q_vals1   = critic_target1(next_state_t)
                    next_q_masked1 = mask_invalid(next_q_vals1, next_valid, n_actions)
                    best_q_next1   = next_q_masked1.max()
                    td_target1     = torch.tensor(reward + gamma * best_q_next1.item(),
                                                 dtype=torch.float32, device=device)

                    '''#For the second critic
                    next_q_vals2   = critic_target2(next_state_t)
                    next_q_masked2 = mask_invalid(next_q_vals2, next_valid, n_actions)
                    best_q_next2   = next_q_masked2.max()
                    td_target2     = torch.tensor(gamma * best_q_next2.item(),
                                                 dtype=torch.float32, device=device)'''


            #Update critic1 network
            q_vals1      = critic1(state_t)
            q_sa1     = q_vals1[action]
            critic_loss1 = (td_target1 - q_sa1) ** 2
            print(f"\n Critic1 Loss = {critic_loss1}")
            total_critic1_loss += critic_loss1.item()

            critic_opt1.zero_grad()
            critic_loss1.backward()
            #torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=1.0)
            critic_opt1.step()

            '''#Update critic2 network
            q_vals2      = critic2(state_t)
            q_sa2        = q_vals2[action]
            critic_loss2 = (td_target2 - q_sa2) ** 2
            print(f"\n Critic2 Loss = {critic_loss2}")
            total_critic2_loss += critic_loss2.item()
            total_critic_loss = total_critic1_loss + total_critic2_loss

            critic_opt2.zero_grad()
            critic_loss2.backward()
            critic_opt2.step()'''

            
            logits   = actor(state_t)
            masked   = mask_invalid(logits, valid, n_actions)
            probs    = torch.softmax(masked, dim=0)
            log_prob = torch.log_softmax(masked, dim=0)[action]
            entropy  = -torch.sum(probs * torch.log(probs + 1e-10))
            
            with torch.no_grad():
                q_sa_baseline = critic1(state_t)[action]

            actor_loss = -I * log_prob * q_sa_baseline - entropy_coeff * entropy
            print(f"\nActor Loss = {actor_loss}")
            total_actor_loss += actor_loss.item()
            actor_opt.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            total_norm_sq = 0.0
            for p in actor.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm_sq += param_norm.item() ** 2

            policy_grad_norm = total_norm_sq ** 0.5
            actor_opt.step()
            episode_grad_norms.append(policy_grad_norm)

            if done:
                print(f"prob :- {prob}. Episode {episode} ends. Terminated={terminated}, Truncated={truncated}")
                print(f"\n Episode Length = {env.step_count}")
                print(f"Reason: {reason}")
                trajectory.append((next_state, None, 0.0))
                if reason in ['ALL FIXED (Desirable)', 'FEASIBLE (Desirable)']:
                    success_count += 1
                    key = make_hashable(x_val)
                    if key not in successful_X:
                        successful_X[key] = (x_val, obj_val)


                trajectory = []  # Clear trajectory for next episode
                break

            I *= gamma

            state  = next_state

           

        # Issue 5 — append terminal state
        time2 = time.perf_counter()
        if env.step_count < min_length:
                min_length = env.step_count
        print(f"\nCurrent Min Length: {min_length}")
        polyak_update_nn(critic1, critic_target1, tau)
        #polyak_update_nn(critic2, critic_target2, tau)
        #episode_returns.append(total_return)
        episode_returns[episode] = total_return
        #episode_returns[episode] = np.mean(step_reward) #calculated per step in this episode i.e. reward per step in this episode      
        #print(f"\nObj_change_coll: {obj_change_coll}")
        #clean_coll = [x for x in obj_change_coll if isinstance(x, (int, float))]
        q_initial = critic1(torch.FloatTensor(env.init_state).to(device))
        qs0[episode] = torch.max(q_initial).item()  # Q-value of the initial state in this episode
        critic1_losses[episode] = total_critic1_loss  # Total critic1 loss in this episode
        #critic2_losses[episode] = total_critic2_loss  # Total critic2 loss in this episode
        critic_losses[episode] = total_critic_loss  # Total critic loss in this episode
        actorGradNorm[episode] = sum(episode_grad_norms)/len(episode_grad_norms) if episode_grad_norms else 0  # Average actor gradient norm in this episode
        #print(f"Cleaned Obj_change_coll: {clean_coll}")
        #obj_change_list[episode] = np.sum(clean_coll) 
        #obj_change_list[episode] = np.mean(clean_coll)  #calculated per step in this episode i.e. ob_change per step in this episode
        #vio_constr[episode] = env.violated_constraints  #calculated at the last state of the trajectory in this episode
        #vio_bounds[episode] = env.violator_bounds       #calculated at the last state of the trajectory in this episode
        ep_times[episode] = time2 - time1     #calculated after the last state of the trajectory in this episode
        

        # Issue 2 — fixed progress print
        #if episode % 1 == 0:
        if episode % 1000 == 0:
            #formatted_coll = [f"{x:.2f}" for x in clean_coll]
            #print(f"Objectve changes in episode {episode}: {formatted_coll}\n") #In this episode at each step
            #print(f"\n Mean obj_change in episode {episode}: {np.mean(clean_coll):.2f}") #In this episode
            print(f"\n\nRunning the learnt policy to solve the {prob} at episode {episode}:**************************************************")
            time3 = time.perf_counter()
            trajectory, total_return, obj_final, x_final, reason, entropy, max_entropy = run_policy_nn(env, actor, device=device, hours = 0.1) 
            uni_trajectory, uni_total_return, uni_obj, uni_x, reason = run_uniform(env, device=device, hours = 0.1)
            time4 = time.perf_counter()
            #print(f"Time taken by {alpha} alpha_policy :", datetime.timedelta(seconds=time6-time5))
            print(f"Time taken by {episode} episode_policy :", datetime.timedelta(seconds=time4-time3))
            print(f"Trajectory length: {len(trajectory)}")
            print(f"Total return: {total_return:.2f}")

            #print(f"\nSolution in Final State: {x_final}")
            print(f"\nObjective in Final State: {obj_final}")
            print(f"\nTotal Return: {total_return:.2f} over {len(trajectory)-1} steps")
            print(f"\nReason for termination: {reason}")
            csv_vals.append({
            "Problem": prob,
            "#Vars": vars,
            "#Cons": cons,
            "#Episodes": episode,
            "RL_Obj": f"{obj_final:.2f}" if obj_final is not None else obj_final, #returned from the policy at this episode
            "Uni_Obj": f"{uni_obj:.2f}" if uni_obj is not None else uni_obj, #returned from the uniform policy at this episode
            #"Mean_Rwd": f"{(np.mean(episode_returns)):.2f}" , #Mean reward per step per episode till this episode
            "Mean_Rwd": f"{(episode_returns[episode]):.2f}" , #Total reward in this episode
            "V(S0)": f"{qs0[episode]:.2f}", #Q-value of the initial state in this episode
            "#Success": success_count, #In this episode
            #"Vio_Constr": f"{(np.mean(vio_constr)):.2f}", #constr_violations per episode till this episode
            #"Vio_Bounds": f"{(np.mean(vio_bounds)):.2f}", #bounds_violations per episode till this episode
            #"Obj_Change": f"{(np.mean(obj_change_list)):.2f}", #Obj_change per step per episode till this episode
            "AvgEp_Time": f"{(np.mean(ep_times)):.2f}",        #Time taken per episode till this episode
            "Ent": f"{entropy:.2f}",                      #Entropy of the policy at this episode
            "Max_Ent": f"{max_entropy:.2f}",              #Maximum entropy of the policy at this episode (taking in to account 2N actions)
            "critic1_loss": f"{total_critic1_loss:.2f}",  #Total critic1 loss in this episode
            #"critic2_loss": f"{total_critic2_loss:.2f}",  #Total critic2 loss in this episode
            "actor_loss": f"{total_actor_loss:.2f}",        #Total actor loss
            "Train_Time": f"{(np.sum(ep_times)):.2f}",    #Time elapsed in training the policy since the start till this episode
            "Policy_Time": f"{(time4-time3):.2f}"}          #Time taken to run the policy at this episode
            )
        
        # 3. Calculate elapsed time since the loop started
        elapsed_time = time.perf_counter() - start_time
        
        # 4. Check if the time limit has been crossed
        if elapsed_time >= Time_Limit * 3600:  # Convert hours to seconds
            print(f"⏱️ Time limit of {Time_Limit}hrs reached for {prob}! Terminating early at {episode} before {n_episodes} episodes.")
            break

    distinct_count = len(successful_X)
    print(f"\nTotal distinct feasible solutions of {prob}: {distinct_count}\n")
    #print_vector_table(successful_X, max_vec_items=10, col_width=35)

    return actor, episode_returns, qs0, critic1_losses, critic2_losses, critic_losses, actorGradNorm, obj_change_list, csv_vals


def actor_critic_with_backtrack(
    env,
    gamma: float               = 0.99,
    alpha_w: float             = 1e-3,
    alpha_theta: float         = 1e-5,
    tau: float                 = 0.005,
    hidden_dim: int            = 128,
    n_episodes: int            = 1000,       # <<< CHANGE >>>
    max_backtracks: int        = 5,          # <<< CHANGE >>> max retries per state
    device: str                = None,
    entropy_coeff_start: float = 0.9,
    entropy_coeff_end: float   = 0.01
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    device = torch.device(device)
    n_state   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    critic        = CriticNet(n_state, n_actions, hidden_dim).to(device)
    critic_target = CriticNet(n_state, n_actions, hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor      = ActorNet(n_state, n_actions, hidden_dim).to(device)
    critic_opt = optim.Adam(critic.parameters(), lr=alpha_w)
    actor_opt  = optim.Adam(actor.parameters(),  lr=alpha_theta)

    episode_returns = []

    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        fraction      = episode / max(1, n_episodes - 1)
        entropy_coeff = (entropy_coeff_start +
                         fraction * (entropy_coeff_end - entropy_coeff_start))

        state, _     = env.reset()
        I            = 1.0
        total_return = 0.0
        trajectory   = []

        # Tracks which actions were tried at each step
        # step -> set of action ids already tried
        tried_actions = defaultdict(set)

        step = 0
        step_back = 0
        terminated = False
        if env.is_terminal(state)[0] == True:
            print(f"  ✓ Initial state is terminal. Reason: {env.is_terminal(state)[4]}")
            trajectory.append((state, None, 0.0))
            episode_returns.append(total_return)
            return actor, critic, episode_returns
        while True:
            if terminated == True:
                term_status = env.is_terminal(state)[4]
                if term_status == 'ALL FIXED (Desirable)' or term_status == 'FEASIBLE (Desirable)':  # Terminal with all vars fixed — no point backtracking
                        # Reached goal — end episode
                        
                        print(f"  ✓ Winner terminal reached at step {step} -- ending episode. Reason: {term_status}")
                        print(f"\n  Terminal state x_val: {env.is_terminal(next_state)[3]}")
                        print(f"  Terminal state obj_val: {env.is_terminal(next_state)[2]}")
                        trajectory.append((state, None, 0.0))
                        break

                elif term_status == 'INFEASIBLE' or term_status == 'INF_OR_UNBD' or term_status == 'UNBOUNDED':
                        # ── Undesirable terminal — BACKTRACK ──────────────
                        print(f"  ✗ Looser terminal at step {step} — backtracking. Reason: {term_status}")

                        if step_back >= 4*max_backtracks:
                            print(f"  ✗ Maximum backtracks ({4*max_backtracks}) reached at step {step} — ending episode")
                            break

                        if not trajectory:
                            print(f"  \n# No previous state to go back to, ending episode")
                            #print("\n NO SOLUTION EXISTS for this problem instance! \n")
                            break

                        # Get previous state and the bad action that led here
                        prev_state, bad_action, _ = trajectory[-1]

                    

                        # ── Check if we have exhausted all actions ─────────
                        prev_valid_set = set(env.compute_feasible_actions(prev_state))
                        remaining      = prev_valid_set - tried_actions[step - 1]

                        if not remaining or len(tried_actions[step-1]) >= max_backtracks:  #DO WE Need to control the depth of backtracking as well? Right now, it can go till the beginning of the episode if we keep exhausting actions at each step
                            # Exhausted retries — go back further
                            print(f"  Exhausted retries at step {step-1} — backtracking further")
                            trajectory.pop()
                            step -= 1
                            step_back += 1
                            if not trajectory:
                                print(f"  \n# No previous state to go back to, ending episode")
                                break
                            prev_state, _, _ = trajectory[-1]
                            state = prev_state
                        else:
                            # Go back to previous state and retry
                            trajectory.pop()
                            state  = prev_state
                            step  -= 1
                            step_back += 1
                        terminated = False
                        continue   # retry from previous state

            # ── Normal step ───────────────────────────────────────────
            state_t = torch.FloatTensor(state).to(device)
            valid   = env.compute_feasible_actions(state)

            # Exclude already-tried actions at this step
            untried = [a for a in valid
                       if a not in tried_actions[step]]

            if untried:
                action = sample_action_nn(env, actor, state_t, untried, n_actions)
            else:
                # All tried — pick by policy anyway
                action = sample_action_nn(env, actor, state_t, valid, n_actions)

            next_state, reward, terminated, truncated, reason, _ = env.step(action)
            trajectory.append((state, action, reward))
            print(f"\nenv.step():{env.var_id_to_actionTuple[int(action)]}, {reward}, {terminated}, {truncated}")


            assert next_state is not None, \
                f"env.step() returned None at episode {episode}, action {action}"

            done = terminated or truncated
           
            total_return += reward

            next_state_t = torch.FloatTensor(next_state).to(device)

            # ── TD target ──────────────────────────────────────────────
            with torch.no_grad():
                if done:
                    td_target = torch.tensor(reward,
                                    dtype=torch.float32, device=device)
                else:
                    next_valid    = env.compute_feasible_actions(next_state)
                    next_q_vals   = critic_target(next_state_t)
                    next_q_masked = mask_invalid(next_q_vals, next_valid,
                                                 n_actions)
                    best_q_next   = next_q_masked.max()
                    td_target     = torch.tensor(
                                        reward + gamma * best_q_next.item(),
                                        dtype=torch.float32, device=device)

            # ── Critic update ──────────────────────────────────────────
            q_vals      = critic(state_t)
            q_sa        = q_vals[action]
            critic_loss = (td_target - q_sa) ** 2
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # ── Actor update ───────────────────────────────────────────
            logits   = actor(state_t)
            masked   = mask_invalid(logits, valid, n_actions)
            probs    = torch.softmax(masked, dim=0)
            log_prob = torch.log_softmax(masked, dim=0)[action]
            entropy  = -torch.sum(probs * torch.log(probs + 1e-10))

            with torch.no_grad():
                q_sa_baseline = critic(state_t)[action]

            actor_loss = (-I * log_prob * q_sa_baseline
                          - entropy_coeff * entropy)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            I     *= gamma
            tried_actions[step].add(action)
            if truncated:
                print(f"Episode ended with truncated={truncated}")
                print(f"Reason: {reason}")
                trajectory.append((next_state, None, 0.0))
                break
            state  = next_state
            step  += 1

        polyak_update_nn(critic, critic_target, tau)
        episode_returns.append(total_return)

        log_every = max(1, n_episodes // 10)
        if (episode + 1) % log_every == 0:
            avg = np.mean(episode_returns[-log_every:])
            print(f"Episode {episode+1:5d} | Avg return: {avg:.2f} | "
                  f"Entropy coeff: {entropy_coeff:.4f}")

    return actor, critic, episode_returns
