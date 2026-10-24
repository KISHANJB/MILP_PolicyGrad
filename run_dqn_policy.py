from stable_baselines3 import DQN
from DQN_sb3 import RUN_ALGORITHM, category

SAVE_PATH = f"D:\\policy_files\\{RUN_ALGORITHM}\\{category}"
model = DQN.load(SAVE_PATH, env=env)

import numpy as np
import torch

def extract_q_values(model, observation):
    """
    Return the learned Q-value for every discrete action.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(observation)

    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor)

    return q_values.cpu().numpy().reshape(-1)


def select_greedy_action(model, env, observation):
    """
    Select the valid action with the highest learned Q-value.
    """
    q_values = extract_q_values(model, observation)

    # YOUR MDP'S VALID-ACTION COMPUTATION
    valid_actions = env.compute_feasible_actions(observation)

    if len(valid_actions) == 0:
        raise RuntimeError("No valid actions available.")

    # Assumes valid_actions contains discrete integer action indices.
    valid_actions = np.asarray(valid_actions, dtype=int)

    action = int(valid_actions[np.argmax(q_values[valid_actions])])

    return action, q_values
