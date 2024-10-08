import numpy as np
import pickle
import copy

from pathlib import Path

from state import State, get_all_states
     
    
def state_space_initalize(size):
    save_path = Path(f"states_{size}.pkl")
    if save_path.is_file():
        with open(save_path, "rb") as f:
            pickled = pickle.load(f)
            all_states, rewards = pickled[0], pickled[1]
            return all_states, rewards

    all_states = get_all_states(size)
    rewards = dict()

    for state in all_states:
        if all_states[state].is_end():
            if all_states[state].winner == 1:
                rewards[state] = (1, -1)
            elif all_states[state].winner == 0:
                rewards[state] = (0.5, 0.5)
            else:
                rewards[state] = (-1, 1)
        else:
            rewards[state] = (0, 0)

    with open(save_path, "wb") as f:
        pickle.dump((all_states, rewards), f)
    
    return all_states, rewards


def policy_evaluation(all_states, rewards, values, policy):
    
    loop = 0
    while True:
        loop += 1
        DELTA = 0
        for state in reversed(all_states):
            if all_states[state].is_end():
                continue
            temp = [values[state][0], values[state][1]]
            next_state = policy[state]
            new_value = [rewards[next_state][0] + GAMMA * values[next_state][0], 
                        rewards[next_state][1] + GAMMA * values[next_state][1]]
            values[state] = [new_value[0], new_value[1]]
            DELTA = max(DELTA, abs(temp[0] - values[state][0]), abs(temp[1] - values[state][1]))
        print(f'Evaluation loop {loop}, delta={DELTA}')
        if DELTA < EPSILON:
            break
    print("Evaluation Done")
    


if __name__ == '__main__':
    INIT_SIZE = 4
    all_states, rewards  = state_space_initalize(INIT_SIZE)

    values = dict.fromkeys(all_states.keys(), [0.1, 0.1])
    for state in all_states:
        if all_states[state].is_end():
            values[state] = [0, 0]

    policy = dict.fromkeys(all_states.keys(), 0)

    # Random Policy
    for state in all_states:
        if not all_states[state].is_end():
            policy[state] = np.random.choice(all_states[state].get_next_states())

    EPSILON = 0.1
    GAMMA = 0.1
    # Policy Improvement
    while True:
        # Policy Evaluation
        policy_evaluation(all_states, rewards, values, policy)
        policy_stable = True
        for state in all_states:
            player = all_states[state].player
            index = 0 if player == 1 else 1
            if not all_states[state].is_end():
                old_action = policy[state]
                next_states = all_states[state].get_next_states()
                next_values = [rewards[next_state][index] + GAMMA * values[next_state][index]
                                for next_state in next_states]
                new_action = next_states[np.argmax(next_values)]
                if old_action != new_action:
                    policy_stable = False
                policy[state] = new_action
        if policy_stable:
            break

    print("Policy Improvement Done")
    print(policy)
                    
    with open(f'policy_{INIT_SIZE}.pkl', "wb") as f:
        pickle.dump(policy, f)