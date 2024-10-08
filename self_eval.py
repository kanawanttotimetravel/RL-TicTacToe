import numpy as np

import pickle

from state import State

if __name__ == "__main__":
    INIT_SIZE = 3
    with open(f'policy_{INIT_SIZE}.pkl', 'rb') as f:
        policy = pickle.load(f)

    with open(f'states_{INIT_SIZE}.pkl', 'rb') as f:
        all_states = pickle.load(f)[0]

    win = 0
    loss = 0
    draw = 0

    for _ in range(10000):
        player = np.random.choice([1, -1])
        opponent = -player
        current_state = State(INIT_SIZE)
        
        while not current_state.is_end():
            if current_state.player == player:
                current_state = all_states[policy[current_state.__hash__()]]
            else:
                current_state = all_states[policy[current_state.__hash__()]]

        if current_state.winner == player:
            win += 1    
        elif current_state.winner == opponent:
            loss += 1
        else:
            draw += 1

    print(f"Win: {win}, Loss: {loss}, Draw: {draw}")
    print(f"Win rate: {win/10000}")
    