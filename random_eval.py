import numpy as np

import pickle

class State:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros((size, size))
        self.player = 1
        self.winner = None
        self.end = None 
        self.hash_val = None

    def __hash__(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val
    
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)
    
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(self.size):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(self.size):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(self.size):
            trace += self.data[i, i]
            reverse_trace += self.data[i, self.size - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == self.size * self.size:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end
    
    def next_state(self, i, j):
        new_state = State(self.size)
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = self.player
        new_state.player = -self.player
        return new_state
    
    def get_next_states(self):
        next_states = []
        if self.is_end():
            return next_states
        for i in range(self.size):
            for j in range(self.size):
                if self.data[i][j] == 0:
                    new_state = self.next_state(i, j)
                    new_hash = new_state.__hash__()
                    next_states.append(new_hash)
        return next_states


    def print_state(self):
        for i in range(self.size):
            out = ' '
            for j in range(self.size):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' '
            print(out)
        print('\n')

if __name__ == "__main__":
    INIT_SIZE = 3
    with open(f'policy_{INIT_SIZE}.pkl', 'rb') as f:
        policy = pickle.load(f)

    with open(f'states_{INIT_SIZE}.pkl', 'rb') as f:
        all_states = pickle.load(f)[0]

    opponent_policy = dict.fromkeys(all_states.keys(), 0)


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
                current_state = all_states[np.random.choice(current_state.get_next_states())]

        if current_state.winner == player:
            win += 1    
        elif current_state.winner == opponent:
            loss += 1
        else:
            draw += 1

    print(f"Win: {win}, Loss: {loss}, Draw: {draw}")
    print(f"Win rate: {win/10000}")
    