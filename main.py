import numpy as np
import pickle

class State:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.data = np.zeros((nrows, ncols))
        self.player = 1
        self.winner = None
        self.end = None 

    def __hash__(self):
        return hash(str(self.data))
    
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)
    
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(self.nrows):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(self.ncols):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(self.nrows):
            trace += self.data[i, i]
            reverse_trace += self.data[i, self.ncols - 1 - i]
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
        if sum_values == self.nrows * self.ncols:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end
    
    def next_state(self, i, j, symbol):
        new_state = State(self.nrows, self.ncols)
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def print_state(self):
        for i in range(self.nrows):
            print('-------------')
            out = '| '
            for j in range(self.ncols):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

    



if __name__ == '__main__':
    state = State(3, 3)
    state.data = np.array([[1, -1, -1], [0, 1, 0], [0, 0, 1]])
    print(state.print_state())