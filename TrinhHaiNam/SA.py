import numpy as np
import random
import math
import time

class SimulatedAnnealingScheduler:
    def __init__(self, N, D, A, B, dayoff):
        self.N = N
        self.D = D
        self.A = A
        self.B = B
        self.dayoff = dayoff

    def is_valid_assignment(self, X):
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1 and X[i][d] != 0:
                    return False
                if d < self.D and X[i][d] == 4 and X[i][d + 1] != 0:
                    return False
        for d in range(1, self.D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < self.A or count > self.B:
                    return False
        return True

    def initialize_solution(self):
        X = np.zeros((self.N + 1, self.D + 1), dtype=int)
        for d in range(1, self.D + 1):
            for shift in range(1, 5):
                count = 0
                tries = 0
                while count < self.A and tries < 100 * self.N:
                    i = random.randint(1, self.N)
                    if X[i][d] != 0 or self.dayoff[i][d] == 1:
                        tries += 1
                        continue
                    if d > 1 and X[i][d - 1] == 4:
                        tries += 1
                        continue
                    X[i][d] = shift
                    if shift == 4 and d < self.D:
                        X[i][d + 1] = 0
                    count += 1
        return X

    def evaluate(self, X):
        penalty = 0
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1 and X[i][d] != 0:
                    penalty += 1000
                if d < self.D and X[i][d] == 4 and X[i][d + 1] != 0:
                    penalty += 1000
        for d in range(1, self.D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < self.A:
                    penalty += 100 * (self.A - count)
                elif count > self.B:
                    penalty += 100 * (count - self.B)

        night_shifts = [np.sum(X[i] == 4) for i in range(1, self.N + 1)]
        penalty += max(night_shifts) * 10
        return penalty

    def generate_neighbor(self, X):
        new_X = X.copy()
        i = random.randint(1, self.N)
        d = random.randint(1, self.D)
        if self.dayoff[i][d] == 1 or (d > 1 and new_X[i][d - 1] == 4):
            return new_X

        current_shift = new_X[i][d]
        new_shift = random.randint(0, 4)
        while new_shift == current_shift:
            new_shift = random.randint(0, 4)

        new_X[i][d] = new_shift
        if new_shift == 4 and d < self.D:
            new_X[i][d + 1] = 0
        return new_X

    def solve(self, max_iter=10000):
        T = 100.0
        T_min = 1e-2
        alpha = 0.98
        iteration = 0

        X = self.initialize_solution()
        best_X = X.copy()
        best_score = self.evaluate(X)

        while T > T_min and iteration < max_iter:
            for _ in range(100):
                neighbor = self.generate_neighbor(X)
                score = self.evaluate(neighbor)
                delta = score - best_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    X = neighbor
                    if score < best_score:
                        best_score = score
                        best_X = neighbor
            T *= alpha
            iteration += 1

        return best_X, best_score

    def save_output(self, file_path, X, exec_time):
        with open(file_path, 'w') as f:
            for i in range(1, self.N + 1):
                f.write(' '.join(map(str, map(int, X[i][1:self.D + 1]))) + '\n')
            f.write(f"Execution time: {exec_time:.2f} seconds\n")

    def print_output(self, X):
        for i in range(1, self.N + 1):
            print(' '.join(str(int(X[i][d])) for d in range(1, self.D + 1)))

    def print_max_night_shifts(self, X):
        night_shifts = [np.sum(X[i] == 4) for i in range(1, self.N + 1)]
        print(f"Max night shifts: {max(night_shifts)}")
