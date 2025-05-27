import numpy as np
import random
import math
import time

def read_input_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    N, D, A, B = map(int, lines[0].split())
    dayoff = np.zeros((N + 1, D + 1), dtype=int)

    for i in range(1, N + 1):
        days = list(map(int, lines[i].split()))
        for d in days[:-1]:
            dayoff[i][d] = 1
    return N, D, A, B, dayoff



def initialize_solution(N, D, A, B, dayoff):
    X = np.zeros((N + 1, D + 1), dtype=int)

    for d in range(1, D + 1):
        for shift in range(1, 5):
            count = 0
            tries = 0
            while count < A and tries < 100 * N:
                i = random.randint(1, N)
                if X[i][d] != 0 or dayoff[i][d] == 1:
                    tries += 1
                    continue
                if d > 1 and X[i][d - 1] == 4:
                    tries += 1
                    continue
                X[i][d] = shift
                if shift == 4 and d < D:
                    X[i][d + 1] = 0
                count += 1
    return X

def evaluate(X, N, D, A, B, dayoff):
    penalty = 0

    for i in range(1, N + 1):
        for d in range(1, D + 1):
            if dayoff[i][d] == 1 and X[i][d] != 0:
                penalty += 1000
            if d < D and X[i][d] == 4 and X[i][d + 1] != 0:
                penalty += 1000

    for d in range(1, D + 1):
        for shift in range(1, 5):
            count = np.sum(X[:, d] == shift)
            if count < A:
                penalty += 100 * (A - count)
            elif count > B:
                penalty += 100 * (count - B)

    # Tối thiểu hóa ca đêm
    night_shifts = [np.sum(X[i] == 4) for i in range(1, N + 1)]
    penalty += max(night_shifts) * 10

    return penalty

def generate_neighbor(X, N, D, dayoff):
    new_X = X.copy()
    i = random.randint(1, N)
    d = random.randint(1, D)
    if dayoff[i][d] == 1 or (d > 1 and new_X[i][d - 1] == 4):
        return new_X

    current_shift = new_X[i][d]
    new_shift = random.randint(0, 4)
    while new_shift == current_shift:
        new_shift = random.randint(0, 4)

    new_X[i][d] = new_shift
    if new_shift == 4 and d < D:
        new_X[i][d + 1] = 0
    return new_X

def simulated_annealing(N, D, A, B, dayoff, max_iter=10000):
    T = 100.0
    T_min = 1e-2
    alpha = 0.98
    iteration = 0

    X = initialize_solution(N, D, A, B, dayoff)
    best_X = X.copy()
    best_score = evaluate(X, N, D, A, B, dayoff)

    while T > T_min and iteration < max_iter:
        for _ in range(100):
            neighbor = generate_neighbor(X, N, D, dayoff)
            score = evaluate(neighbor, N, D, A, B, dayoff)
            delta = score - best_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                X = neighbor
                if score < best_score:
                    best_score = score
                    best_X = neighbor
        T *= alpha
        iteration += 1

    return best_X, best_score

def save_output(file_path, X, N, D, exec_time):
    with open(file_path, 'w') as f:
        for i in range(1, N + 1):
            f.write(' '.join(map(str, map(int, X[i][1:D + 1]))) + '\n')
        f.write(f"Execution time: {exec_time:.2f} seconds\n")
def print_max_night_shifts(X, N):
    night_shifts = [np.sum(X[i] == 4) for i in range(1, N + 1)]
    max_night = max(night_shifts)
    print(f"Max night shifts: {max_night} ")

# Main Execution
if __name__ == "__main__":
    file_base = "planning_optimization/Evaluate/Testcase/tc"
    
    
    tc = int(input("test case: "))
    input_file = f"{file_base}{tc}.txt"
    
    start = time.time()

    N, D, A, B, dayoff = read_input_from_file(input_file)
    best_X, best_score = simulated_annealing(N, D, A, B, dayoff)

    end = time.time()
    
    print(f"Test case {tc} done. Cost: {best_score}, Time: {end - start:.2f}s")
    print_max_night_shifts(best_X, N)
