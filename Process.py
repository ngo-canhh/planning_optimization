from Linear_C import Linear
from Heuristic_C import Heuristic_C
from GA_C import GA_C
from ACO_C import AntColony
from HC_C import HC_C
from SA_C import SimulatedAnnealingScheduler
import numpy as np
import random
import time
import copy

def read_input_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    N, D, A, B = map(int, lines[0].split())
    dayoff = np.zeros((N+1, D+1))
    for i in range(1, N+1):
        numbers = list(map(int, lines[i].split()))
        for j in range(0, len(numbers) - 1):
            dayoff[i][numbers[j]] = 1
    return N, D, A, B, dayoff

if __name__ == "__main__":
    list_file = [1, 2, 10, 4]
    file = 'Testcase/tc'
    for i in list_file:
        filename = file + str(i) + '.txt'
        N, D, A, B, dayoff = read_input_from_file(filename)
        solver = SimulatedAnnealingScheduler()
        X = solver.solve(N, D, A, B, dayoff)
        print(f"File {i}:")

