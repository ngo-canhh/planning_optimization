import numpy as np
from ortools.linear_solver import pywraplp
import time

class Linear:
    def __init__(self):
        pass

    def solve(self, N, D, A, B, dayoff):
        solver = pywraplp.Solver.CreateSolver('SCIP')

        #tao bien rang buoc
        x = {}
        for i in range(1, N+1):
            for j in range(1, D+1):
                for k in range(0, 5):
                    x[i, j, k] = solver.IntVar(0, 1, f'x[{i},{j},{k}]')

        #rang buoc: moi nhan vien moi ngay lam nhieu nhat 1 ca: 
        for i in range(1, N+1):
            for j in range(1, D+1):
                solver.Add(sum(x[i, j, k] for k in range(1, 5)) <= 1)

        #rang buoc neu ngay hom truoc lam ca dem ( ca 4) thi ngay hom sau phai nghi
        for i in range(1, N+1):
            for j in range(1, D):
                solver.Add(x[i, j, 4] + sum(x[i, j+1, k] for k in range(1, 5)) <= 1)

        #moi ca trong ngay co it nhat A nhan vien va nhieu nhat B nhan vien
        for j in range(1, D+1): # cac ngay
            for k in range(1, 5): #cac ca
                solver.Add(sum(x[i, j, k] for i in range(1, N+1)) >= A)
                solver.Add(sum(x[i, j, k] for i in range(1, N+1)) <= B)

        #rang buoc ngay nghi cho nhan vien
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j] == 1:
                    solver.Add(sum(x[i, j, k] for k in range(1, 5)) <= 0)

        #check thu cach  moi


        #bien muc tieu
        goal = solver.IntVar(0, D, 'goal')
        for i in range(1, N+1):
            solver.Add(goal >= sum(x[i, j, 4] for j in range(1, D+1)))
        start = time.time()
        solver.Minimize(goal)
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            # print(f'Test case {tc}', end=' ')
            print("Optimal solution found.", end=' ')
            print("Objective value:", int(solver.Objective().Value()))
            endtime = time.time()
            print("Execution time: {:.2f} seconds".format(endtime - start))
            best_solution = np.zeros((N+1, D+1))
            for i in range(1, N+1):
                for j in range(1, D+1):
                    for k in range(0, 5):
                        if x[i, j, k].solution_value() == 1:
                            best_solution[i][j] = k
            return best_solution
        else:
            print("The problem does not have an optimal solution.")
            return None

