from ortools.linear_solver import pywraplp
import numpy as np

# def read_input_from_file(filename):
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#     N, D, A, B = map(int, lines[0].split())
#     dayoff = np.zeros((N+1, D+1))
#     for i in range(1, N+1):
#         numbers = list(map(int, lines[i].split()))
#         for j in range(0, len(numbers) - 1):
#             dayoff[i][numbers[j]] = 1
#     return N, D, A, B, dayoff

# N, D, A, B, dayoff = read_input_from_file('test.txt')

#Nhap input bang tay
N, D, A, B = map(int, input().split())
dayoff = np.zeros((N+1, D+1))
for i in range(1, N+1):
    numbers = list(map(int, input().split()))
    for j in range(0, len(numbers) - 1):
        dayoff[i][numbers[j]] = 1


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

solver.Minimize(goal)
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    # print(int(solver.Objective().Value()))
    result = np.zeros((N+1, D+1))
    for i in range(1, N+1):
        for j in range(1, D+1):
            for k in range(0, 5):
                if x[i, j, k].solution_value() == 1:
                    result[i][j] = k
    # in ket qua
    # with open('output.txt', 'w') as file:
    #     file.write(str(int(solver.Objective().Value())) + '\n')
    #     for i in range(1, N+1):
    #         file.write(' '.join([str(int(x)) for x in result[i][1:]]) + '\n')
    for i in range(1, N+1):
        print(' '.join([str(int(x)) for x in result[i][1:]]))
    #test case
    """
8 6 1 3
 1 -1
 3 -1
 4 -1
 5 -1
 2 4 -1
 -1
 -1
 3 -1
    """
