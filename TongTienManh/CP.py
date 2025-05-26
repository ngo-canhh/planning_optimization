from ortools.sat.python import cp_model

def read_input():
    N, D, A, B = map(int, input().split())  # Số nhân viên, số ngày, số nhân viên tối thiểu & tối đa cho mỗi ca
    f = []  # Danh sách ngày nghỉ cố định của mỗi nhân viên
    rest_day = [[] for _ in range(N)]  # Danh sách ngày nghỉ tự động phát sinh do làm ca đêm
    for _ in range(N):
        days = list(map(int, input().split()))  # Nhập ngày nghỉ, -1 nếu không nghỉ
        f.append([d for d in days if d != -1])
    return N, D, A, B, f, rest_day

def cp(N, D, A, B, F):
    # N: số nhân viên
    # D: số ngày
    # A: số nhân viên tối thiểu cho mỗi ca
    # B: số nhân viên tối đa cho mỗi ca
    # F: danh sách ngày nghỉ cố định của mỗi nhân viên
    model = cp_model.CpModel()
    x = [[model.NewIntVar(0, 4, 'x{i}_{d}') for d in range(D)] for i in range(N)]  # Ma trận phân công: x[i][d] = ca làm của nhân viên i ngày d

    # Ràng buộc 1: Nhân viên nghỉ vào ngày nghỉ cố định
    for i in range(N):
        for d in F[i]:
            model.Add(x[i][d - 1] == 0)

    # Ràng buộc 2: Nếu làm ca đêm hôm trước thì hôm sau phải nghỉ
    for i in range(N):
        for d in range(D - 1):
            is_night = model.NewBoolVar(f'is_night_{i}_{d}')
            model.Add(x[i][d] == 4).OnlyEnforceIf(is_night)
            model.Add(x[i][d] != 4).OnlyEnforceIf(is_night.Not())
            model.Add(x[i][d + 1] == 0).OnlyEnforceIf(is_night)
    
    # Ràng buộc 3: Số nhân viên mỗi ca trong mỗi ngày phải nằm trong khoảng [A, B]
    for d in range(D):
        for k in range(1, 5):  # Shift 1 to 4
            is_on_shift = [model.NewBoolVar(f'is_on_shift_{i}_{d}_{k}') for i in range(N)]
            for i in range(N):
                model.Add(x[i][d] == k).OnlyEnforceIf(is_on_shift[i])
                model.Add(x[i][d] != k).OnlyEnforceIf(is_on_shift[i].Not())
            model.Add(sum(is_on_shift) >= A)
            model.Add(sum(is_on_shift) <= B)

    # Ràng buộc 4: Tổng số ca đêm tối đa của mỗi nhân viên <= y
    y = model.NewIntVar(0, D, 'y')
    for i in range(N):
        night_shifts = [model.NewBoolVar(f'night_shift_{i}_{d}') for d in range(D)]
        for d in range(D):
            model.Add(x[i][d] == 4).OnlyEnforceIf(night_shifts[d])
            model.Add(x[i][d] != 4).OnlyEnforceIf(night_shifts[d].Not())
        model.Add(y >= sum(night_shifts))
    
    model.Minimize(y)  

    # Solve the model
    
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result = [[solver.Value(x[i][d]) for d in range(D)] for i in range(N)]
        return result, solver.ObjectiveValue()
    else:
        return None, None
    
def print_sol(x):
    if x is None:
        print("No solution found.")
    else:
        for row in x:
            print(" ".join(map(str, row)))

if __name__ == "__main__":
    N, D, A, B, f, rest_day = read_input()
    x, min_y = cp(N, D, A, B, f)
    print_sol(x)
    #if min_y is not None:
        #print(f"Min y (số ca đêm tối đa của 1 nhân viên): {min_y}")
