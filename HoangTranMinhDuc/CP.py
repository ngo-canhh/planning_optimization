from ortools.sat.python import cp_model
import time

def solve_cp_pure(N, D, A, B, F):
    model = cp_model.CpModel()

    # Biến quyết định: X[i][d] = ca làm việc của nhân viên i vào ngày d
    X = [[model.NewIntVar(0, 4, f'X_{i}_{d}') for d in range(D)] for i in range(N)]

    # Ràng buộc 1: Ngày nghỉ phép
    for i in range(N):
        for day in F[i]:
            if 1 <= day <= D:
                model.Add(X[i][day-1] == 0)

    # Ràng buộc 2: Sau ca đêm phải nghỉ
    for i in range(N):
        for d in range(D-1):
            model.AddImplication(X[i][d] == 4, X[i][d+1] == 0)

    # Ràng buộc 3: Số nhân viên mỗi ca
    for d in range(D):
        for shift in range(1, 5):
            count = model.NewIntVar(0, N, f'count_{d}_{shift}')
            
            # Đếm số nhân viên trong ca này
            bool_vars = []
            for i in range(N):
                is_shift = model.NewBoolVar(f'is_{i}_{d}_{shift}')
                model.Add(X[i][d] == shift).OnlyEnforceIf(is_shift)
                model.Add(X[i][d] != shift).OnlyEnforceIf(is_shift.Not())
                bool_vars.append(is_shift)
            
            model.Add(count == sum(bool_vars))
            model.Add(count >= A)
            model.Add(count <= B)

    # Mục tiêu: Minimize max night shifts
    max_nights = model.NewIntVar(0, D, 'max_nights')
    
    for i in range(N):
        night_count = model.NewIntVar(0, D, f'nights_{i}')
        
        night_vars = []
        for d in range(D):
            is_night = model.NewBoolVar(f'night_{i}_{d}')
            model.Add(X[i][d] == 4).OnlyEnforceIf(is_night)
            model.Add(X[i][d] != 4).OnlyEnforceIf(is_night.Not())
            night_vars.append(is_night)
        
        model.Add(night_count == sum(night_vars))
        model.Add(max_nights >= night_count)
    
    model.Minimize(max_nights)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = False
    
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return [[solver.Value(X[i][d]) for d in range(D)] for i in range(N)]
    else:
        return None

def main():
    # Đọc input
    N, D, A, B = map(int, input().split())
    F = []
    
    for i in range(N):
        days = list(map(int, input().split()))
        valid_days = [d for d in days if d != -1 and 1 <= d <= D]
        F.append(set(valid_days))
    
    # Giải bài toán
    solution = solve_cp_pure(N, D, A, B, F)
    
    if solution:
        for row in solution:
            print(" ".join(map(str, row)))
    else:
        # Fallback solution
        for i in range(N):
            row = []
            for d in range(D):
                if (d+1) in F[i]:
                    row.append(0)
                else:
                    row.append((i % 4) + 1)
            print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()