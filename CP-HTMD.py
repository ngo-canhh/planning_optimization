from ortools.sat.python import cp_model


def schedule_shifts(N, D, A, B, F):
    model = cp_model.CpModel()

    X = [[model.NewIntVar(0, 4, f'X_{i}_{d}') for d in range(D)] for i in range(N)]

    # Constraint 1: Respect day-off constraints
    for i in range(N):
        for d in F[i]:
            model.Add(X[i][d - 1] == 0)

    # Constraint 2: If a staff works night shift (4), they rest the next day
    for i in range(N):
        for d in range(D - 1):
            is_night_shift = model.NewBoolVar(f'is_night_shift_{i}_{d}')
            is_next_day_off = model.NewBoolVar(f'is_next_day_off_{i}_{d}')

            # Ràng buộc: is_night_shift đúng khi X[i][d] == 4
            model.Add(X[i][d] == 4).OnlyEnforceIf(is_night_shift)
            model.Add(X[i][d] != 4).OnlyEnforceIf(is_night_shift.Not())

            # Ràng buộc: is_next_day_off đúng khi X[i][d+1] == 0
            model.Add(X[i][d + 1] == 0).OnlyEnforceIf(is_next_day_off)
            model.Add(X[i][d + 1] != 0).OnlyEnforceIf(is_next_day_off.Not())

            model.AddImplication(is_night_shift, is_next_day_off)

    # Constraint 3: Maintain A-B staff in each shift per day
    for d in range(D):
        for k in range(1, 5):  # Shift 1 to 4
            is_on_shift = [model.NewBoolVar(f'is_on_shift_{i}_{d}_{k}') for i in range(N)]
            for i in range(N):
                model.Add(X[i][d] == k).OnlyEnforceIf(is_on_shift[i])
                model.Add(X[i][d] != k).OnlyEnforceIf(is_on_shift[i].Not())
            model.AddLinearConstraint(sum(is_on_shift), A, B)


    max_night_shifts = model.NewIntVar(0, D, 'max_night_shifts')
    for i in range(N):
        night_shifts = [model.NewBoolVar(f'night_shift_{i}_{d}') for d in range(D)]
        for d in range(D):
            model.Add(X[i][d] == 4).OnlyEnforceIf(night_shifts[d])
            model.Add(X[i][d] != 4).OnlyEnforceIf(night_shifts[d].Not())
        model.Add(max_night_shifts >= sum(night_shifts))
    model.Minimize(max_night_shifts)


    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print("\n Optimal")
        result = [[solver.Value(X[i][d]) for d in range(D)] for i in range(N)]
        return result, solver.ObjectiveValue()
    else: 
        if status == cp_model.FEASIBLE:
            print("\n Feasible")
            result = [[solver.Value(X[i][d]) for d in range(D)] for i in range(N)]
            return result, solver.ObjectiveValue()
        else:
            return None, None
def print_schedule(schedule):
    if schedule is None:
        print("Không tìm được lời giải.")
    else:
        for row in schedule:
            print(" ".join(map(str, row)))

if __name__ == "__main__":

    N, D, A, B = map(int, input().split())
    F = []

    for i in range(N):
        days = list(map(int, input().split()))
        F.append(set(days[:-1]))


    schedule, max_night_shifts = schedule_shifts(N, D, A, B, F)

    print("\nMột lời giải khả thi:")
    print_schedule(schedule)
    if max_night_shifts is not None:
        print(f"\nSố ca đêm tối đa: {int(max_night_shifts)}")

