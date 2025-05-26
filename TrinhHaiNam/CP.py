"""Example of a simple nurse scheduling problem."""
from ortools.sat.python import cp_model

def nhap_input():
    
    print("nhập dữ liệu: ")
    N, D, A, B = map(int, input().split())
    
    
    days_off = []
    for _ in range(N):
        day_list = list(map(int, input().split()))
        days = []
        for day in day_list:
            if day == -1:
                break
            days.append(day - 1)
        days_off.append(days)
    
    return N, D, A, B, days_off

def main():
    
    N, D, A, B, days_off = nhap_input()
    num_shifts = 4  
    staffs = range(N)
    all_shifts = range(1, num_shifts + 1)  
    all_days = range(D)
    
    model = cp_model.CpModel()
    
    
    shifts = {}
    for n in staffs:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.new_bool_var(f"shift_n{n}_d{d}_s{s}")
    
     
    X = {}
    for n in staffs:
        for d in all_days:
            X[(n, d)] = model.new_int_var(0, 4, f"X_{n}_{d}")
            
            model.add(X[(n, d)] == sum(s * shifts[(n, d, s)] for s in all_shifts))

    # Ràng buộc 1: Mỗi ca phải có ít nhất A nhân viên và tối đa B nhân viên
    for d in all_days:
        for s in all_shifts:
            model.add(A <= sum(shifts[(n, d, s)] for n in staffs))
            model.add(sum(shifts[(n, d, s)] for n in staffs) <= B)

    # Ràng buộc 2: Mỗi y tá làm tối đa một ca mỗi ngày
    for n in staffs:
        for d in all_days:
            model.add_at_most_one(shifts[(n, d, s)] for s in all_shifts)

    # Ràng buộc 3: Nghỉ sau khi làm ca đêm 
    for n in staffs:
        for d in range(D - 1):
            model.add(shifts[(n, d, 4)] + sum(shifts[(n, d + 1, s)] for s in all_shifts) <= 1)

    # Ràng buộc 4: Ngày nghỉ của nhân viên
    for n in staffs:
        for d in days_off[n]:
            for s in all_shifts:
                model.add(shifts[(n, d, s)] == 0)
    
    # Hàm mục tiêu: Giảm thiểu số ca đêm được phân công cho mỗi nhân viên
    max_night_shifts = model.new_int_var(0, D, "max_night_shifts")   
    for n in staffs:
        model.add(sum(shifts[(n, d, 4)] for d in all_days) <= max_night_shifts) 
    model.minimize(max_night_shifts)
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    
    
    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Số ca đêm tối đa mà một nhân viên phải làm: {solver.value(max_night_shifts)}")
        
        if( status ==4 ):
            print("FEASIBLE")
        elif(status==3):
            print("OPtimal")
        for n in staffs:
            row = []
            for d in all_days:
                value = 0
                for s in all_shifts:
                    if solver.value(shifts[(n, d, s)]) > 0.5:
                        value = s
                row.append(str(value))
            print(" ".join(row))

if __name__ == "__main__":
    main()