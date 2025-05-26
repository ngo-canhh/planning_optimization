import random
import copy
import math
# Đọc dữ liệu đầu vào từ người dùng
def read_input():
    N, D, A, B = map(int, input().split())  # Số nhân viên, số ngày, số nhân viên tối thiểu & tối đa cho mỗi ca
    f = []  # Danh sách ngày nghỉ cố định của mỗi nhân viên
    rest_day = [[] for _ in range(N)]  # Danh sách ngày nghỉ tự động phát sinh do làm ca đêm
    for _ in range(N):
        days = list(map(int, input().split()))  # Nhập ngày nghỉ, -1 nếu không nghỉ
        f.append([d for d in days if d != -1])
    return N, D, A, B, f, rest_day

def init_sol(N, D, A, B, f, rest_day):
    x = [[0 for _ in range(D)] for _ in range(N)]  # Ma trận phân công: x[i][d] = ca làm của nhân viên i ngày d
    night_count = [0] * N  # Đếm số ca đêm của mỗi nhân viên
    count_staffs_of_shift_k_per_day = [[0] * 5 for _ in range(D)]  # Đếm số nhân viên mỗi ca (1–4) cho từng ngày

    # Gán 0 cho những ngày nhân viên đã đăng ký nghỉ
    for i in range(N):
        for d in f[i]:
            x[i][d - 1] = 0

    # Gán ca làm theo vòng luân phiên 1–4 cho nhân viên chưa nghỉ
    for d in range(D):
        u = 1  # Bắt đầu từ ca 1
        for i in range(N):
            if (d + 1) in rest_day[i] or (d + 1) in f[i]:
                continue  # Bỏ qua nếu ngày nghỉ hoặc ngày nghỉ bù
            if u > 4:
                u = 1  # Quay lại ca 1
            x[i][d] = u
            count_staffs_of_shift_k_per_day[d][u] += 1
            # Nếu làm ca đêm, thêm ngày nghỉ kế tiếp vào danh sách nghỉ
            if u == 4 and (d + 2) not in f[i]:
                rest_day[i].append(d + 2)
            u += 1
    return x, night_count, count_staffs_of_shift_k_per_day

def is_valid_solution(x, N, D, A, B, f, rest_day):
    for d in range(D): # kiểm tra số nv mỗi ca trong mỗi ngày có thuộc [A, B] ko
        for k in range(1, 5):
            cnt = sum(1 for i in range(N) if x[i][d] == k)
            if cnt < A or cnt > B:
                return False 
    
    for i in range(N):
        for d in range(D):
            if d + 1 in f[i] or d + 1 in rest_day[i]:
                if x[i][d] != 0:
                    return False # check nếu nv đi làm vào ngày nghỉ cố định
            if d > 0 and x[i][d-1] == 4 and x[i][d] != 0:
                return False # check nếu hôm qua lm ca đêm nhưng vẫn lm hôm nay
            
    return True

def calculate_cost(N, D, x):
    night_count = [0] * len(x)
    for i in range(N):
        for d in range(D):
            if x[i][d] == 4:
                night_count[i] += 1
    return max(night_count) # đưa ra số ca đêm max của 1 nv

def create_neighbor(x, N, D, rest_day): # sinh hàng xóm
    x_new = copy.deepcopy(x)
    rest_day_new = copy.deepcopy(rest_day)
    while not is_valid_solution(x_new, N, D, A, B, f, rest_day_new):
        i = random.randint(0, N - 1)
        d = random.randint(0, D - 1)
        ca_lam_cu = x_new[i][d]
        ca_lam_moi = random.choice([j for j in range(5) if j != ca_lam_cu and j not in f[i]]) 
        if ca_lam_moi == 4:
            if d < D - 1: 
                if d + 2 not in f[i]:
                    x_new[i][d + 1] = 0
                    rest_day_new[i].append(d + 2) # thêm ngày hôm sau vào rest_day
        x_new[i][d] = ca_lam_moi
    return x_new, rest_day_new

def s_a(N, D, A, B, f, rest_day, night_count):
    x, night_count, count_staffs_of_shift_k_per_day = init_sol(N, D, A, B, f, rest_day)
    best_x = copy.deepcopy(x)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        i = 1
        while i <= 100:
            x_new, rest_day_new = create_neighbor(x, N, D, rest_day)
            new_cost = calculate_cost(N, D, x_new)
            old_cost = calculate_cost(N, D, x)
            delta = new_cost - old_cost
            if delta < 0 or random.uniform(0, 1) < pow(2.71828, -delta / T):
                x = copy.deepcopy(x_new)
                old_cost = new_cost
                if new_cost < calculate_cost(N, D, best_x):
                    best_x = copy.deepcopy(x_new)
            i += 1
        T *= alpha
    return best_x

def print_sol(x, N, D):
    for row in x:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    N, D, A, B, f, rest_day = read_input()
    x, night_count, count_staffs_of_shift_k_per_day = init_sol(N, D, A, B, f, rest_day)
    best_x = s_a(N, D, A, B, f, rest_day, night_count)
    print_sol(best_x, N, D)
    print(calculate_cost(best_x, N, D))