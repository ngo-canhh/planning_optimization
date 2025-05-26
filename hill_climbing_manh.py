import random
import copy
import math

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

def evaluate(x, N, D): # hàm đánh giá 1 phương án
    max_nightcount = 0
    for i in range(N):
        cnt = 0
        for d in range(D):
            if x[i][d] == 4:
                cnt += 1
        max_nightcount = max(max_nightcount, cnt)
    return max_nightcount

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

def get_neighbor(x, N, D, A, B, f, rest_day):
    best_x = None
    best_score = evaluate(x, N, D)
    for d in range(D):
        for i in range(N):
            for j in range(i + 1, N):
                if x[i][d] not in f[i] and x[j][d] not in f[j]:
                    if x[i][d] != x[j][d]:
                        new_x = copy.deepcopy(x)
                        new_x[i][d], new_x[j][d] = new_x[j][d], new_x[i][d]
                        if is_valid_solution(new_x, N, D, A, B, f, rest_day):
                            score = evaluate(new_x, N, D)
                            if score < best_score:
                                best_score = score
                                best_x = new_x
    return best_x 

def hill_climbing(N, D, A, B, f, rest_day):
    x, night_count, count_staffs_of_shift_k_per_day = init_sol(N, D, A, B, f, rest_day)
    current_score = evaluate(x, N, D)
    while True:
        neighbor = get_neighbor(x, N, D, A, B, f, rest_day)
        if neighbor is None:
            return x
        
        neighbor_score = evaluate(neighbor, N, D)
        if neighbor_score < current_score:
            x = neighbor
            current_score = neighbor_score
        else:
            return x

def print_sol(x, N, D):
    for row in x:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    N, D, A, B, f, rest_day = read_input()
    x, night_count, count_staffs_of_shift_k_per_day = init_sol(N, D, A, B, f, rest_day)
    best_x = hill_climbing(N, D, A, B, f, rest_day)
    print_sol(best_x, N, D)




    