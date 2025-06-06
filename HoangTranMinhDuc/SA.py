import numpy as np
import time
import math
import random

def adjust_schedule(X, N, D, A, B, dayoff):
    # Điều chỉnh lịch để mỗi ca có từ A đến B nhân viên
    for d in range(1, D + 1):
        for shift in range(1, 5):  # Các ca 1, 2, 3, 4
            count = np.sum(X[:, d] == shift)
            while count < A:
                available = [i for i in range(1, N + 1) 
                           if X[i][d] == 0 and dayoff[i][d] == 0 
                           and (d == 1 or X[i][d - 1] != 4)]
                if not available:
                    break
                i = random.choice(available)
                X[i][d] = shift
                if shift == 4 and d < D:
                    X[i][d + 1] = 0  # Nghỉ sau ca đêm
                count += 1
            while count > B:
                assigned = [i for i in range(1, N + 1) if X[i][d] == shift]
                if not assigned:
                    break
                i = random.choice(assigned)
                X[i][d] = 0
                count -= 1
    
    # Sửa lỗi ca đêm (ngày sau ca đêm phải nghỉ)
    for i in range(1, N + 1):
        for d in range(1, D):
            if X[i][d] == 4 and X[i][d + 1] != 0:
                X[i][d + 1] = 0
    
    return X

class SimulatedAnnealingScheduler:
    def __init__(self, T0=5000, alpha=0.99, max_iterations=500, iterations_per_temp=50):
        self.T0 = T0
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.iterations_per_temp = iterations_per_temp
        self.best_solutions = []
        self.k = 5
        self.stagnation_counter = 0
        self.stagnation_limit = 50

    def initialize_solution(self):
        """Khởi tạo giải pháp ban đầu"""
        X = np.zeros((self.N + 1, self.D + 1), dtype=int)
        
        # Gán ngày nghỉ phép trước
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1:
                    X[i][d] = 0

        # Gán ca cho từng ngày
        for d in range(1, self.D + 1):
            # Kiểm tra ràng buộc ca đêm
            if d > 1:
                for i in range(1, self.N + 1):
                    if X[i][d-1] == 4 and X[i][d] == 0 and self.dayoff[i][d] == 0:
                        X[i][d] = 0  # Bắt buộc nghỉ sau ca đêm
            
            # Gán ca cho từng shift
            for s in range(1, 5):
                available = [i for i in range(1, self.N + 1) 
                           if X[i][d] == 0 and self.dayoff[i][d] == 0 
                           and (d == 1 or X[i][d-1] != 4)]
                random.shuffle(available)
                
                count = 0
                for i in available:
                    if count >= self.A:
                        break
                    X[i][d] = s
                    if s == 4 and d < self.D:  # Nếu là ca đêm
                        if self.dayoff[i][d+1] == 0:  # Và ngày mai không nghỉ phép
                            X[i][d+1] = 0  # Thì nghỉ ngày mai
                    count += 1
        
        return X

    def evaluate_solution(self, X):
        """Đánh giá giải pháp"""
        penalty = 0
        
        # Penalty cho vi phạm ngày nghỉ phép
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1 and X[i][d] != 0:
                    penalty += 1000
        
        # Penalty cho vi phạm số lượng nhân viên mỗi ca
        for d in range(1, self.D + 1):
            for s in range(1, 5):
                count = np.sum(X[:, d] == s)
                if count < self.A:
                    penalty += 1000 * (self.A - count)
                elif count > self.B:
                    penalty += 1000 * (count - self.B)
        
        # Penalty cho vi phạm ca đêm
        for i in range(1, self.N + 1):
            for d in range(1, self.D):
                if X[i][d] == 4 and X[i][d+1] != 0:
                    penalty += 1000
        
        # Objective: max night shifts
        max_night_shifts = 0
        for i in range(1, self.N + 1):
            night_count = np.sum(X[i, 1:self.D + 1] == 4)
            max_night_shifts = max(max_night_shifts, night_count)
        
        return max_night_shifts + penalty

    def get_neighbor(self, X, aggressive=False):
        """Tạo giải pháp lân cận"""
        X_new = X.copy()
        
        if aggressive or np.random.random() < 0.3:
            # Thay đổi lớn: hoán đổi ca của nhiều nhân viên
            d = np.random.randint(1, self.D + 1)
            num_swaps = np.random.randint(2, min(4, self.N))
            
            for _ in range(num_swaps):
                if self.N < 2:
                    break
                i1, i2 = np.random.choice(range(1, self.N + 1), 2, replace=False)
                
                # Kiểm tra ràng buộc
                if (self.dayoff[i1][d] == 1 or self.dayoff[i2][d] == 1):
                    continue
                if d > 1 and (X_new[i1][d-1] == 4 or X_new[i2][d-1] == 4):
                    continue
                
                # Hoán đổi
                X_new[i1][d], X_new[i2][d] = X_new[i2][d], X_new[i1][d]
        else:
            # Thay đổi nhỏ: thay đổi ca của 1 nhân viên
            i = np.random.randint(1, self.N + 1)
            d = np.random.randint(1, self.D + 1)
            
            # Kiểm tra ràng buộc
            if self.dayoff[i][d] == 1:
                return X_new
            
            new_shift = np.random.randint(0, 5)
            
            # Nếu ngày trước làm ca đêm thì phải nghỉ
            if d > 1 and X_new[i][d-1] == 4:
                new_shift = 0
            
            X_new[i][d] = new_shift
            
            # Nếu gán ca đêm thì ngày sau phải nghỉ
            if new_shift == 4 and d < self.D and self.dayoff[i][d+1] != 1:
                X_new[i][d+1] = 0
        
        # Điều chỉnh để thỏa mãn ràng buộc A, B
        X_new = adjust_schedule(X_new, self.N, self.D, self.A, self.B, self.dayoff)
        
        return X_new

    def solve(self, N, D, A, B, dayoff):
        """Chạy thuật toán SA"""
        self.N = N
        self.D = D
        self.A = A
        self.B = B
        self.dayoff = dayoff
        
        X = self.initialize_solution()
        cost = self.evaluate_solution(X)
        best_X = X.copy()
        best_cost = cost
        T = self.T0

        for iteration in range(self.max_iterations):
            for _ in range(self.iterations_per_temp):
                # Tạo giải pháp lân cận
                aggressive = self.stagnation_counter > self.stagnation_limit
                X_new = self.get_neighbor(X, aggressive=aggressive)
                new_cost = self.evaluate_solution(X_new)
                delta_E = new_cost - cost

                if delta_E <= 0 or (T > 0 and np.random.random() < math.exp(-delta_E / T)):
                    X = X_new
                    cost = new_cost
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

                if cost < best_cost:
                    best_cost = cost
                    best_X = X.copy()
                    self.best_solutions.append((best_X.copy(), best_cost))
                    self.best_solutions = sorted(self.best_solutions, key=lambda x: x[1])[:self.k]

            T *= self.alpha
            if T < 0.01:
                break

        # Chỉ output ma trận ca làm việc
        for i in range(1, self.N + 1):
            row = []
            for d in range(1, self.D + 1):
                row.append(str(int(best_X[i][d])))
            print(" ".join(row))
        
        return best_X, best_cost

def main():
    # Đọc input
    N, D, A, B = map(int, input().split())
    dayoff = np.zeros((N + 1, D + 1), dtype=int)
    
    for i in range(1, N + 1):
        days = list(map(int, input().split()))
        for day in days:
            if day == -1:
                break
            if 1 <= day <= D:
                dayoff[i][day] = 1
    
    # Giải bài toán
    sa = SimulatedAnnealingScheduler(
        T0=1000,
        alpha=0.95,
        max_iterations=100,
        iterations_per_temp=10
    )
    sa.solve(N, D, A, B, dayoff)

if __name__ == "__main__":
    main()