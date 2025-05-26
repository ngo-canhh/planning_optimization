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
                available = [i for i in range(1, N + 1) if X[i][d] == 0 and dayoff[i][d] == 0 and (d == 1 or X[i][d - 1] != 4)]
                if not available:
                    break
                i = random.choice(available)
                X[i][d] = shift
                if shift == 4 and d < D:
                    X[i][d + 1] = 0  # Nghỉ sau ca đêm
                count += 1
            while count > A:
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

    # Đảm bảo các ca không bị thiếu
    for d in range(1, D + 1):
        for shift in range(1, 5):  # Các ca 1, 2, 3, 4
            count = np.sum(X[:, d] == shift)
            while count < A:
                available = [i for i in range(1, N + 1) if X[i][d] == 0 and dayoff[i][d] == 0 and (d == 1 or X[i][d - 1] != 4)]
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
    return X

class SimulatedAnnealingScheduler:
    def __init__(self, T0=5000, alpha=0.99, max_iterations=500, iterations_per_temp=50):
        
        self.T0 = T0
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.iterations_per_temp = iterations_per_temp
        self.best_solutions = []
        self.k = 5
        self.stagnation_counter = 0  # Đếm số lần không cải thiện
        self.stagnation_limit = 50  # Giới hạn trước khi khởi động lại

    def initialize_solution(self):
        """Khởi tạo giải pháp ban đầu"""
        X = np.zeros((self.N + 1, self.D + 1), dtype=int)
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1:
                    X[i][d] = 0

        for d in range(1, self.D + 1):
            for s in range(1, 5):
                available = [i for i in range(1, self.N + 1) if X[i][d] == 0 and self.dayoff[i][d] == 0 and (d == 1 or X[i][d-1] != 4)]
                np.random.shuffle(available)
                count = np.sum(X[:, d] == s)
                for i in available:
                    if count >= self.A:
                        break
                    X[i][d] = s
                    count += 1
        return X

    def evaluate_solution(self, X):
        """Đánh giá giải pháp"""
        penalty = 0
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1 and X[i][d] != 0:
                    penalty += 1000
        for d in range(1, self.D + 1):
            for s in range(1, 5):
                count = np.sum(X[:, d] == s)
                if count < self.A or count > self.B:
                    penalty += 1000
        for i in range(1, self.N + 1):
            for d in range(1, self.D):
                if X[i][d] == 4 and X[i][d+1] != 0:
                    penalty += 1000
        
        max_night_shifts = max(np.sum(X[i, 1:self.D + 1] == 4) for i in range(1, self.N + 1))
        return max_night_shifts + penalty

    def get_neighbor(self, X, aggressive=False):
        """Tạo giải pháp lân cận, có thể thay đổi nhiều ca nếu aggressive=True"""
        X_new = X.copy()
        if aggressive or np.random.random() < 0.3:  # Thay đổi lớn hơn nếu không cải thiện lâu
            # Hoán đổi ca của nhiều nhân viên trong một ngày
            d = np.random.randint(1, self.D + 1)
            num_swaps = np.random.randint(2, 4)  # Hoán đổi 2-4
            for _ in range(num_swaps):
                i1, i2 = np.random.choice(range(1, self.N + 1), 2, replace=False)
                if self.dayoff[i1][d] == 1 or self.dayoff[i2][d] == 1:
                    continue
                if d > 1 and (X_new[i1][d-1] == 4 or X_new[i2][d-1] == 4):
                    continue
                X_new[i1][d], X_new[i2][d] = X_new[i2][d], X_new[i1][d]

        else:
            # Thay đổi ca của một nhân viên, ưu tiên nhân viên có nhiều ca đêm
            night_shifts = [np.sum(X[i, 1:self.D + 1] == 4) for i in range(1, self.N + 1)]
            i = np.argmax(night_shifts) + 1  # Chọn nhân viên có nhiều ca đêm nhất
            d = np.random.randint(1, self.D + 1)
            if self.dayoff[i][d] == 1:
                return X_new

            new_shift = np.random.randint(0, 5)
            if d > 1 and X_new[i][d-1] == 4:
                new_shift = 0
            X_new[i][d] = new_shift
            if X_new[i][d] == 4 and d < self.D and self.dayoff[i][d+1] != 1:
                X_new[i][d+1] = 0

        #Điều chỉnh để thỏa mãn ràng buộc A, B
        for d in range(1, self.D + 1):
            for s in range(1, 5):
                count = np.sum(X_new[:, d] == s)
                while count < self.A:
                    avaiable = [i for i in range(1, self.N + 1) if X_new[i][d] == 0 and self.dayoff[i][d] == 0 and (d == 1 or X[i][d - 1] != 4)]
                    if avaiable:
                        i = np.random.choice(avaiable)
                        X_new[i][d] = s
                        count += 1
                    else:
                        break
                while count > self.B:
                    avaiable = [i for i in range(1, self.N + 1) if X_new[i][d] == s]
                    if avaiable:
                        i = np.random.choice(avaiable)
                        X_new[i][d] = 0
                        count -= 1
                    else:
                        break
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
            print(f"Iteration: {iteration} / {self.max_iterations}   and best_cost = {best_cost}")
            for _ in range(self.iterations_per_temp):
                # Tạo giải pháp lân cận, thay đổi mạnh nếu không cải thiện lâu
                aggressive = self.stagnation_counter > self.stagnation_limit
                X_new = self.get_neighbor(X, aggressive=aggressive)
                new_cost = self.evaluate_solution(X_new)
                delta_E = new_cost - cost

                if delta_E <= 0 or np.random.random() < math.exp(-delta_E / T):
                    X = X_new
                    cost = new_cost
                    self.stagnation_counter = 0  # Reset nếu có thay đổi
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

        return best_X, best_cost

