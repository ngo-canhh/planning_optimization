import numpy as np
import random
import time

class HC_C:
    def __init__(self):
        pass

    def initialize_random_solution(self, N, D, dayoff, A, B):
        X = np.zeros((N+1, D+1))
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j] == 1:
                    X[i][j] = 0
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j]:
                    continue
                if j > 1 and X[i][j - 1] == 4:
                    X[i][j] = 0
                else: 
                    X[i][j] = np.random.randint(0, 5)  # Sửa: bao gồm ca 4
        for d in range(1, D+1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                while count < A: 
                    available = [i for i in range(1, N+1) 
                               if X[i][d] == 0 and dayoff[i][d] == 0 
                               and (d == 1 or X[i][d-1] != 4)]  # Sửa: thêm điều kiện d==1
                    if not available:
                        break
                    i = np.random.choice(available)
                    X[i][d] = shift
                    count += 1
                while count > B:
                    assigned = np.where(X[:, d] == shift)[0]
                    if not assigned.size:
                        break
                    i = np.random.choice(assigned)
                    X[i][d] = 0
                    count -= 1
        return X
    
    def evaluate_fitness(self, X, N, D, A, B, dayoff):
        penalty = 0
        # check vi pham ngay nghi
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] and X[i][d] != 0:
                    penalty -= 1000

        # check vi pham so luong nguoi lam
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                if count < A:
                    penalty -= 100 * (A - count)
                elif count > B:
                    penalty -= 100 * (count - B)

        # check vi pham ca dem 
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    penalty -= 1000

        max_night_shift = 0
        for i in range(1, N + 1):
            count = np.sum(X[i] == 4)
            if count > max_night_shift:
                max_night_shift = count
        return -max_night_shift + penalty
    
    def get_neighbor(self, X, N, D, A, B, dayoff):
        X_new = np.copy(X)
        i = np.random.randint(1, N + 1)
        d = np.random.randint(1, D + 1)

        if dayoff[i][d] == 1: 
            return X_new
        if d > 1 and X[i][d - 1] == 4:
            return X_new
        else: 
            new_shift = np.random.randint(0, 5)
            while new_shift == X[i][d]:
                new_shift = np.random.randint(0, 5)
            X_new[i][d] = new_shift
            if new_shift == 4 and d < D:
                X_new[i][d + 1] = 0
        
        # check vi pham ngay nghi
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if dayoff[i][d] == 1:
                    X_new[i][d] = 0

        # check vi pham ca dem
        for i in range(1, N + 1):
            for d in range(1, D):
                if X_new[i][d] == 4 and X_new[i][d + 1] != 0:
                    X_new[i][d+1] = 0

        # check vi pham so nguoi lam
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(X_new[:, d] == shift)
                while count < A: 
                    available = [i for i in range(1, N + 1) 
                               if X_new[i][d] == 0 and dayoff[i][d] == 0 
                               and (d == 1 or X_new[i][d-1] != 4)]  # Sửa: thêm điều kiện d==1
                    if not available:
                        break
                    i = np.random.choice(available)
                    X_new[i][d] = shift
                    count += 1
                while count > B:
                    assigned = np.where(X_new[:, d] == shift)[0]
                    if not assigned.size:
                        break
                    i = np.random.choice(assigned)
                    X_new[i][d] = 0
                    count -= 1
                    
        # Sửa lỗi ca đêm lần cuối
        for i in range(1, N + 1):
            for d in range(1, D):
                if X_new[i][d] == 4 and X_new[i][d + 1] != 0:
                    X_new[i][d + 1] = 0 
        return X_new
    
    def solve(self, N, D, A, B, dayoff, max_iter=1000):
        X = self.initialize_random_solution(N, D, dayoff, A, B)
        best_fitness = self.evaluate_fitness(X, N, D, A, B, dayoff)
        best_X = np.copy(X)

        for iteration in range(max_iter):
            X_new = self.get_neighbor(X, N, D, A, B, dayoff)
            new_fitness = self.evaluate_fitness(X_new, N, D, A, B, dayoff)

            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_X = np.copy(X_new)
                X = np.copy(X_new)
                
        # Chỉ output ma trận ca làm việc
        for i in range(1, N + 1):
            row = []
            for d in range(1, D + 1):
                row.append(str(int(best_X[i][d])))
            print(" ".join(row))
        
        return best_X

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
    hc = HC_C()
    hc.solve(N, D, A, B, dayoff)

if __name__ == "__main__":
    main()