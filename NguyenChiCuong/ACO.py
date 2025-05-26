import numpy as np
import random
import time

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

class AntColony:
    def __init__(self,  num_ants=15, max_iterations=30, alpha=1, beta=2, rho=0.1, Q=1):
        
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha # Trọng số pheromone
        self.beta = beta # Trọng số heuristic
        self.rho = rho # Hệ số bay hơi pheromone
        self.Q = Q # Hệ số pheromone
        

    def initialize_pheromone(self):
        pheromone = {}
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                for s in range(5):  # 0: nghỉ, 1-4: các ca
                    pheromone[(i, d, s)] = 0.1
        return pheromone

    def heuristic(self, i, d, s, X):
        # Giá trị heuristic
        if self.dayoff[i][d] == 1 and s != 0:
            return 0
        if d > 1 and X[i][d - 1] == 4 and s != 0:
            return 0
        if s == 4:
            return 0.1  # Ưu tiên ít ca đêm
        return 1

    def construct_solution(self):
        X = np.zeros((self.N + 1, self.D + 1))
        for i in range(1, self.N + 1):
            for d in range(1, self.D + 1):
                if self.dayoff[i][d] == 1:
                    X[i][d] = 0
                    continue
                allowed_shifts = [0] if (d > 1 and X[i][d - 1] == 4) else range(5)
                probs = []
                for s in allowed_shifts:
                    tau = self.pheromone[(i, d, s)]
                    eta = self.heuristic(i, d, s, X)
                    prob = (tau ** self.alpha) * (eta ** self.beta)
                    probs.append(prob)
                probs = np.array(probs) / np.sum(probs)
                s = np.random.choice(allowed_shifts, p=probs)
                X[i][d] = s
        X = adjust_schedule(X, self.N, self.D, self.A, self.B, self.dayoff)
        return X

    def evaluate_fitness(self, X):
        # Tối thiểu hóa số ca đêm tối đa của một nhân viên
        max_night = 0
        for i in range(1, self.N + 1):
            night_count = np.sum(X[i] == 4)
            max_night = max(max_night, night_count)
        return -max_night

    def update_pheromone(self, solutions, fitness_scores):
        # Bay hơi pheromone
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)
        # Cộng thêm pheromone từ các lời giải
        for k in range(self.num_ants):
            X = solutions[k]
            fitness = fitness_scores[k]
            cost = -fitness  # Chuyển fitness thành cost
            delta_tau = self.Q / (cost + 1e-10)  # Tránh chia cho 0
            for i in range(1, self.N + 1):
                for d in range(1, self.D + 1):
                    s = int(X[i][d])
                    self.pheromone[(i, d, s)] += delta_tau

    def solve(self, N, D, A, B, dayoff):
        start = time.time()
        self.N = N  # Số nhân viên
        self.D = D  # Số ngày
        self.A = A  # Số nhân viên tối thiểu mỗi ca
        self.B = B  # Số nhân viên tối đa mỗi ca
        self.dayoff = dayoff  # Ma trận ngày nghỉ
        self.pheromone = self.initialize_pheromone()
        best_solution = None
        best_fitness = float('-inf')
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            if iteration > 0:
                # In gia tri toi uu hien tai
                print(f"Best fitness: {best_fitness}")
            solutions = []
            fitness_scores = []
            for _ in range(self.num_ants):
                X = self.construct_solution()
                fitness = self.evaluate_fitness(X)
                solutions.append(X)
                fitness_scores.append(fitness)
                if fitness > best_fitness:
                    best_solution = X
                    best_fitness = fitness
            self.update_pheromone(solutions, fitness_scores)
        def check(X, N, D, A, B, dayoff):
            for i in range(1, N + 1):
                for d in range(1, D + 1):
                    if dayoff[i][d] == 1 and X[i][d] != 0:
                        return False
            for d in range(1, D + 1):
                for shift in range(1, 5):
                    count = np.sum(X[:, d] == shift)
                    if count < A or count > B:
                        return False
            for i in range(1, N + 1):
                for d in range(1, D):
                    if X[i][d] == 4 and X[i][d + 1] != 0:
                        return False
            return True
        if check(best_solution, self.N, self.D, self.A, self.B, self.dayoff):
            print("Best solution found:")
            print(f"Số ca đêm tối đa của một nhân viên: {-best_fitness}")
            print("Thơig gian thực hiện:", time.time() - start)
            return best_solution
        else:
            print("No valid solution found.")
        


    