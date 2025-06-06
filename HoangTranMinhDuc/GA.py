import numpy as np
import time

class GA_C:
    def __init__(self):
        pass

    def initialize_random_solution(self, N, D, dayoff, A, B):  # Thêm self
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
                    X[i][j] = np.random.randint(0, 5)  # Sửa: 0-4
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
    
    def evaluate_fitness(self, X, N, D, A, B, dayoff):  # Thêm self
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
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):  # Thêm self
        selected_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return population[best_index]
    
    def crossover(self, parent1, parent2, crossover_rate=0.6):  # Thêm self
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        for i in range(1, len(parent1)):
            if np.random.rand() > crossover_rate:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        return child1, child2
    
    def mutate(self, X, N, D, dayoff, A, B, mutation_rate=0.1):  # Thêm self
        X = np.copy(X)  # Tạo bản sao
        # Bước đột biến
        for i in range(1, N + 1):
            for d in range(1, D + 1):
                if np.random.rand() < mutation_rate:
                    if dayoff[i][d] == 1:
                        X[i][d] = 0
                        continue
                    if d > 1 and X[i][d - 1] == 4:
                        X[i][d] = 0
                    else:
                        X[i][d] = np.random.randint(0, 5)  # Sửa: Bao gồm ca 4
                        if X[i][d] == 4 and d < D:
                            X[i][d + 1] = 0
        # Bước sửa lỗi ca đêm
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    X[i][d + 1] = 0
        # Bước điều chỉnh A <= số nhân viên mỗi ca <= B
        for d in range(1, D + 1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                while count < A:
                    available = [i for i in range(1, N + 1) 
                               if X[i][d] == 0 and dayoff[i][d] == 0 
                               and (d == 1 or X[i][d - 1] != 4)]
                    if not available:
                        break
                    i = np.random.choice(available)
                    X[i][d] = shift
                    if shift == 4 and d < D:  # Thêm: Nếu gán ca đêm, ngày tiếp theo phải nghỉ
                        X[i][d + 1] = 0
                    count += 1
                while count > B:
                    assigned = np.where(X[:, d] == shift)[0]
                    if not assigned.size:
                        break
                    i = np.random.choice(assigned)
                    X[i][d] = 0
                    count -= 1
        # Bước sửa lỗi ca đêm lần cuối
        for i in range(1, N + 1):
            for d in range(1, D):
                if X[i][d] == 4 and X[i][d + 1] != 0:
                    X[i][d + 1] = 0
        return X

    def solve(self, N, D, A, B, dayoff):
        population_size = 50
        generations = 20
        mutation_rate = 0.1
        population = [self.initialize_random_solution(N, D, dayoff, A, B) for _ in range(population_size)]

        for generation in range(generations):
            fitness_scores = np.array([self.evaluate_fitness(individual, N, D, A, B, dayoff) for individual in population])
            best_index = np.argmax(fitness_scores)
            best_solution = population[best_index]
            new_population = [best_solution]
            
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1, N, D, dayoff, A, B, mutation_rate))
                new_population.append(self.mutate(child2, N, D, dayoff, A, B, mutation_rate))

            population = new_population[:population_size]

        # Tính fitness cuối cùng
        fitness_scores = np.array([self.evaluate_fitness(individual, N, D, A, B, dayoff) for individual in population])
        best_index = np.argmax(fitness_scores)
        best_solution = population[best_index]

        # Chỉ output ma trận ca làm việc
        for i in range(1, N + 1):
            row = []
            for d in range(1, D + 1):
                row.append(str(int(best_solution[i][d])))
            print(" ".join(row))
        
        return best_solution

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
    ga = GA_C()
    ga.solve(N, D, A, B, dayoff)

if __name__ == "__main__":
    main()