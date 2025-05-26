import numpy as np
import time

class GA_C:
    def __init__(self):
        pass

    def initialize_random_solution(N, D, dayoff, A, B):
        X = np.zeros((N+1, D+1))
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j] == 1:
                    X[i][j] = 0
        for i in range(1, N+1):
            for j in range(1, D+1):
                if dayoff[i][j]:
                    continue
                if j> 1 and X[i][j - 1] == 4:
                    X[i][j] = 0
                else: 
                    X[i][j] = np.random.randint(0, 4)
        for d in range(1, D+1):
            for shift in range(1, 5):
                count = np.sum(X[:, d] == shift)
                while count < A: 
                    available = [i for i in range(1, N+1) if X[i][d] == 0 and dayoff[i][d] == 0 and X[i][d-1] != 4]
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
    
    def evaluate_fitness(X, N, D, A, B, dayoff):
        penalty = 0
        # check vi pham ngay nghi
        for i in range(1, N + 1):
            for d in range(1, D +1):
                if dayoff[i][d] and X[i][d] != 0:
                    penalty -= 1000

        # check vi pham so luong nguoi lam
        for d in range(1, D +1):
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
    
    def tournament_selection(population, fitness_scores, tournament_size=3):
        selected_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return population[best_index]
    
    def crossover(parent1, parent2, crossover_rate=0.6):
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        for i in range(1, len(parent1)):
            if np.random.rand() > crossover_rate:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        return child1, child2
    
    def mutate(X, N, D, dayoff, A, B, mutation_rate=0.1):
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
                    available = [i for i in range(1, N + 1) if X[i][d] == 0 and dayoff[i][d] == 0 and (d == 1 or X[i][d - 1] != 4)]
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
        starttime = time.time()
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

            if generation % 10 == 0:
                best_fitness = max(fitness_scores)
                print(f"Generation {generation}: Best Fitness = {best_fitness}")
        best_index = np.argmax(fitness_scores)
        best_solution = population[best_index]

        # Kiểm tra tính hợp lệ của giải pháp tốt nhất
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
        if not check(best_solution, N, D, A, B, dayoff):
            print("No valid solution found.")
            return None
        print("Best solution found.")
        print("Best fitness:", self.evaluate_fitness(best_solution, N, D, A, B, dayoff))
        print("Best solution schedule:")
        for i in range(1, N + 1):
            print(f"Staff {i}: ", end='')
            for d in range(1, D + 1):
                print(int(best_solution[i][d]), end=' ')
            print()
        endtime = time.time()
        print("Execution time: {:.2f} seconds".format(endtime - starttime))
        return best_solution