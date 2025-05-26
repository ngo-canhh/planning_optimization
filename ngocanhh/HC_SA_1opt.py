import random
import time
import math
import copy
import os

class StaffRosteringSolver:
    def __init__(self,
                num_restarts=5,
                max_iterations_per_restart=200,
                max_no_improvement_per_restart=50,
                num_neighbors_to_explore_per_step=20,
                use_simulated_annealing=True,
                initial_temperature=100.0,
                cooling_rate=0.99,
                min_temperature=0.1,):
        
        # Tham số thuật toán
        self.num_restarts = num_restarts
        self.max_iterations_per_restart = max_iterations_per_restart
        self.max_no_improvement_per_restart = max_no_improvement_per_restart
        self.num_neighbors_to_explore_per_step = num_neighbors_to_explore_per_step
        
        self.use_simulated_annealing = use_simulated_annealing
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

        # Tham số bài toán
        self.N = None
        self.D = None
        self.A = None
        self.B = None
        self.IS_PRE_ASSIGNED_OFF = None # (N, D) - boolean

    def _create_initial_solution(self):
        # individual[staff_idx][day_idx]
        individual = [[-1 for _ in range(self.D)] for _ in range(self.N)]

        # Gán ngày nghỉ
        for d in range(self.D):
            for i in range(self.N):
                if self.IS_PRE_ASSIGNED_OFF[i][d]:
                    individual[i][d] = 0

        # Xếp lịch cho từng ngày
        for d in range(self.D):

            # Nếu ngày hôm qua làm đêm -> hôm nay nghỉ
            if d > 0:
                for i in range(self.N):
                    if individual[i][d - 1] == 4:
                        if individual[i][d] == -1:
                            individual[i][d] = 0
        
            # Lấy danh sách nhân viên có thể làm
            available_staff = [i for i in range(self.N) if individual[i][d] == -1]
            # Xáo trộn để đa dạng hoá
            random.shuffle(available_staff)

            # Lấy danh sách nhân viên có thể làm đêm
            night_shift_candidates = available_staff.copy()
            num_to_assign_night = min(self.A, len(night_shift_candidates))
            assigned_count = 0
            temp_available_after_night = available_staff.copy()

            # Trộn danh sách nhân viên có thể làm đêm
            random.shuffle(night_shift_candidates)

            # Lấy nhân viên đầu tiên có thể làm đêm
            for staff_idx in night_shift_candidates:
                if assigned_count < num_to_assign_night:
                    if staff_idx in temp_available_after_night:
                        individual[staff_idx][d] = 4
                        assigned_count += 1
                        temp_available_after_night.remove(staff_idx)
                else:
                    break
            
            # Lấy danh sách nhân viên có thể làm ngày
            available_staff = temp_available_after_night
            day_shifts = [1, 2, 3]
            random.shuffle(day_shifts)
            for shift_type in day_shifts:
                num_to_assign_day_shift = min(self.A, len(available_staff))
                assigned_count = 0
                temp_available_after_shift = available_staff.copy()

                random.shuffle(available_staff)
                for staff_idx in available_staff:
                    if assigned_count < num_to_assign_day_shift:
                        if individual[staff_idx][d] == -1: # Chưa gán
                            individual[staff_idx][d] = shift_type
                            assigned_count += 1
                            temp_available_after_shift.remove(staff_idx)
                    else:
                        break
                available_staff = temp_available_after_shift

            # Gán những người còn lại nghỉ
            for staff_idx in range(self.N):
                if individual[staff_idx][d] == -1:
                    individual[staff_idx][d] = 0

        # Check lại: cần được gán cho một giá trị
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] == -1:
                    print(f"LỖI NGHIÊM TRỌNG trong _create_initial_solution: Ô chưa được gán [{n}][{d}]! Đặt mặc định là 0.")
                    individual[n][d] = 0

        return individual
    
    def _calculate_fitness(self, individual):
        
        # Ràng buộc ngày nghỉ
        violation_fixed_off = 0
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] > 0 and self.IS_PRE_ASSIGNED_OFF[n][d]:
                    violation_fixed_off += 1

        # Ràng buộc A <= số người trong 1 ca <= B
        violation_coverage = 0
        # Đếm số người làm việc trong từng ca
        shifts_coverage = [[0 for _ in range(5)] for _ in range(self.D)]
        for d in range(self.D):
            for n in range(self.N):
                shift_val = individual[n][d]
                if 0 <= shift_val <= 4:
                    shifts_coverage[d][shift_val] += 1

        # Tính toán số lượng người thiếu và thừa trong từng ca
        min_shortage_total = 0
        max_excess_total = 0
        for d in range(self.D):
            for shift_type in range(1, 5):
                actual_coverage = shifts_coverage[d][shift_type]
                min_shortage_total += max(0, self.A - actual_coverage)
                max_excess_total += max(0, actual_coverage - self.B)
        violation_coverage = min_shortage_total + max_excess_total

        # Ràng buộc ngày đêm
        violation_night_then_off = 0
        if self.D > 1:
            for n in range(self.N):
                for d in range(self.D - 1):
                    if individual[n][d] == 4 and individual[n][d+1] != 0:
                        violation_night_then_off += 1

        # Ràng buộc số lượng đêm làm nhiều nhất
        night_shifts_per_staff = [0] * self.N
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] == 4:
                    night_shifts_per_staff[n] += 1

        max_night_shifts_objective = 0.0
        if self.N > 0 and night_shifts_per_staff:
            max_night_shifts_objective = float(max(night_shifts_per_staff))

        # Tính toán fitness
        w_fixed_off = 1000.0
        w_coverage = 1000.0
        w_night_off = 1000.0
        w_max_nights = 1.0

        total_violation_penalty = (
            w_fixed_off * violation_fixed_off +
            w_coverage * violation_coverage +
            w_night_off * violation_night_then_off
        )
        
        fitness_score = -(total_violation_penalty + w_max_nights * max_night_shifts_objective)
        return fitness_score    
    
    def _swap_neighbor(self, current_solution):
        neighbor = copy.deepcopy(current_solution)
        if self.N < 2 or self.D == 0:
            return neighbor
        
        # Thử tối đa 20 lần
        max_attempts = 20
        for _ in range(max_attempts):
            day_idx = random.randint(0, self.D - 1)
            
            # Chọn 1 nhân viên bất kỳ làm việc ca 4
            available_staff_indices_for_night = [n for n in range(self.N) if neighbor[n][day_idx] == 4]
            if len(available_staff_indices_for_night) == 0:
                continue
            n1_idx = random.choice(available_staff_indices_for_night)

            # Chọn 1 nhân viên bất kỳ làm việc ca 1, 2, 3
            available_staff_indices_for_day = [n for n in range(self.N) if neighbor[n][day_idx] == -1]
            if len(available_staff_indices_for_day) == 0:
                continue

            # Hoán đổi ca làm việc của 2 nhân viên
            n2_idx = random.choice(available_staff_indices_for_day)
            neighbor[n1_idx][day_idx] = neighbor[n2_idx][day_idx]
            neighbor[n2_idx][day_idx] = 4
            return neighbor
        return copy.deepcopy(current_solution)
    
    def _fix_night_shift_violation_neighbor(self, current_solution):
        neighbor = copy.deepcopy(current_solution)
        if self.D <= 1:
            return neighbor
        
        # Tìm các nhân viên làm đêm nhiều nhất
        violation_positions = []
        for n in range(self.N):
            for d in range(self.D-1):
                if neighbor[n][d] == 4 and neighbor[n][d+1] != 0:
                    violation_positions.append((n, d))
        if not violation_positions:
            return neighbor
        
        # Chọn 1 nhân viên bất kỳ làm việc ca 1, 2, 3
        n1_idx, d1_idx = random.choice(violation_positions)
        next_day_idx = d1_idx + 1
        
        # Chọn một nhân viên nghỉ vào ngày tiếp theo
        available_staff_indices_next_day = [n for n in range(self.N) if neighbor[n][next_day_idx] == 0 and not self.IS_PRE_ASSIGNED_OFF[n][d1_idx] and neighbor[n][d1_idx] == 0]
        if not available_staff_indices_next_day:
            return neighbor
        
        n2_idx = random.choice(available_staff_indices_next_day)
        neighbor[n1_idx][d1_idx] = 0
        neighbor[n2_idx][next_day_idx] = 4
        return neighbor
    
    def _generate_neighbor(self, current_solution):
        operators = [
            (self._swap_neighbor, 0.70),       
            (self._fix_night_shift_violation_neighbor, 0.30),      
        ]
        rand_val = random.random()
        cumulative_prob = 0
        for operator_func, prob in operators:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return operator_func(current_solution)
        return self._swap_neighbor(current_solution)
    
    def _hill_climbing_or_sa(self,
                             initial_solution_func, # Sẽ là self._create_initial_solution
                             fitness_func,          # Sẽ là self._calculate_fitness
                             neighbor_func,         # Sẽ là self._generate_neighbor
                             max_iterations, 
                             max_iterations_no_improvement, 
                             num_neighbors_to_explore, 
                             time_limit_for_run_secs=None):
        run_start_time = time.time()
        current_solution = initial_solution_func()
        current_fitness = fitness_func(current_solution)
        
        best_solution_in_run = copy.deepcopy(current_solution) 
        best_fitness_in_run = current_fitness
        temperature = self.initial_temperature

        iterations_without_improvement_in_best = 0

        for iteration in range(max_iterations):
            if time_limit_for_run_secs and (time.time() - run_start_time) > time_limit_for_run_secs:
                break

            candidate_neighbor = None
            candidate_neighbor_fitness = -float('inf')

            if num_neighbors_to_explore == 1 and self.use_simulated_annealing:
                candidate_neighbor = neighbor_func(current_solution)
                candidate_neighbor_fitness = fitness_func(candidate_neighbor)
            else:
                for _ in range(num_neighbors_to_explore):
                    neighbor = neighbor_func(current_solution) 
                    neighbor_fitness = fitness_func(neighbor)
                    if neighbor_fitness > candidate_neighbor_fitness:
                        candidate_neighbor = copy.deepcopy(neighbor) 
                        candidate_neighbor_fitness = neighbor_fitness
            
            accepted_move = False
            if candidate_neighbor is not None:
                if candidate_neighbor_fitness > current_fitness:
                    current_solution = copy.deepcopy(candidate_neighbor)
                    current_fitness = candidate_neighbor_fitness
                    accepted_move = True
                elif self.use_simulated_annealing and temperature > self.min_temperature:
                    delta_fitness = candidate_neighbor_fitness - current_fitness
                    if temperature > 1e-6 : 
                        acceptance_probability = math.exp(delta_fitness / temperature)
                        if random.random() < acceptance_probability:
                            current_solution = copy.deepcopy(candidate_neighbor)
                            current_fitness = candidate_neighbor_fitness
                            accepted_move = True
                
            if accepted_move:
                if current_fitness > best_fitness_in_run:
                     best_fitness_in_run = current_fitness
                     best_solution_in_run = copy.deepcopy(current_solution)
                     iterations_without_improvement_in_best = 0
                else:
                    iterations_without_improvement_in_best += 1
            else:
                iterations_without_improvement_in_best += 1
                
            if self.use_simulated_annealing:
                temperature = max(self.min_temperature, temperature * self.cooling_rate)

            if iterations_without_improvement_in_best >= max_iterations_no_improvement:
                break
            
        run_time = time.time() - run_start_time

        print(f"HC/SA: Fitness tốt nhất trong lần chạy = {best_fitness_in_run:.4f}. Số vòng lặp = {iteration+1}. Thời gian = {run_time:.2f}s")
        
        return best_solution_in_run, best_fitness_in_run, run_time
    
    def _run_multi_start_local_search(self, overall_time_limit_secs_for_this_solve=None):
        overall_start_time = time.time()
        best_overall_solution = None
        best_overall_fitness = -float('inf') 

        for restart_num in range(self.num_restarts):
            if overall_time_limit_secs_for_this_solve and \
               (time.time() - overall_start_time) > overall_time_limit_secs_for_this_solve:
                break
            
            time_left_for_hc_run = None
            if overall_time_limit_secs_for_this_solve:
                elapsed_time = time.time() - overall_start_time
                time_left_for_hc_run = overall_time_limit_secs_for_this_solve - elapsed_time
                if time_left_for_hc_run <= 0:
                    break
            
            solution_from_hc, fitness_from_hc, run_time = self._hill_climbing_or_sa(
                initial_solution_func=self._create_initial_solution,
                fitness_func=self._calculate_fitness,
                neighbor_func=self._generate_neighbor,
                max_iterations=self.max_iterations_per_restart,
                max_iterations_no_improvement=self.max_no_improvement_per_restart,
                num_neighbors_to_explore=self.num_neighbors_to_explore_per_step,
                time_limit_for_run_secs=time_left_for_hc_run
            )

            if fitness_from_hc > best_overall_fitness:
                best_overall_fitness = fitness_from_hc
                best_overall_solution = copy.deepcopy(solution_from_hc) 

            print(f"Khởi động lại {restart_num + 1}: Fitness TỔNG THỂ tốt nhất MỚI = {best_overall_fitness:.4f}")

        total_time_taken = time.time() - overall_start_time
            
        return best_overall_solution, best_overall_fitness, total_time_taken
    
    def _handle_output(self, solution, fitness, run_time, testcase_filename_original, output_dir, is_empty_case=False):
        if solution is not None and not is_empty_case:
                
            if self.N > 0 and self.D > 0 :
                    fixed_off_violations = 0
                    for n in range(self.N):
                        for d in range(self.D):
                            if solution[n][d] > 0 and self.IS_PRE_ASSIGNED_OFF[n][d]:
                                fixed_off_violations +=1
                    print(f"Số vi phạm ngày nghỉ cố định: {fixed_off_violations}")
                    
                    night_shifts_per_employee_list = [0] * self.N
                    for n_idx in range(self.N):
                        for d_idx in range(self.D):
                            if solution[n_idx][d_idx] == 4:
                                night_shifts_per_employee_list[n_idx] +=1
                    print(f"Số ca đêm mỗi nhân viên: {night_shifts_per_employee_list}")
                    max_nights_val = 0
                    if night_shifts_per_employee_list: 
                        max_nights_val = max(night_shifts_per_employee_list)
                    print(f"Số ca đêm nhiều nhất cho một nhân viên: {max_nights_val}")
                    
                    night_off_violations_count = 0
                    if self.D > 1:
                        for n in range(self.N):
                            for d in range(self.D - 1):
                                if solution[n][d] == 4 and solution[n][d+1] != 0:
                                    night_off_violations_count += 1
                    print(f"Số vi phạm ca đêm theo sau bởi ca không nghỉ: {night_off_violations_count}")

                    coverage_violations_final_count = 0
                    temp_shifts_coverage = [[0 for _ in range(5)] for _ in range(self.D)] 
                    for d_val in range(self.D):
                        for n_val in range(self.N):
                            shift_value = solution[n_val][d_val]
                            if 0 <= shift_value <= 4:
                                temp_shifts_coverage[d_val][shift_value] +=1
                            else:
                                print(f"Cảnh báo: Giá trị ca không hợp lệ {shift_value} cho NV {n_val} ngày {d_val} trong giải pháp cuối cùng.")
                    for d_val in range(self.D):
                        for shift_type in range(1,5): 
                            actual_cov = temp_shifts_coverage[d_val][shift_type]
                            coverage_violations_final_count += max(0, self.A - actual_cov) 
                            coverage_violations_final_count += max(0, actual_cov - self.B) 
                    print(f"Số vi phạm về độ bao phủ (thiếu/thừa): {coverage_violations_final_count}")
            elif is_empty_case:
                 print(f"Giải pháp rỗng được tạo cho N=0 hoặc D=0.")
            else:
                print(f"\n--- Không tìm thấy giải pháp nào bởi Solver ---")

        output_filename = testcase_filename_original.replace(".txt", ".out")
        output_filepath = os.path.join(output_dir, output_filename)
        with open(output_filepath, "w") as f:
            if self.N > 0:
                if solution is not None : # Handles N > 0
                    for n_idx in range(self.N):
                        # If D_DAYS is 0, solution[n_idx] is [], join results in ""
                        shifts_for_staff_n = [str(solution[n_idx][d_idx]) for d_idx in range(self.D)]
                        line = " ".join(shifts_for_staff_n)
                        f.write(line + "\n")
                else: # solution is None, but N_STAFF > 0 (e.g. algo failed to produce one)
                    for _ in range(self.N):
                        f.write("\n") # N empty lines
                
                # write runtime
                f.write(f"{run_time:.2f}\n")
            # If N == 0, file remains empty.
        print(f"Giải pháp đã được lưu vào {output_filepath}")

    def solve(self, N, D, A, B, pre_assigned_off_list_of_lists, testcase_filename_original, output_dir="Testcase", time_limit_for_solve_secs=None):
        # 1. Thiết lập tham số bài toán cho instance này
        self.N = N
        self.D = D
        self.A = A
        self.B = B
        
        self.IS_PRE_ASSIGNED_OFF = [[False for _ in range(self.D)] for _ in range(self.N)]
        for staff_idx, days_off_for_staff in enumerate(pre_assigned_off_list_of_lists):
            for day_input_val in days_off_for_staff:
                if day_input_val == -1:
                    break
                if 1 <= day_input_val <= self.D:
                    day_idx = day_input_val - 1
                    if 0 <= day_idx < self.D and 0 <= staff_idx < self.N: # Kiểm tra biên
                        self.IS_PRE_ASSIGNED_OFF[staff_idx][day_idx] = True
        
        print("\n--- Input cho Solver Instance ---")
        print(f"N={self.N}, D={self.D}, A={self.A}, B={self.B}")
        print("-" * 20)

        # 2. Chạy thuật toán
        best_solution, best_fitness, run_time = self._run_multi_start_local_search(
            overall_time_limit_secs_for_this_solve = time_limit_for_solve_secs
        )

        # 3. Định dạng và xuất giải pháp
        self._handle_output(best_solution, best_fitness, run_time, testcase_filename_original, output_dir)
        
        return best_solution, best_fitness, run_time

if __name__ == "__main__":
    testcase_dir = "Testcase" 
    output_dir = "Output_ngocanhh"
    if not os.path.isdir(testcase_dir):
        os.makedirs(testcase_dir, exist_ok=True)
        print(f"Thư mục testcase đã được tạo: {testcase_dir}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Thư mục output đã được tạo: {output_dir}")
    
    testcases_files = [f for f in os.listdir(testcase_dir) if f.endswith(".txt")]
    testcases_files.sort() 

    solver_config = {
        "num_restarts": 10,  
        "max_iterations_per_restart": 500, 
        "max_no_improvement_per_restart": 50, 
        "num_neighbors_to_explore_per_step": 30,
        "use_simulated_annealing": True,
        "initial_temperature": 100.0,
        "cooling_rate": 0.99,
        "min_temperature": 0.1,
    }
    TIME_LIMIT_PER_TESTCASE_SECS = 60

    solver = StaffRosteringSolver(**solver_config)

    for testcase_filename in testcases_files:
        print(f"\n--- Đang xử lý testcase: {testcase_filename} ---")
        current_testcase_path = os.path.join(testcase_dir, testcase_filename)

        # Đọc input cho testcase hiện tại
        pre_assigned_off_data_local = []
        with open(current_testcase_path, 'r') as f:
            N_local, D_local, A_local, B_local = map(int, f.readline().split())
            for _ in range(N_local):
                line = f.readline()
                if line.strip():
                    days_off_input = list(map(int, line.split()))
                    pre_assigned_off_data_local.append(days_off_input)
                else:
                    pre_assigned_off_data_local.append([-1])
        
        # print(f'N: {N_local}, D: {D_local}, A: {A_local}, B: {B_local}')
        # print(f'Pre-assigned off: {pre_assigned_off_data_local}')

        solver.solve(
            N=N_local, 
            D=D_local, 
            A=A_local, 
            B=B_local,
            pre_assigned_off_list_of_lists=pre_assigned_off_data_local,
            testcase_filename_original=testcase_filename,
            output_dir=output_dir,
            time_limit_for_solve_secs=TIME_LIMIT_PER_TESTCASE_SECS)