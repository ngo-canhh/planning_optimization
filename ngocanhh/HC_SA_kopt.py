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
                min_temperature=0.1,
                # New parameters for _swap_neighbor modification
                initial_swap_percentage_k=10.0, # k% of N for initial number of swaps
                max_swaps_per_op_cap=5 # Absolute cap on number of swaps in one _swap_neighbor call
                ):
        
        # Tham số thuật toán
        self.num_restarts = num_restarts
        self.max_iterations_per_restart = max_iterations_per_restart
        self.max_no_improvement_per_restart = max_no_improvement_per_restart
        self.num_neighbors_to_explore_per_step = num_neighbors_to_explore_per_step
        
        self.use_simulated_annealing = use_simulated_annealing
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

        # New parameters
        self.initial_swap_percentage_k = initial_swap_percentage_k
        self.max_swaps_per_op_cap = max_swaps_per_op_cap
        
        # Tham số bài toán (sẽ được đặt trong solve())
        self.N = None
        self.D = None
        self.A = None
        self.B = None
        self.IS_PRE_ASSIGNED_OFF = None # (N, D) - boolean
        self.max_swaps_at_start_calculated = 1 # Giá trị mặc định, sẽ tính lại trong solve()


    def _create_initial_solution(self):
        # individual[staff_idx][day_idx]
        individual = [[-1 for _ in range(self.D)] for _ in range(self.N)]

        # Gán ngày nghỉ cố định trước
        for d in range(self.D):
            for i in range(self.N):
                if self.IS_PRE_ASSIGNED_OFF[i][d]:
                    individual[i][d] = 0

        # Xếp lịch cho từng ngày
        for d in range(self.D):
            # Nếu ngày hôm qua làm đêm -> hôm nay nghỉ (nếu chưa được gán gì khác)
            if d > 0:
                for i in range(self.N):
                    if individual[i][d - 1] == 4: # Làm đêm hôm qua
                        if individual[i][d] == -1: # Chưa được gán (nghỉ cố định đã được gán ở trên)
                            individual[i][d] = 0
        
            # Lấy danh sách nhân viên CHƯA được gán lịch cho ngày d
            available_staff_for_assignment = [i for i in range(self.N) if individual[i][d] == -1]
            random.shuffle(available_staff_for_assignment)

            # Gán ca Đêm (4)
            # Ưu tiên những người không bị IS_PRE_ASSIGNED_OFF[i][d] (đã được xử lý)
            # và không phải nghỉ sau ca đêm (đã được xử lý)
            night_shift_candidates = available_staff_for_assignment.copy() # Chỉ những người còn -1
            num_to_assign_night = min(self.A, len(night_shift_candidates)) # Cố gắng đạt ít nhất A
            assigned_night_count = 0
            
            temp_available_after_night = night_shift_candidates.copy()
            random.shuffle(night_shift_candidates)

            for staff_idx in night_shift_candidates:
                if assigned_night_count < num_to_assign_night:
                    individual[staff_idx][d] = 4
                    assigned_night_count += 1
                    temp_available_after_night.remove(staff_idx)
                else:
                    break
            
            available_staff_for_day_shifts = temp_available_after_night

            # Gán ca Ngày (1, 2, 3)
            day_shifts_types = [1, 2, 3]
            random.shuffle(day_shifts_types)

            for shift_type in day_shifts_types:
                num_to_assign_day_shift = min(self.A, len(available_staff_for_day_shifts))
                assigned_day_shift_count = 0
                temp_available_after_this_shift = available_staff_for_day_shifts.copy()
                
                random.shuffle(available_staff_for_day_shifts)

                for staff_idx in available_staff_for_day_shifts:
                    if assigned_day_shift_count < num_to_assign_day_shift:
                        individual[staff_idx][d] = shift_type
                        assigned_day_shift_count += 1
                        temp_available_after_this_shift.remove(staff_idx)
                    else:
                        break
                available_staff_for_day_shifts = temp_available_after_this_shift
            
            # Gán những người còn lại (-1) nghỉ (0)
            for staff_idx in range(self.N):
                if individual[staff_idx][d] == -1:
                    individual[staff_idx][d] = 0
        
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] == -1:
                    print(f"LỖI NGHIÊM TRỌNG trong _create_initial_solution: Ô chưa được gán [{n}][{d}]! Đặt mặc định là 0.")
                    individual[n][d] = 0
        return individual
    
    def _calculate_fitness(self, individual):
        
        violation_fixed_off = 0
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] > 0 and self.IS_PRE_ASSIGNED_OFF[n][d]:
                    violation_fixed_off += 1

        violation_coverage = 0
        shifts_coverage = [[0 for _ in range(5)] for _ in range(self.D)]
        for d in range(self.D):
            for n in range(self.N):
                shift_val = individual[n][d]
                if 0 <= shift_val <= 4:
                    shifts_coverage[d][shift_val] += 1

        min_shortage_total = 0
        max_excess_total = 0
        for d in range(self.D):
            for shift_type in range(1, 5): 
                actual_coverage = shifts_coverage[d][shift_type]
                min_shortage_total += max(0, self.A - actual_coverage)
                max_excess_total += max(0, actual_coverage - self.B)
        violation_coverage = min_shortage_total + max_excess_total

        violation_night_then_off = 0
        if self.D > 1:
            for n in range(self.N):
                for d in range(self.D - 1):
                    if individual[n][d] == 4 and individual[n][d+1] != 0:
                        violation_night_then_off += 1

        night_shifts_per_staff = [0] * self.N
        for n in range(self.N):
            for d_val in range(self.D): # Changed d to d_val to avoid conflict with outer scope if any
                if individual[n][d_val] == 4:
                    night_shifts_per_staff[n] += 1
        
        max_night_shifts_objective = 0.0
        if self.N > 0 and night_shifts_per_staff: 
            max_night_shifts_objective = float(max(night_shifts_per_staff)) if night_shifts_per_staff else 0.0

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
    
    def _swap_night_with_off_neighbor(self, current_solution, num_swaps_to_perform=1):
        neighbor = copy.deepcopy(current_solution)
        if self.N < 1 or self.D == 0:
            return neighbor

        swaps_actually_done = 0
        for _ in range(num_swaps_to_perform):
            max_attempts_one_swap = 10 
            found_one_swap = False
            for attempt in range(max_attempts_one_swap):
                if self.N < 2 and num_swaps_to_perform > 0 : # Cần ít nhất 2 nhân viên để swap
                    return neighbor # Không thể thực hiện swap này

                day_idx = random.randint(0, self.D - 1)

                staff_on_night_shift_today = [
                    n for n in range(self.N) if neighbor[n][day_idx] == 4
                ]
                staff_on_off_duty_today_can_work_night = [
                    n for n in range(self.N) 
                    if neighbor[n][day_idx] == 0 and not self.IS_PRE_ASSIGNED_OFF[n][day_idx]
                ]

                if not staff_on_night_shift_today or not staff_on_off_duty_today_can_work_night:
                    continue 

                n1_idx = random.choice(staff_on_night_shift_today) 
                
                # Tìm nhân viên khác n1 để swap
                possible_n2_candidates = [n2 for n2 in staff_on_off_duty_today_can_work_night if n2 != n1_idx]
                if not possible_n2_candidates:
                    continue # Không tìm thấy n2 khác n1
                
                n2_idx = random.choice(possible_n2_candidates)
                
                neighbor[n1_idx][day_idx] = 0 
                neighbor[n2_idx][day_idx] = 4 
                swaps_actually_done += 1
                found_one_swap = True
                break 
        
        # if num_swaps_to_perform > 0 and swaps_actually_done > 0:
        # print(f"DEBUG: Requested {num_swaps_to_perform} swaps, did {swaps_actually_done}.")
        return neighbor
    
    def _fix_night_shift_violation_neighbor(self, current_solution):
        neighbor = copy.deepcopy(current_solution)
        if self.D <= 1:
            return neighbor
        
        violation_positions = []
        for n in range(self.N):
            for d_loop in range(self.D - 1):
                if neighbor[n][d_loop] == 4 and neighbor[n][d_loop+1] != 0:
                    violation_positions.append((n, d_loop))
        
        if not violation_positions:
            return neighbor
        
        n_violating, d_violating_night = random.choice(violation_positions)
        d_violating_work = d_violating_night + 1
        
        original_shift_on_d_violating_work = neighbor[n_violating][d_violating_work]
        neighbor[n_violating][d_violating_work] = 0 # Buộc nghỉ
        return neighbor
    
    def _generate_neighbor(self, current_solution, num_swaps_for_swap_op=1):
        operators = [
            (self._swap_night_with_off_neighbor, 0.70), # Pass num_swaps_for_swap_op here     
            (self._fix_night_shift_violation_neighbor, 0.30),      
        ]
        rand_val = random.random()
        cumulative_prob = 0
        for operator_func, prob in operators:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                if operator_func == self._swap_night_with_off_neighbor:
                    return operator_func(current_solution, num_swaps_for_swap_op)
                else:
                    return operator_func(current_solution)
        
        return self._swap_night_with_off_neighbor(current_solution, num_swaps_for_swap_op)
    
    def _hill_climbing_or_sa(self,
                             initial_solution_func, 
                             fitness_func,          
                             neighbor_func,        
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
            
            num_swaps_for_op = 1 
            if self.use_simulated_annealing and self.initial_swap_percentage_k > 0 and self.N > 0:
                if self.initial_temperature > self.min_temperature: 
                    temp_ratio = (temperature - self.min_temperature) / (self.initial_temperature - self.min_temperature)
                    temp_ratio = max(0, min(1, temp_ratio)) 
                else: 
                    temp_ratio = 1.0 if temperature >= self.initial_temperature else 0.0
                
                num_swaps_for_op = max(1, int(round(self.max_swaps_at_start_calculated * temp_ratio)))
            elif not self.use_simulated_annealing:
                 num_swaps_for_op = 1 

            candidate_neighbor = None
            candidate_neighbor_fitness = -float('inf')

            if num_neighbors_to_explore == 1 and self.use_simulated_annealing:
                candidate_neighbor = neighbor_func(current_solution, num_swaps_for_op)
                candidate_neighbor_fitness = fitness_func(candidate_neighbor)
            else:
                for _ in range(num_neighbors_to_explore):
                    neighbor = neighbor_func(current_solution, num_swaps_for_op) 
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
                    if temperature > 1e-9 : 
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
        # print(f"HC/SA: Fitness tốt nhất trong lần chạy = {best_fitness_in_run:.4f}. Số vòng lặp = {iteration+1}. Thời gian = {run_time:.2f}s. Cuối Temp={temperature:.2f if self.use_simulated_annealing else 'N/A'}")
        return best_solution_in_run, best_fitness_in_run, run_time
    
    def _run_multi_start_local_search(self, overall_time_limit_secs_for_this_solve=None):
        overall_start_time = time.time()
        best_overall_solution = None
        best_overall_fitness = -float('inf') 

        for restart_num in range(self.num_restarts):
            # print(f"--- Bắt đầu Khởi động lại {restart_num + 1}/{self.num_restarts} ---") # Verbose
            if overall_time_limit_secs_for_this_solve and \
               (time.time() - overall_start_time) > overall_time_limit_secs_for_this_solve:
                # print("Hết thời gian tổng thể cho phép trước khi bắt đầu khởi động lại này.") # Verbose
                break
            
            time_left_for_hc_run = None
            if overall_time_limit_secs_for_this_solve:
                elapsed_time = time.time() - overall_start_time
                time_left_for_hc_run = overall_time_limit_secs_for_this_solve - elapsed_time
                if time_left_for_hc_run <= 0:
                    # print("Hết thời gian còn lại cho lần chạy HC/SA này.") # Verbose
                    break
            
            solution_from_hc, fitness_from_hc, run_time_one_restart = self._hill_climbing_or_sa(
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
                print(f"Khởi động lại {restart_num + 1}: Fitness TỔNG THỂ tốt nhất MỚI = {best_overall_fitness:.4f} (thời gian lần chạy này: {run_time_one_restart:.2f}s)") # Verbose
            else: # Verbose
                print(f"Khởi động lại {restart_num + 1}: Không cải thiện (fitness {fitness_from_hc:.4f}). Fitness tổng thể tốt nhất vẫn là {best_overall_fitness:.4f} (thời gian lần chạy này: {run_time_one_restart:.2f}s)")


        total_time_taken = time.time() - overall_start_time
        # print(f"--- Hoàn thành tất cả các lần khởi động lại. Tổng thời gian: {total_time_taken:.2f}s ---") # Verbose
            
        return best_overall_solution, best_overall_fitness, total_time_taken
    
    def _handle_output(self, solution, fitness, run_time, testcase_filename_original, output_dir, is_empty_case=False):

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            # print(f"Thư mục output đã được tạo: {output_dir}") # Verbose

        output_filename = testcase_filename_original.replace(".txt", ".out")
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, "w") as f:
            if self.N == 0 : # Trường hợp N=0, file output rỗng (không có cả runtime)
                pass 
            else: # N > 0
                if solution is not None: # N > 0, có giải pháp (có thể là D=0 hoặc D>0)
                    for n_idx in range(self.N):
                        # Nếu D=0, solution[n_idx] là [], join results in "" (dòng trống)
                        shifts_for_staff_n = [str(solution[n_idx][d_idx]) for d_idx in range(self.D)]
                        line = " ".join(shifts_for_staff_n)
                        f.write(line + "\n")
                else: # N > 0, nhưng solution là None (ví dụ, thuật toán thất bại)
                      # Ghi N dòng trống
                    for _ in range(self.N):
                        f.write("\n")
                f.write(f"{run_time:.2f}\n") # Ghi runtime cho mọi trường hợp N > 0
        
        # print(f"Giải pháp đã được lưu vào {output_filepath}") # Verbose

    def solve(self, N, D, A, B, pre_assigned_off_list_of_lists, testcase_filename_original, output_dir="Testcase", time_limit_for_solve_secs=None):
        self.N = N
        self.D = D
        self.A = A
        self.B = B
        
        if self.N > 0 :
             self.max_swaps_at_start_calculated = max(1, int(self.N * self.initial_swap_percentage_k / 100.0))
             self.max_swaps_at_start_calculated = min(self.max_swaps_at_start_calculated, self.max_swaps_per_op_cap)
        else:
            self.max_swaps_at_start_calculated = 1

        self.IS_PRE_ASSIGNED_OFF = [[False for _ in range(self.D)] for _ in range(self.N)]
        if self.N > 0 and self.D > 0: 
            for staff_idx, days_off_for_staff in enumerate(pre_assigned_off_list_of_lists):
                if staff_idx >= self.N: continue 
                for day_input_val in days_off_for_staff:
                    if day_input_val == -1: 
                        break
                    if 1 <= day_input_val <= self.D: 
                        day_idx = day_input_val - 1 
                        self.IS_PRE_ASSIGNED_OFF[staff_idx][day_idx] = True
        
        # print("\n--- Input cho Solver Instance ---") # Verbose
        # print(f"N={self.N}, D={self.D}, A={self.A}, B={self.B}") # Verbose
        # print(f"Cấu hình SwapOp: k={self.initial_swap_percentage_k}%, cap={self.max_swaps_per_op_cap}, max_swaps_start={self.max_swaps_at_start_calculated}") # Verbose
        # print("-" * 20) # Verbose

        if self.N == 0: # Xử lý trường hợp N=0 (output file rỗng, không runtime)
            self._handle_output(None, 0, 0.00, testcase_filename_original, output_dir, is_empty_case=True)
            return [], 0, 0.00 # Trả về giải pháp rỗng
        
        if self.D == 0: # Xử lý trường hợp D=0 (N dòng trống, có runtime)
            # Tạo giải pháp N dòng trống
            empty_solution_d0 = [[] for _ in range(self.N)]
            actual_runtime = 0.00 
            self._handle_output(empty_solution_d0, 0, actual_runtime, testcase_filename_original, output_dir, is_empty_case=False) # Not truly "empty case" if N>0
            return empty_solution_d0, 0, actual_runtime


        best_solution, best_fitness, run_time = self._run_multi_start_local_search(
            overall_time_limit_secs_for_this_solve = time_limit_for_solve_secs
        )

        if best_solution is None: 
            # print("Không tìm thấy giải pháp nào. Sẽ tạo đầu ra với lịch trống.") # Verbose
            if self.N > 0 and self.D > 0: # Chỉ khi N và D > 0 mà không có giải pháp
                 best_solution = [[0 for _ in range(self.D)] for _ in range(self.N)]
            # best_fitness vẫn là -inf, run_time là tổng thời gian đã chạy

        self._handle_output(best_solution, best_fitness, run_time, testcase_filename_original, output_dir, is_empty_case=(self.N==0 or self.D==0))
        
        return best_solution, best_fitness, run_time

if __name__ == "__main__":
    testcase_dir = "Testcase" 
    # Đặt tên thư mục output theo cấu hình hoặc ngày giờ để dễ quản lý
    output_dir_name = f"Output_ngocanhh"
    output_dir = output_dir_name

    if not os.path.isdir(testcase_dir):
        os.makedirs(testcase_dir, exist_ok=True)
        print(f"Thư mục testcase đã được tạo: {testcase_dir}")
        # Tạo file test mẫu nếu chưa có
        sample_test_file = os.path.join(testcase_dir, "sample_01.txt")
        if not os.path.exists(sample_test_file):
            with open(sample_test_file, "w") as f:
                f.write("5 7 1 2\n") # N D A B
                f.write("2 3\n")    # Staff 0 off days 2, 3 (1-indexed)
                f.write("-1\n")     # Staff 1 no pre-assigned off days
                f.write("7\n")      # Staff 2 off day 7
                f.write("1 5\n")    # Staff 3 off days 1, 5
                f.write("4\n")      # Staff 4 off day 4
            print(f"Tạo file test mẫu {sample_test_file}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Thư mục output đã được tạo: {output_dir}")
    
    testcases_files = [f for f in os.listdir(testcase_dir) if f.endswith(".txt")]
    if not testcases_files:
        print(f"KHÔNG tìm thấy file .txt nào trong {testcase_dir}. Vui lòng kiểm tra lại.")
    testcases_files.sort() 

    solver_config = {
        "num_restarts": 10,  
        "max_iterations_per_restart": 500, 
        "max_no_improvement_per_restart": 75, 
        "num_neighbors_to_explore_per_step": 25, 
        "use_simulated_annealing": True,
        "initial_temperature": 120.0,
        "cooling_rate": 0.98, 
        "min_temperature": 0.05,
        "initial_swap_percentage_k": 15.0, 
        "max_swaps_per_op_cap": 10        
    }
    TIME_LIMIT_PER_TESTCASE_SECS = 58 

    # Khởi tạo solver một lần với cấu hình
    solver = StaffRosteringSolver(**solver_config)

    total_execution_start_time = time.time()
    for testcase_filename in testcases_files:
        print(f"--- Xử lý: {testcase_filename} ---")
        current_testcase_path = os.path.join(testcase_dir, testcase_filename)

        pre_assigned_off_data_local = []
        try:
            with open(current_testcase_path, 'r') as f:
                N_local, D_local, A_local, B_local = map(int, f.readline().split())
                if N_local > 0 : 
                    for _ in range(N_local):
                        line = f.readline()
                        if line.strip(): 
                            days_off_input = list(map(int, line.split()))
                            pre_assigned_off_data_local.append(days_off_input)
                        else: 
                            pre_assigned_off_data_local.append([-1]) 
        except Exception as e:
            print(f"LỖI khi đọc file {testcase_filename}: {e}")
            continue 
        
        # Gọi solver.solve() cho từng testcase
        # solver đã được khởi tạo ở trên, các tham số N, D,... sẽ được cập nhật bên trong solve()
        solver.solve(
            N=N_local, 
            D=D_local, 
            A=A_local, 
            B=B_local,
            pre_assigned_off_list_of_lists=pre_assigned_off_data_local,
            testcase_filename_original=testcase_filename,
            output_dir=output_dir,
            time_limit_for_solve_secs=TIME_LIMIT_PER_TESTCASE_SECS
        )
    total_execution_end_time = time.time()
    print(f"\n--- HOÀN THÀNH TẤT CẢ TESTCASES ---")
    print(f"Tổng thời gian thực thi toàn bộ: {total_execution_end_time - total_execution_start_time:.2f} giây.")
    print(f"Kết quả được lưu trong thư mục: {output_dir}")