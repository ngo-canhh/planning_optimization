import random
import time
import math
import copy
import os
from typing import List, Tuple, Optional, Any, Callable

class LocalSearchSolver:
    def __init__(self,
                num_restarts=5,
                max_iterations_per_restart=200,
                max_no_improvement_per_restart=50,
                num_neighbors_to_explore_per_step=10,  # Changed default to 10 for hybrid approach
                initial_temperature=100.0,
                cooling_rate=0.99,
                min_temperature=0.1,
                initial_swap_percentage_k=10.0,  # k% of N for initial number of swaps
                max_swaps_per_op_cap=5,  # Absolute cap on number of swaps in one _swap_neighbor call
                verbose=True  # Control detailed console output
                ):
        
        # Algorithm parameters
        self.num_restarts = num_restarts
        self.max_iterations_per_restart = max_iterations_per_restart
        self.max_no_improvement_per_restart = max_no_improvement_per_restart
        self.num_neighbors_to_explore_per_step = num_neighbors_to_explore_per_step
        
        # SA parameters
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

        # Parameters for k-opt neighbor generation
        self.initial_swap_percentage_k = initial_swap_percentage_k
        self.max_swaps_per_op_cap = max_swaps_per_op_cap
        
        # Output control
        self.verbose = verbose
        
        # Problem parameters (will be set in solve())
        self.N = None  # Number of staff
        self.D = None  # Number of days
        self.A = None  # Minimum staff per shift
        self.B = None  # Maximum staff per shift
        self.IS_PRE_ASSIGNED_OFF = None  # (N, D) - boolean
        self.max_swaps_at_start_calculated = 1  # Default value, will be recalculated in solve()

    def _create_initial_solution(self):
        # individual[staff_idx][day_idx]
        individual = [[-1 for _ in range(self.D)] for _ in range(self.N)]

        # Assign fixed off days first
        for d in range(self.D):
            for i in range(self.N):
                if self.IS_PRE_ASSIGNED_OFF[i][d]:
                    individual[i][d] = 0

        # Schedule for each day
        for d in range(self.D):
            # If worked night shift yesterday -> off today (if not already assigned)
            if d > 0:
                for i in range(self.N):
                    if individual[i][d - 1] == 4:  # Night shift yesterday
                        if individual[i][d] == -1:  # Not already assigned
                            individual[i][d] = 0
        
            # Get staff who haven't been assigned yet for day d
            available_staff_for_assignment = [i for i in range(self.N) if individual[i][d] == -1]
            random.shuffle(available_staff_for_assignment)

            # Assign Night shift (4)
            # Prioritize those not affected by IS_PRE_ASSIGNED_OFF[i][d] (already handled)
            # and not needing rest after night shift (already handled)
            night_shift_candidates = available_staff_for_assignment.copy()
            num_to_assign_night = min(self.A, len(night_shift_candidates))
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

            # Assign Day shifts (1, 2, 3)
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
            
            # Assign remaining unassigned staff (-1) to off (0)
            for staff_idx in range(self.N):
                if individual[staff_idx][d] == -1:
                    individual[staff_idx][d] = 0
        
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] == -1:
                    if self.verbose:
                        print(f"CRITICAL ERROR in _create_initial_solution: Cell not assigned [{n}][{d}]! Setting default to 0.")
                    individual[n][d] = 0
        return individual
    
    def _calculate_fitness(self, individual):
        # Check fixed off day violations
        violation_fixed_off = 0
        for n in range(self.N):
            for d in range(self.D):
                if individual[n][d] > 0 and self.IS_PRE_ASSIGNED_OFF[n][d]:
                    violation_fixed_off += 1

        # Check coverage violations
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

        # Check night shift followed by work violations
        violation_night_then_off = 0
        if self.D > 1:
            for n in range(self.N):
                for d in range(self.D - 1):
                    if individual[n][d] == 4 and individual[n][d+1] != 0:
                        violation_night_then_off += 1

        # Calculate max night shifts per staff (objective)
        night_shifts_per_staff = [0] * self.N
        for n in range(self.N):
            for d in range(self.D): 
                if individual[n][d] == 4:
                    night_shifts_per_staff[n] += 1
        
        max_night_shifts_objective = 0.0
        if self.N > 0 and night_shifts_per_staff: 
            max_night_shifts_objective = float(max(night_shifts_per_staff)) if night_shifts_per_staff else 0.0

        # Weights for different violations
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
            for _ in range(max_attempts_one_swap):
                if self.N < 2 and num_swaps_to_perform > 0:  # Need at least 2 staff to swap
                    return neighbor

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
                
                # Find a different staff than n1 to swap with
                possible_n2_candidates = [n2 for n2 in staff_on_off_duty_today_can_work_night if n2 != n1_idx]
                if not possible_n2_candidates:
                    continue  # No n2 different from n1 found
                
                n2_idx = random.choice(possible_n2_candidates)
                
                neighbor[n1_idx][day_idx] = 0 
                neighbor[n2_idx][day_idx] = 4 
                swaps_actually_done += 1
                break  # Break after successful swap
        
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
        
        neighbor[n_violating][d_violating_work] = 0  # Force off
        return neighbor
    
    def _generate_neighbor(self, current_solution, num_swaps_for_swap_op=1):
        operators = [
            (self._swap_night_with_off_neighbor, 0.70),  # Pass num_swaps_for_swap_op here     
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
    
    def _calculate_num_swaps_for_op(self, temperature):
        """Calculate the number of swaps based on current temperature."""
        if self.initial_swap_percentage_k <= 0 or self.N <= 0:
            return 1
            
        if self.initial_temperature > self.min_temperature: 
            temp_ratio = (temperature - self.min_temperature) / (self.initial_temperature - self.min_temperature)
            temp_ratio = max(0, min(1, temp_ratio)) 
        else: 
            temp_ratio = 1.0 if temperature >= self.initial_temperature else 0.0
        
        return max(1, int(round(self.max_swaps_at_start_calculated * temp_ratio)))
    
    def _local_search(self,
                            initial_solution_func, 
                            fitness_func,          
                            neighbor_func,        
                            max_iterations, 
                            max_iterations_no_improvement, 
                            time_limit_for_run_secs=None):
        run_start_time = time.time()
        current_solution = initial_solution_func()
        current_fitness = fitness_func(current_solution)
        
        best_solution_in_run = copy.deepcopy(current_solution) 
        best_fitness_in_run = current_fitness
        temperature = self.initial_temperature

        iterations_without_improvement_in_best = 0

        for iteration in range(max_iterations):
            # Check time limit
            if time_limit_for_run_secs and (time.time() - run_start_time) > time_limit_for_run_secs:
                break
            
            # Calculate how many swaps to perform based on temperature
            num_swaps_for_op = self._calculate_num_swaps_for_op(temperature)

            # Generate multiple neighbor solutions and select the best one
            best_neighbor = None
            best_neighbor_fitness = -float('inf')
            
            for _ in range(self.num_neighbors_to_explore_per_step):
                neighbor = neighbor_func(current_solution, num_swaps_for_op)
                neighbor_fitness = fitness_func(neighbor)
                
                if neighbor_fitness > best_neighbor_fitness:
                    best_neighbor = copy.deepcopy(neighbor)
                    best_neighbor_fitness = neighbor_fitness
            
            # Decide whether to accept the best neighbor solution
            if best_neighbor_fitness >= current_fitness:
                # Always accept better or equal solutions
                current_solution = copy.deepcopy(best_neighbor)
                current_fitness = best_neighbor_fitness
                
                # Update best solution if improved
                if current_fitness > best_fitness_in_run:
                    best_fitness_in_run = current_fitness
                    best_solution_in_run = copy.deepcopy(current_solution)
                    iterations_without_improvement_in_best = 0
                else:
                    iterations_without_improvement_in_best += 1
            else:
                # For worse solutions, accept with a probability based on temperature
                delta_fitness = best_neighbor_fitness - current_fitness
                if temperature > 1e-9:
                    acceptance_probability = math.exp(delta_fitness / temperature)
                    if random.random() < acceptance_probability:
                        current_solution = copy.deepcopy(best_neighbor)
                        current_fitness = best_neighbor_fitness
                
                iterations_without_improvement_in_best += 1
                
            # Cool down temperature
            temperature = max(self.min_temperature, temperature * self.cooling_rate)

            # Break if no improvement for a while
            if iterations_without_improvement_in_best >= max_iterations_no_improvement:
                break
            
        run_time = time.time() - run_start_time
        return best_solution_in_run, best_fitness_in_run, run_time
    
    def _run_multi_start_search(self, overall_time_limit_secs_for_this_solve=None):
        overall_start_time = time.time()
        best_overall_solution = None
        best_overall_fitness = -float('inf') 

        for restart_num in range(self.num_restarts):
            if overall_time_limit_secs_for_this_solve and \
               (time.time() - overall_start_time) > overall_time_limit_secs_for_this_solve:
                break
            
            time_left_for_sa_run = None
            if overall_time_limit_secs_for_this_solve:
                elapsed_time = time.time() - overall_start_time
                time_left_for_sa_run = overall_time_limit_secs_for_this_solve - elapsed_time
                if time_left_for_sa_run <= 0:
                    break
            
            solution_from_sa, fitness_from_sa, run_time_one_restart = self._local_search(
                initial_solution_func=self._create_initial_solution,
                fitness_func=self._calculate_fitness,
                neighbor_func=self._generate_neighbor, 
                max_iterations=self.max_iterations_per_restart,
                max_iterations_no_improvement=self.max_no_improvement_per_restart,
                time_limit_for_run_secs=time_left_for_sa_run
            )

            if fitness_from_sa > best_overall_fitness:
                best_overall_fitness = fitness_from_sa
                best_overall_solution = copy.deepcopy(solution_from_sa) 
                if self.verbose:
                    print(f"Restart {restart_num + 1}: NEW BEST overall fitness = {best_overall_fitness:.4f} (run time: {run_time_one_restart:.2f}s)")
            else:
                if self.verbose:
                    print(f"Restart {restart_num + 1}: No improvement (fitness {fitness_from_sa:.4f}). Best overall fitness still {best_overall_fitness:.4f} (run time: {run_time_one_restart:.2f}s)")

        total_time_taken = time.time() - overall_start_time
        return best_overall_solution, best_overall_fitness, total_time_taken
    


    def solve(self, N, D, A, B, pre_assigned_off_list_of_lists, time_limit_for_solve_secs=None):
        self.N = N
        self.D = D
        self.A = A
        self.B = B
        
        if self.N > 0:
             self.max_swaps_at_start_calculated = max(1, int(self.N * self.initial_swap_percentage_k / 100.0))
             self.max_swaps_at_start_calculated = min(self.max_swaps_at_start_calculated, self.max_swaps_per_op_cap)
        else:
            self.max_swaps_at_start_calculated = 1

        # Initialize pre-assigned off days
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

        # Handle edge cases
        if self.N == 0:  # Handle N=0 case (empty output file, no runtime)
            return [], 0, 0.00  # Return empty solution
        
        if self.D == 0:  # Handle D=0 case (N empty lines, with runtime)
            empty_solution_d0 = [[] for _ in range(self.N)]
            actual_runtime = 0.00 
            return empty_solution_d0, 0, actual_runtime

        # Normal case: run the algorithm
        best_solution, best_fitness, run_time = self._run_multi_start_search(
            overall_time_limit_secs_for_this_solve=time_limit_for_solve_secs
        )

        if best_solution is None: 
            if self.N > 0 and self.D > 0:  # Only when N and D > 0 and no solution found
                 best_solution = [[0 for _ in range(self.D)] for _ in range(self.N)]
        
        return best_solution, best_fitness, run_time

def handle_output(solution, fitness, run_time, N, D, testcase_filename_original, output_dir, is_empty_case=False):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_filename = testcase_filename_original.replace(".txt", ".out")
    output_filepath = os.path.join(output_dir, output_filename)
    
    with open(output_filepath, "w") as f:
        if N == 0:  # Case N=0, empty output file (no runtime)
            pass 
        else:  # N > 0
            if solution is not None:  # N > 0, has solution (could be D=0 or D>0)
                for n_idx in range(N):
                    # If D=0, solution[n_idx] is [], join results in "" (empty line)
                    shifts_for_staff_n = [str(solution[n_idx][d_idx]) for d_idx in range(D)]
                    line = " ".join(shifts_for_staff_n)
                    f.write(line + "\n")
            else:  # N > 0, but solution is None (e.g., algorithm failed)
                    # Write N empty lines
                for _ in range(N):
                    f.write("\n")
            f.write(f"{run_time:.2f}\n")  # Write runtime for all cases where N > 0

if __name__ == "__main__":
    
    testcase_dir = "Testcase" 
    output_dir = f"Output_ngocanhh"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    testcases_files = [f for f in os.listdir(testcase_dir) if f.endswith(".txt")]
    if not testcases_files:
        print(f"NO .txt files found in {testcase_dir}. Please check the directory.")
    testcases_files.sort()
    
    solver_config = {
        "num_restarts": 10,  
        "max_iterations_per_restart": 400, 
        "max_no_improvement_per_restart": 100, 
        "num_neighbors_to_explore_per_step": 10,
        "initial_temperature": 100.0,
        "cooling_rate": 0.99, 
        "min_temperature": 0.05,
        "initial_swap_percentage_k": 10.0, 
        "max_swaps_per_op_cap": 15,
        "verbose": False
    }
    TIME_LIMIT_PER_TESTCASE_SECS = 60

    # Initialize solver once with configuration
    solver = LocalSearchSolver(**solver_config)

    total_execution_start_time = time.time()
    for testcase_filename in testcases_files:
        print(f"--- Processing: {testcase_filename} ---")
        current_testcase_path = os.path.join(testcase_dir, testcase_filename)

        pre_assigned_off_data_local = []
        try:
            with open(current_testcase_path, 'r') as f:
                N_local, D_local, A_local, B_local = map(int, f.readline().split())
                if N_local > 0: 
                    for _ in range(N_local):
                        line = f.readline()
                        if line.strip(): 
                            days_off_input = list(map(int, line.split()))
                            pre_assigned_off_data_local.append(days_off_input)
                        else: 
                            pre_assigned_off_data_local.append([-1]) 
        except Exception as e:
            print(f"ERROR reading file {testcase_filename}: {e}")
            continue 
        
        # Call solver.solve() for each testcase
        solution, fitness, run_time = solver.solve(
            N=N_local, 
            D=D_local, 
            A=A_local, 
            B=B_local,
            pre_assigned_off_list_of_lists=pre_assigned_off_data_local,
            time_limit_for_solve_secs=TIME_LIMIT_PER_TESTCASE_SECS
        )
        
        # Handle output using the standalone function
        handle_output(
            solution=solution,
            fitness=fitness,
            run_time=run_time,
            N=N_local,
            D=D_local,
            testcase_filename_original=testcase_filename,
            output_dir=output_dir,
            is_empty_case=(N_local == 0 or D_local == 0)
        )
    total_execution_end_time = time.time()
    print(f"\n--- ALL TESTCASES COMPLETED ---")
    print(f"Total execution time: {total_execution_end_time - total_execution_start_time:.2f} seconds.")
    print(f"Results saved in directory: {output_dir}")