"""Có N nhân viên 1,2,…, N cần được xếp ca trực làm việc cho các ngày 1,  2, …, D. Mỗi ngày được chia thành 4 kíp: sáng, trưa, chiều, đêm. Biết rằng:
Mỗi ngày, một nhân viên chỉ làm nhiều nhất 1 ca 
Ngày hôm trước làm ca đêm thì hôm sau được nghỉ
Mỗi ca trong mỗi ngày có ít nhất A nhân viên và nhiều nhất B nhân viên 
F(i): danh sách các ngày nghỉ phép của nhân viên i 
Xây dựng phương án xếp ca trực cho N nhân viên sao cho
Số ca đêm nhiều nhất phân cho 1 nhân viên nào đó là nhỏ nhất
A solution is represented by a matrix X[1..N][1..D] in which x[i][d] is the shift scheduled to staff i on day d (value 1 means shift morning; value 2 means shift afternoon; value 3 means shift evening; value 4 means shift night; value 0 means day-off)
Input
Line 1: contains 4 positive integers N, D, A, B (1 <= N <= 500, 1 <= D <= 200, 1 <= A <= B <= 500)
Line i + 1 (i = 1, 2, . . ., N): contains a list of positive integers which are the day off of the staff i (days are indexed from 1 to D), terminated by -1
 
Output
Line i (i = 1, 2, . . ., N): write the i
th
 row of the solution matrix X
"""

from collections import defaultdict
import random
import numpy as np
import time
import os

class ACO_D:
    def __init__(self):
        pass
    
    def read_input_from_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        N, D, A, B = map(int, lines[0].split())
        days_off = []
        
        for i in range(1, N+1):
            day_list = list(map(int, lines[i].split()))
            days = []
            for day in day_list:
                if day == -1:
                    break
                days.append(day - 1)  # Chuyển sang index 0-based
            days_off.append(days)
        
        return N, D, A, B, days_off
        
    def improved_solution_construction(self, N, D, A, days_off, pheromone, alpha, beta):
        try:
            solution = np.zeros((N, D), dtype=np.int8)
            
            # Sắp xếp các ngày theo thứ tự khó giảm dần để xử lý trước
            day_complexities = []
            for d in range(D):
                # Số nhân viên khả dụng cho ngày này
                available_count = sum(1 for i in range(N) if d not in days_off[i])
                day_complexities.append((d, available_count))
            
            # Sắp xếp theo số nhân viên khả dụng tăng dần (ngày khó xếp trước)
            day_order = [d for d, _ in sorted(day_complexities, key=lambda x: x[1])]
            
            # Xây dựng lời giải theo thứ tự khó trước dễ sau
            for d in day_order:
                # Xác định nhân viên không thể làm việc vì ca đêm hôm trước
                night_shifts_prev = np.zeros(N, dtype=bool)
                if d > 0:
                    night_shifts_prev = (solution[:, d-1] == 4)
                
                # Ưu tiên ca đêm trước để đảm bảo đủ người
                shift_order = [4, 3, 2, 1]  # Ưu tiên ca đêm, chiều, trưa, sáng
                
                for s in shift_order:
                    # Xác định nhân viên khả dụng
                    available = np.ones(N, dtype=bool)
                    
                    # Nhân viên nghỉ phép
                    for i in range(N):
                        if d in days_off[i]:
                            available[i] = False
                    
                    # Nhân viên đã được phân công ca khác trong ngày này
                    assigned_today = (solution[:, d] > 0)
                    available &= ~assigned_today
                    
                    # Nhân viên làm ca đêm ngày hôm trước
                    available &= ~night_shifts_prev
                    
                    available_staff = np.where(available)[0]
                    
                    if len(available_staff) == 0:
                        continue
                    
                    # Đảm bảo mỗi ca có ít nhất A nhân viên nếu có thể
                    required_staff = min(A, len(available_staff))
                    
                    # Với bài toán lớn, cân nhắc phân phối đều nhân viên
                    if N > 100:
                        # Đếm số ca đã làm của mỗi nhân viên
                        shift_counts = np.sum(solution > 0, axis=1)
                        night_counts = np.sum(solution == 4, axis=1)
                        
                        # Tính heuristic và pheromone
                        probabilities = []
                        for i in available_staff:
                            # Heuristic cao hơn cho người có ít ca làm việc
                            eta = 1.0
                            if s == 4:  # Ca đêm - ưu tiên người ít ca đêm
                                eta = 1.0 / (night_counts[i] + 1)
                            else:  # Ca khác - cân bằng tổng số ca
                                eta = 1.0 / (shift_counts[i] + 1)
                            
                            tau = pheromone.get((i, d, s), 0.01)
                            probabilities.append((tau ** alpha) * (eta ** beta))
                        
                        # Chuẩn hóa xác suất
                        sum_prob = sum(probabilities)
                        if sum_prob > 0:
                            probabilities = [p / sum_prob for p in probabilities]
                            
                            # Chọn nhân viên dựa trên xác suất
                            try:
                                chosen_indices = np.random.choice(
                                    len(available_staff), 
                                    size=required_staff,
                                    replace=False,
                                    p=probabilities
                                )
                                for chosen_idx in chosen_indices:
                                    chosen_staff = available_staff[chosen_idx]
                                    solution[chosen_staff, d] = s
                            except:
                                # Fallback: sắp xếp theo số ca đã làm
                                sorted_staff = sorted([(i, shift_counts[i]) for i in available_staff], 
                                                    key=lambda x: x[1])
                                for i, _ in sorted_staff[:required_staff]:
                                    solution[i, d] = s
                    else:
                        # Cách tiếp cận hiện tại tốt cho bài toán nhỏ và vừa
                        probabilities = np.zeros(len(available_staff))
                        work_days = np.sum(solution > 0, axis=1)
                        same_shifts = np.sum(solution == s, axis=1)
                        
                        for idx, i in enumerate(available_staff):
                            eta = 1.0 / (work_days[i] + same_shifts[i] + 1)
                            tau = pheromone.get((i, d, s), 0.01)
                            probabilities[idx] = (tau ** alpha) * (eta ** beta)
                        
                        if probabilities.sum() > 0:
                            probabilities = probabilities / probabilities.sum()
                            try:
                                chosen_indices = np.random.choice(
                                    len(available_staff), 
                                    size=min(required_staff, len(available_staff)),
                                    replace=False,
                                    p=probabilities
                                )
                                for chosen_idx in chosen_indices:
                                    solution[available_staff[chosen_idx], d] = s
                            except:
                                # Fallback nếu gặp vấn đề với np.random.choice
                                available_list = list(available_staff)
                                random.shuffle(available_list)
                                for i in available_list[:required_staff]:
                                    solution[i, d] = s
            
            # Kiểm tra và đảm bảo mỗi nhân viên làm đủ số ngày tối thiểu (A)
            staff_work_days = np.sum(solution > 0, axis=1)
            for i in np.where(staff_work_days < A)[0]:
                days_needed = A - staff_work_days[i]
                if days_needed <= 0:
                    continue
                    
                # Tìm các ngày có thể gán thêm
                potential_days = []
                for d in range(D):
                    if solution[i, d] == 0 and d not in days_off[i]:
                        # Kiểm tra điều kiện ca đêm hôm trước
                        if d > 0 and solution[i, d-1] == 4:
                            continue
                        potential_days.append(d)
                
                if potential_days:
                    # Ưu tiên các ngày có ít nhân viên
                    day_loads = [(d, np.sum(solution[:, d] > 0)) for d in potential_days]
                    day_loads.sort(key=lambda x: x[1])  # Sắp xếp theo số nhân viên tăng dần
                    
                    for d, _ in day_loads[:days_needed]:
                        # Ưu tiên các ca có ít nhân viên hơn
                        shift_counts = np.bincount(solution[:, d], minlength=5)
                        shift_counts[0] = float('inf')  # Không tính ca nghỉ
                        best_shift = np.argmin(shift_counts[1:5]) + 1  # Chọn ca có ít người nhất
                        solution[i, d] = best_shift
            
            return solution.tolist()
            
        except Exception as e:
            # Fallback đơn giản
            fallback = [[0 for _ in range(D)] for _ in range(N)]
            for i in range(N):
                for d in range(D):
                    if d not in days_off[i]:
                        fallback[i][d] = (i + d) % 4 + 1
            return fallback
    
    def enhanced_local_optimization(self, solution, N, D, A, B, days_off):
        try:
            solution_array = np.array(solution, dtype=np.int8)
            improved = True
            max_iterations = 5 if N > 200 else 10  # Giảm số vòng lặp cho bài toán cực lớn
            iterations = 0
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Sửa vi phạm ràng buộc nghỉ sau ca đêm
                for i in range(N):
                    for d in range(D-1):
                        if solution_array[i, d] == 4 and solution_array[i, d+1] > 0:
                            solution_array[i, d+1] = 0  # Buộc nghỉ ngày hôm sau
                            improved = True
                
                # 2. Đảm bảo đủ nhân viên cho mỗi ca (ít nhất A)
                for d in range(D):
                    shift_counts = np.bincount(solution_array[:, d], minlength=5)
                    
                    for s in range(1, 5):
                        current_count = shift_counts[s]
                        if current_count < A:
                            # Tìm thêm nhân viên cho ca này
                            needed = A - current_count
                            candidates = []
                            
                            for i in range(N):
                                # Kiểm tra các ràng buộc
                                if d in days_off[i] or solution_array[i, d] > 0:
                                    continue
                                if d > 0 and solution_array[i, d-1] == 4:
                                    continue
                                if s == 4 and d < D-1 and solution_array[i, d+1] > 0:
                                    continue
                                    
                                # Tính ưu tiên (càng thấp càng tốt)
                                night_count = np.sum(solution_array[i] == 4)
                                work_days = np.sum(solution_array[i] > 0)
                                
                                if s == 4:  # Ca đêm
                                    priority = night_count * 10  # Ưu tiên thấp cho người nhiều ca đêm
                                else:
                                    priority = work_days
                                
                                candidates.append((i, priority))
                            
                            # Sắp xếp theo ưu tiên tăng dần
                            candidates.sort(key=lambda x: x[1])
                            
                            # Gán thêm nhân viên
                            for i, _ in candidates[:needed]:
                                solution_array[i, d] = s
                                improved = True
                
                # 3. Đảm bảo mỗi ca không quá B nhân viên
                for d in range(D):
                    shift_counts = np.bincount(solution_array[:, d], minlength=5)
                    
                    for s in range(1, 5):
                        if shift_counts[s] > B:
                            # Cần bớt nhân viên
                            excess = shift_counts[s] - B
                            staff_in_shift = [i for i in range(N) if solution_array[i, d] == s]
                            
                            # Ưu tiên loại bỏ nhân viên có nhiều ca làm việc
                            staff_priorities = []
                            for i in staff_in_shift:
                                if s == 4:  # Ca đêm
                                    night_count = np.sum(solution_array[i] == 4)
                                    staff_priorities.append((i, night_count))
                                else:
                                    work_count = np.sum(solution_array[i] > 0)
                                    staff_priorities.append((i, work_count))
                            
                            # Sắp xếp theo số ca giảm dần
                            staff_priorities.sort(key=lambda x: x[1], reverse=True)
                            
                            # Loại bỏ nhân viên thừa
                            for i, _ in staff_priorities[:excess]:
                                solution_array[i, d] = 0  # Cho nghỉ
                                improved = True
                
                # 4. Cân đối ca đêm
                if iterations == 1:  # Chỉ làm một lần
                    night_counts = np.sum(solution_array == 4, axis=1)
                    if night_counts.size > 0:  # Kiểm tra mảng không rỗng
                        max_nights = np.max(night_counts)
                        if max_nights > 0:
                            # Tìm nhân viên có nhiều ca đêm nhất
                            staff_with_max = np.where(night_counts == max_nights)[0]
                            
                            for i in staff_with_max:
                                # Tìm nhân viên có ít ca đêm để chuyển giao
                                night_shifts = np.where(solution_array[i] == 4)[0]
                                
                                for d in night_shifts:
                                    # Tìm nhân viên có thể làm ca đêm này
                                    candidates = []
                                    for j in range(N):
                                        if j == i or d in days_off[j] or solution_array[j, d] > 0:
                                            continue
                                        if d > 0 and solution_array[j, d-1] == 4:
                                            continue
                                        if d < D-1 and solution_array[j, d+1] > 0:
                                            continue
                                            
                                        night_count_j = night_counts[j]
                                        if night_count_j < max_nights - 1:
                                            candidates.append((j, night_count_j))
                                    
                                    # Sắp xếp theo số ca đêm tăng dần
                                    candidates.sort(key=lambda x: x[1])
                                    
                                    if candidates:
                                        chosen_j = candidates[0][0]
                                        # Chuyển ca đêm sang nhân viên khác
                                        solution_array[chosen_j, d] = 4
                                        solution_array[i, d] = 0
                                        improved = True
                                        break  # Chỉ chuyển 1 ca mỗi lần
            
            return solution_array.tolist()
            
        except Exception as e:
            return solution
    
    def simple_construction(self, N, D, A, B, days_off):
        """Tạo lời giải đơn giản để sử dụng trong trường hợp có lỗi"""
        solution = [[0 for _ in range(D)] for _ in range(N)]
        
        # Phân bố ca làm việc đều 
        for d in range(D):
            staff_per_shift = max(1, min(N // 4, B))  # Chia đều nhân viên cho 4 ca
            
            staff_idx = 0
            for s in range(1, 5):
                count = 0
                while count < staff_per_shift and staff_idx < N:
                    if d not in days_off[staff_idx]:
                        solution[staff_idx][d] = s
                        count += 1
                    staff_idx += 1
                    if staff_idx >= N:
                        staff_idx = 0  # Quay lại từ đầu nếu cần
                        
            # Đảm bảo mỗi ca có ít nhất A nhân viên
            for s in range(1, 5):
                count = sum(1 for i in range(N) if solution[i][d] == s)
                while count < A:
                    for i in range(N):
                        if solution[i][d] == 0 and d not in days_off[i]:
                            solution[i][d] = s
                            count += 1
                            break
                    else:
                        break  # Không tìm thêm được nhân viên
        
        return solution
    
    def evaluate_solution(self, solution, N, D, A, B, days_off):
        try:
            penalty = 0
            solution_array = np.array(solution)
            
            # 1. Ràng buộc ngày nghỉ phép
            for i in range(N):
                for day in days_off[i]:
                    if solution_array[i, day] > 0:
                        penalty += 1000
            
            # 2. Ràng buộc ca đêm liên tiếp
            for i in range(N):
                for d in range(D-1):
                    if solution_array[i, d] == 4 and solution_array[i, d+1] > 0:
                        penalty += 500
            
            # 3. Ràng buộc số nhân viên mỗi ca
            for d in range(D):
                shift_counts = np.bincount(solution_array[:, d], minlength=5)
                for s in range(1, 5):
                    # Phạt nếu ca không đủ hoặc quá nhiều nhân viên
                    if shift_counts[s] < A:
                        penalty += 200 * (A - shift_counts[s])
                    if shift_counts[s] > B:
                        penalty += 200 * (shift_counts[s] - B)
            
            # 4. Ràng buộc số ngày làm việc
            work_days = np.sum(solution_array > 0, axis=1)
            min_work_days = min(work_days) if len(work_days) > 0 else 0
            max_work_days = max(work_days) if len(work_days) > 0 else 0
            penalty += 50 * (max_work_days - min_work_days)
            
            # 5. Mục tiêu: giảm thiểu số ca đêm tối đa
            night_shifts = np.sum(solution_array == 4, axis=1)
            max_night_shifts = max(night_shifts) if len(night_shifts) > 0 else 0
            penalty += 100 * max_night_shifts
            
            # Chất lượng càng cao khi penalty càng thấp
            return 10000.0 / (penalty + 1.0)
            
        except Exception as e:
            return 0.01  # Trả về chất lượng thấp nếu có lỗi
    
    def efficient_max_min_ant_system(self, colony_id, N, D, A, B, days_off, n_ants, iterations, result_queue=None):
        try:
            # Điều chỉnh tham số dựa trên kích thước bài toán
            if N > 100:
                alpha = 1.5 + 0.1 * colony_id
                beta = 3.0 + 0.15 * colony_id
                rho = 0.05 + 0.01 * colony_id
            else:
                alpha = 1.0 + 0.2 * colony_id
                beta = 2.0 + 0.25 * colony_id
                rho = 0.02 + 0.01 * colony_id
            
            # Giới hạn pheromone
            tau_max = 5.0
            tau_min = 0.01
            
            # Khởi tạo pheromone với dictionary mặc định
            pheromone = defaultdict(lambda: tau_max)
            
            best_solution = None
            best_quality = 0
            stagnation_count = 0
            last_improvement_iteration = 0
            
            # Tạo lời giải ban đầu
            initial_solution = self.improved_solution_construction(N, D, A, days_off, pheromone, alpha, beta)
            best_solution = [row[:] for row in initial_solution]
            best_quality = self.evaluate_solution(best_solution, N, D, A, B, days_off)
            
            for it in range(iterations):
                iteration_best_solution = None
                iteration_best_quality = 0
                
                # Chỉ chạy optimization trên số kiến giảm dần
                effective_n_ants = max(3, n_ants - it // 5)
                
                # Early stopping nếu không cải thiện sau nhiều vòng lặp
                if it - last_improvement_iteration > 15:
                    if result_queue is not None:
                        result_queue.put((best_solution, best_quality, colony_id))
                    return best_solution, best_quality
                
                # Mỗi kiến xây dựng một lời giải
                for ant in range(effective_n_ants):
                    solution = self.improved_solution_construction(N, D, A, days_off, pheromone, alpha, beta)
                    
                    # Tối ưu cục bộ với xác suất giảm dần theo vòng lặp
                    if random.random() < max(0.1, 0.3 - it / (iterations * 2)):
                        solution = self.enhanced_local_optimization(solution, N, D, A, B, days_off)
                    
                    # Đánh giá lời giải
                    quality = self.evaluate_solution(solution, N, D, A, B, days_off)
                    
                    # Cập nhật lời giải tốt nhất vòng lặp
                    if quality > iteration_best_quality:
                        iteration_best_solution = [row[:] for row in solution]
                        iteration_best_quality = quality
                        
                        # Cập nhật lời giải tốt nhất toàn cầu
                        if quality > best_quality:
                            best_solution = [row[:] for row in solution]
                            best_quality = quality
                            stagnation_count = 0
                            last_improvement_iteration = it
                        else:
                            stagnation_count += 1
                
                # Bay hơi pheromone - chỉ cập nhật cho các phần tử có trong từ điển
                for key in list(pheromone.keys()):
                    pheromone[key] *= (1 - rho)
                    # Xóa các giá trị gần với tau_min để tiết kiệm bộ nhớ
                    if pheromone[key] < tau_min * 1.01:
                        del pheromone[key]
                
                # Chọn giải pháp để cập nhật pheromone
                if stagnation_count > 10:
                    # Diversification: sử dụng lời giải tốt nhất vòng lặp
                    update_solution = iteration_best_solution
                    update_quality = iteration_best_quality
                else:
                    # Exploitation: sử dụng lời giải tốt nhất toàn cầu
                    update_solution = best_solution
                    update_quality = best_quality
                
                # Cập nhật pheromone chọn lọc (chỉ cập nhật các thành phần đã sử dụng)
                if update_solution:
                    for i in range(N):
                        for d in range(D):
                            s = update_solution[i][d]
                            if s > 0:  # Chỉ cập nhật khi có ca làm việc
                                pheromone[(i, d, s)] = min(tau_max, pheromone[(i, d, s)] + update_quality)
                
                # Điều chỉnh giới hạn pheromone theo tiến trình (thực hiện ít hơn)
                if it % 15 == 0 and it > 0:
                    ratio = (iterations - it) / iterations
                    tau_max = 5.0 * ratio + 3.0 * (1 - ratio)
                    tau_min = 0.01 * ratio + 0.05 * (1 - ratio)
            
            if result_queue is not None:
                result_queue.put((best_solution, best_quality, colony_id))
            return best_solution, best_quality
        
        except Exception as e:
            # Tạo giải pháp đơn giản khi gặp lỗi
            fallback_solution = self.simple_construction(N, D, A, B, days_off)
            if result_queue is not None:
                result_queue.put((fallback_solution, 0.01, colony_id))
            return fallback_solution, 0.01
    
    def solve(self, N, D, A, B, dayoff):
        """
        Phương thức giải bài toán - đảm bảo tính đồng nhất với các thuật toán khác
        dayoff[i][j] = 1 => nhân viên i nghỉ vào ngày j
        """
        # Chuyển đổi định dạng dayoff thành dạng danh sách mà thuật toán ACO sử dụng
        days_off = []
        for i in range(1, N+1):
            staff_days_off = []
            for j in range(1, D+1):
                if dayoff[i][j] == 1:
                    # Chuyển sang index 0-based cho ACO
                    staff_days_off.append(j-1)
            days_off.append(staff_days_off)
        
        start = time.time()
        
        # Thiết lập tham số dựa trên kích thước bài toán
        if N < 15:
            n_ants = 5
            iterations = 30
        elif N < 30:
            n_ants = 10
            iterations = 50
        elif N < 100:
            n_ants = 8
            iterations = 40
        else:
            # Cho bài toán lớn, dùng tham số hiệu quả hơn
            n_ants = max(5, min(8, N // 50))
            iterations = max(20, min(30, D // 5))
        
        # Sử dụng single-process để đảm bảo độ ổn định
        solution, _ = self.efficient_max_min_ant_system(0, N, D, A, B, days_off, n_ants, iterations, None)
        
        # Kiểm tra và sửa chữa lời giải cuối cùng
        if N <= 200:  # Chỉ tối ưu cho bài toán không quá lớn
            solution = self.enhanced_local_optimization(solution, N, D, A, B, days_off)
        
        # Chuyển đổi lời giải 0-indexed thành 1-indexed để khớp với định dạng output
        result = np.zeros((N+1, D+1), dtype=int)
        for i in range(N):
            for d in range(D):
                result[i+1][d+1] = solution[i][d]
        
        end = time.time()
        
        return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        tc_num = int(sys.argv[1])
        filename = f'Testcase/tc{tc_num}.txt'
        
        aco = ACO_D()
        N, D, A, B, days_off = aco.read_input_from_file(filename)
        
        # Chuyển đổi định dạng days_off sang dayoff matrix
        dayoff = np.zeros((N+1, D+1))
        for i in range(N):
            for day in days_off[i]:
                dayoff[i+1][day+1] = 1
        
        # Bắt đầu tính thời gian
        start = time.time()
        
        # Giải bài toán
        solution = aco.solve(N, D, A, B, dayoff)
        
        # Kết thúc tính thời gian
        end = time.time()
        
        # Ghi kết quả ra file
        os.makedirs('Output', exist_ok=True)
        output_file = f'Output/output{tc_num}.txt'
        with open(output_file, 'w') as f:
            for i in range(1, N+1):
                line = ' '.join(str(solution[i][d]) for d in range(1, D+1))
                f.write(line + '\n')
            f.write(f"Execution time: {end - start:.2f} seconds\n")
