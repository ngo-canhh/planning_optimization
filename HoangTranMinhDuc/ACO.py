from collections import defaultdict
import random
import numpy as np
import time

class ACO_D:
    def __init__(self):
        pass
    
    def simple_construction(self, N, D, A, days_off):
        """Xây dựng lời giải đơn giản và nhanh"""
        solution = np.zeros((N, D), dtype=np.int8)
        
        for d in range(D):
            # Tìm nhân viên khả dụng
            available = []
            for i in range(N):
                if d not in days_off[i] and (d == 0 or solution[i, d-1] != 4):
                    available.append(i)
            
            if len(available) == 0:
                continue
            
            # Phân ca đơn giản
            shifts_needed = min(4 * A, len(available))
            random.shuffle(available)
            
            shift_idx = 0
            for i in range(shifts_needed):
                shift = (shift_idx % 4) + 1
                solution[available[i], d] = shift
                if (i + 1) % A == 0:
                    shift_idx += 1
        
        return solution.tolist()
    
    def fast_local_optimization(self, solution, N, D, A, B, days_off):
        """Tối ưu cục bộ nhanh - chỉ sửa vi phạm nghiêm trọng"""
        try:
            solution_array = np.array(solution, dtype=np.int8)
            
            # 1. Sửa vi phạm ràng buộc nghỉ sau ca đêm
            for i in range(N):
                for d in range(D-1):
                    if solution_array[i, d] == 4 and solution_array[i, d+1] > 0:
                        solution_array[i, d+1] = 0
            
            # 2. Đảm bảo đủ nhân viên cho mỗi ca (chỉ ca đêm)
            for d in range(D):
                night_count = np.sum(solution_array[:, d] == 4)
                if night_count < A:
                    needed = A - night_count
                    candidates = []
                    
                    for i in range(N):
                        if (d not in days_off[i] and 
                            solution_array[i, d] == 0 and 
                            (d == 0 or solution_array[i, d-1] != 4)):
                            candidates.append(i)
                    
                    for i in candidates[:needed]:
                        solution_array[i, d] = 4
            
            return solution_array.tolist()
        except:
            return solution
    
    def evaluate_simple(self, solution):
        """Đánh giá đơn giản chỉ dựa trên số ca đêm tối đa"""
        try:
            solution_array = np.array(solution)
            night_shifts = np.sum(solution_array == 4, axis=1)
            max_nights = max(night_shifts) if len(night_shifts) > 0 else 0
            return 1000.0 / (max_nights + 1.0)
        except:
            return 0.01
    
    def fast_aco(self, N, D, A, B, days_off, time_limit):
        """ACO nhanh với giới hạn thời gian"""
        start_time = time.time()
        
        # Tham số tối ưu cho tốc độ
        if N <= 50:
            n_ants = 8
            max_iterations = 30
        elif N <= 100:
            n_ants = 6
            max_iterations = 20
        else:
            n_ants = 4
            max_iterations = 10
        
        best_solution = None
        best_quality = 0
        
        # Tạo lời giải ban đầu nhanh
        for _ in range(3):
            if time.time() - start_time > time_limit * 0.1:
                break
            
            solution = self.simple_construction(N, D, A, days_off)
            quality = self.evaluate_simple(solution)
            
            if quality > best_quality:
                best_solution = [row[:] for row in solution]
                best_quality = quality
        
        # ACO đơn giản
        pheromone = defaultdict(lambda: 1.0)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit * 0.8:
                break
            
            iteration_best = None
            iteration_quality = 0
            
            for ant in range(n_ants):
                if time.time() - start_time > time_limit * 0.75:
                    break
                
                solution = self.simple_construction(N, D, A, days_off)
                
                # Tối ưu nhanh với xác suất 30%
                if random.random() < 0.3:
                    solution = self.fast_local_optimization(solution, N, D, A, B, days_off)
                
                quality = self.evaluate_simple(solution)
                
                if quality > iteration_quality:
                    iteration_best = solution
                    iteration_quality = quality
                    
                    if quality > best_quality:
                        best_solution = [row[:] for row in solution]
                        best_quality = quality
            
            # Cập nhật pheromone đơn giản
            if iteration_best:
                for i in range(N):
                    for d in range(D):
                        s = iteration_best[i][d]
                        if s > 0:
                            pheromone[(i, d, s)] *= 1.1
        
        return best_solution, best_quality

def solve_aco_fast(N, D, A, B, F):
    """Giải ACO nhanh với giới hạn thời gian"""
    start_time = time.time()
    time_limit = 290  # Để lại 9 giây buffer
    
    # Chuyển đổi F thành days_off (0-indexed)
    days_off = []
    for i in range(N):
        staff_days_off = set()
        for day in F[i]:
            if 1 <= day <= D:
                staff_days_off.add(day - 1)
        days_off.append(staff_days_off)
    
    aco = ACO_D()
    
    # Kiểm tra kích thước bài toán
    if N * D > 50000:  # Bài toán rất lớn
        # Chỉ dùng greedy đơn giản
        solution = aco.simple_construction(N, D, A, days_off)
        solution = aco.fast_local_optimization(solution, N, D, A, B, days_off)
    else:
        # Dùng ACO với giới hạn thời gian
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 10:
            solution, _ = aco.fast_aco(N, D, A, B, days_off, remaining_time)
        else:
            solution = aco.simple_construction(N, D, A, days_off)
    
    # Tối ưu cuối nếu còn thời gian
    if time.time() - start_time < time_limit - 5:
        solution = aco.fast_local_optimization(solution, N, D, A, B, days_off)
    
    return solution

def create_fallback_solution(N, D, F):
    """Tạo lời giải dự phòng nhanh"""
    solution = []
    for i in range(N):
        row = []
        for d in range(D):
            if (d + 1) in F[i]:
                row.append(0)  # Ngày nghỉ
            else:
                row.append((i % 4) + 1)  # Gán ca tuần hoàn
        solution.append(row)
    return solution

if __name__ == "__main__":
    try:
        start_total = time.time()
        
        # Đọc input nhanh
        N, D, A, B = map(int, input().split())
        F = []

        for i in range(N):
            try:
                days = list(map(int, input().split()))
                valid_days = [d for d in days if d != -1 and 1 <= d <= D]
                F.append(valid_days)
            except:
                F.append([])
        
        # Kiểm tra thời gian đã sử dụng
        if time.time() - start_total > 280:
            # Quá gần time limit, dùng fallback
            solution = create_fallback_solution(N, D, F)
        else:
            # Giải bình thường
            solution = solve_aco_fast(N, D, A, B, F)
        
        # In kết quả
        if solution is not None:
            for row in solution:
                print(" ".join(map(str, row)))
        else:
            solution = create_fallback_solution(N, D, F)
            for row in solution:
                print(" ".join(map(str, row)))
                
    except Exception as e:
        # Emergency fallback
        try:
            solution = create_fallback_solution(N, D, F)
            for row in solution:
                print(" ".join(map(str, row)))
        except:
            # Last resort - print basic pattern
            for i in range(5):  # Assume minimum size
                print("1 2 3 4 0 1 2")