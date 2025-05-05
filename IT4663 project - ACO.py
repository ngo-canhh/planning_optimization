import random
import numpy as np
import time
import multiprocessing as mp

def read_input():
    line = input().strip()
    N, D, A, B = map(int, line.split())
    
    days_off = []
    for i in range(N):
        day_list = list(map(int, input().strip().split()))
        days = []
        for day in day_list:
            if day == -1:
                break
            days.append(day - 1)  # Chuyển sang index 0-based
        days_off.append(days)
    
    return N, D, A, B, days_off

def improved_solution_construction(N, D, A, days_off, pheromone, alpha, beta):
    """Xây dựng lời giải cải tiến theo ngày và ca"""
    solution = [[0 for _ in range(D)] for _ in range(N)]
    
    # Xây dựng lời giải theo ngày và ca
    for d in range(D):
        for s in range(1, 5):  # Mỗi ca làm việc (1-4)
            available_staff = []
            for i in range(N):
                # Nhân viên không nghỉ và không làm ca đêm ngày trước
                if d not in days_off[i] and (d == 0 or solution[i][d-1] != 4):
                    available_staff.append(i)
            
            # Số nhân viên cần cho mỗi ca
            required_staff = max(1, len(available_staff) // 4)
            assigned = 0
            
            # Sắp xếp nhân viên theo xác suất dựa trên pheromone
            if available_staff:
                probabilities = []
                for i in available_staff:
                    # Tính giá trị heuristic
                    work_days = sum(1 for day in range(D) if solution[i][day] > 0)
                    same_shifts = sum(1 for day in range(D) if solution[i][day] == s)
                    
                    # Heuristic cao hơn cho nhân viên có ít ngày làm việc
                    eta = 1.0 / (work_days + same_shifts + 1)
                    tau = pheromone[(i, d, s)]
                    
                    probabilities.append((tau ** alpha) * (eta ** beta))
                
                # Chuẩn hóa xác suất
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                    
                    # Chọn nhân viên theo xác suất cho đến khi đủ số lượng
                    while assigned < required_staff and available_staff:
                        if len(available_staff) == 1:
                            chosen_idx = 0
                        else:
                            try:
                                chosen_idx = np.random.choice(len(available_staff), p=probabilities)
                            except:
                                # Backup nếu có vấn đề với xác suất
                                chosen_idx = random.randint(0, len(available_staff)-1)
                        
                        chosen_staff = available_staff[chosen_idx]
                        
                        # Gán ca làm việc
                        solution[chosen_staff][d] = s
                        
                        # Cập nhật danh sách available
                        del available_staff[chosen_idx]
                        del probabilities[chosen_idx]
                        
                        # Chuẩn hóa lại xác suất nếu còn nhân viên
                        if probabilities:
                            total = sum(probabilities)
                            if total > 0:
                                probabilities = [p / total for p in probabilities]
                        
                        assigned += 1
    
    # Đảm bảo ràng buộc về số ngày làm việc
    for i in range(N):
        work_days = sum(1 for d in range(D) if solution[i][d] > 0)
        
        # Thêm ngày làm việc nếu chưa đủ A
        if work_days < A:
            days_to_add = A - work_days
            for d in range(D):
                if days_to_add <= 0:
                    break
                if solution[i][d] == 0 and d not in days_off[i] and (d == 0 or solution[i][d-1] != 4):
                    solution[i][d] = random.randint(1, 4)
                    days_to_add -= 1
    
    return solution

def local_optimization(solution, N, D, A, B, days_off):
    """Tối ưu cục bộ để sửa chữa và cải thiện lời giải"""
    improved = True
    max_iterations = 20
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Đảm bảo mỗi ca có ít nhất một nhân viên
        for d in range(D):
            shift_counts = [0, 0, 0, 0, 0]
            for i in range(N):
                shift_counts[solution[i][d]] += 1
            
            for s in range(1, 5):
                if shift_counts[s] == 0:
                    # Tìm nhân viên có thể chuyển sang ca này
                    for i in range(N):
                        if d not in days_off[i] and solution[i][d] != s and solution[i][d] != 0:
                            # Kiểm tra ràng buộc ca đêm
                            if (s != 4 or d == D-1 or solution[i][d+1] == 0) and \
                               (d == 0 or solution[i][d-1] != 4 or solution[i][d] == 0):
                                solution[i][d] = s
                                improved = True
                                break
                    
                    # Nếu không thể chuyển, tìm nhân viên đang nghỉ
                    if shift_counts[s] == 0:
                        for i in range(N):
                            if d not in days_off[i] and solution[i][d] == 0:
                                # Kiểm tra ràng buộc ca đêm
                                if (s != 4 or d == D-1 or solution[i][d+1] == 0) and \
                                   (d == 0 or solution[i][d-1] != 4):
                                    solution[i][d] = s
                                    improved = True
                                    break
        
        # Tối ưu số ngày làm việc
        for i in range(N):
            work_days = sum(1 for d in range(D) if solution[i][d] > 0)
            
            # Thêm ngày làm việc nếu chưa đủ A
            if work_days < A:
                for d in range(D):
                    if solution[i][d] == 0 and d not in days_off[i]:
                        if (d == 0 or solution[i][d-1] != 4) and (d == D-1 or solution[i][d+1] != 4):
                            # Tìm ca có ít nhân viên nhất
                            shift_counts = [0, 0, 0, 0, 0]
                            for j in range(N):
                                shift_counts[solution[j][d]] += 1
                            
                            best_shift = shift_counts.index(min(shift_counts[1:]), 1)
                            solution[i][d] = best_shift
                            work_days += 1
                            improved = True
                            if work_days >= A:
                                break
            
            # Giảm ngày làm việc nếu vượt quá B
            elif work_days > B:
                days_to_remove = work_days - B
                for d in range(D):
                    if solution[i][d] > 0 and days_to_remove > 0:
                        shift = solution[i][d]
                        shift_count = sum(1 for j in range(N) if solution[j][d] == shift)
                        
                        if shift_count > 1:  # Vẫn đảm bảo đủ nhân viên
                            solution[i][d] = 0
                            days_to_remove -= 1
                            improved = True
                            if days_to_remove == 0:
                                break
    
    return solution

def adaptive_evaluation(solution, N, D, A, B, days_off, iteration, max_iterations):
    """Đánh giá lời giải với trọng số thay đổi theo thời gian"""
    progress = min(1.0, iteration / max_iterations)
    
    # Trọng số thay đổi theo tiến trình
    w_days_off = 1000 + 500 * progress
    w_consecutive = 500 + 200 * progress
    w_work_days = 100 + 400 * progress
    w_coverage = 200 * (1 - progress) + 500 * progress
    w_balance = 50 * (1 - progress) + 150 * progress
    
    penalty = 0
    
    # Ràng buộc ngày nghỉ
    for i in range(N):
        for day in days_off[i]:
            if solution[i][day] > 0:
                penalty += w_days_off
    
    # Ràng buộc ca đêm liên tiếp
    for i in range(N):
        for d in range(D-1):
            if solution[i][d] == 4 and solution[i][d+1] > 0:
                penalty += w_consecutive
    
    # Ràng buộc số ngày làm việc
    for i in range(N):
        work_days = sum(1 for d in range(D) if solution[i][d] > 0)
        if work_days < A:
            penalty += w_work_days * (A - work_days)
        elif work_days > B:
            penalty += w_work_days * (work_days - B)
    
    # Đủ nhân viên mỗi ca
    for d in range(D):
        for s in range(1, 5):
            staff_count = sum(1 for i in range(N) if solution[i][d] == s)
            if staff_count < 1:
                penalty += w_coverage
    
    # Cân bằng ca làm việc
    for i in range(N):
        shift_counts = [0] * 5
        for d in range(D):
            shift_counts[solution[i][d]] += 1
        
        # Độ lệch của số ca mỗi loại
        working_shifts = shift_counts[1:]
        if sum(working_shifts) > 0:
            mean = sum(working_shifts) / 4
            variance = sum((count - mean) ** 2 for count in working_shifts) / 4
            std_dev = variance ** 0.5
            penalty += w_balance * std_dev
    
    # Số ca đêm tối đa
    max_night = max(sum(1 for d in range(D) if solution[i][d] == 4) for i in range(N))
    penalty += 50 * max_night * progress
    
    return 1.0 / (penalty + 1)  # Chất lượng: cao = tốt

def max_min_ant_system(colony_id, N, D, A, B, days_off, n_ants=10, iterations=50, result_queue=None):
    """MAX-MIN Ant System với các giới hạn pheromone"""
    # Tham số
    alpha = 1.0 + 0.2 * colony_id  # Khác nhau giữa các quần thể
    beta = 2.0 + 0.25 * colony_id
    rho = 0.02 + 0.01 * colony_id
    
    # Giới hạn pheromone
    tau_max = 5.0
    tau_min = 0.01
    
    # Khởi tạo pheromone
    pheromone = {}
    for i in range(N):
        for d in range(D):
            for s in range(5):  # 0-4 shifts
                pheromone[(i, d, s)] = tau_max
    
    best_solution = None
    best_quality = 0
    stagnation_count = 0
    
    for it in range(iterations):
        iteration_best_solution = None
        iteration_best_quality = 0
        
        # Mỗi kiến xây dựng một lời giải
        all_solutions = []
        for ant in range(n_ants):
            solution = improved_solution_construction(N, D, A, days_off, pheromone, alpha, beta)
            
            # Tối ưu cục bộ với xác suất
            if random.random() < 0.3:
                solution = local_optimization(solution, N, D, A, B, days_off)
            
            # Đánh giá lời giải
            quality = adaptive_evaluation(solution, N, D, A, B, days_off, it, iterations)
            all_solutions.append((solution, quality))
            
            # Cập nhật lời giải tốt nhất vòng lặp
            if quality > iteration_best_quality:
                iteration_best_solution = [row[:] for row in solution]
                iteration_best_quality = quality
                
                # Cập nhật lời giải tốt nhất toàn cầu
                if quality > best_quality:
                    best_solution = [row[:] for row in solution]
                    best_quality = quality
                    stagnation_count = 0
                else:
                    stagnation_count += 1
        
        # Bay hơi pheromone
        for key in pheromone:
            pheromone[key] = pheromone[key] * (1 - rho)
        
        # Chọn giải pháp để cập nhật pheromone
        if stagnation_count > 10:
            # Diversification: sử dụng lời giải tốt nhất vòng lặp
            update_solution = iteration_best_solution
            update_quality = iteration_best_quality
        else:
            # Exploitation: sử dụng lời giải tốt nhất toàn cầu
            update_solution = best_solution
            update_quality = best_quality
        
        # Cập nhật pheromone
        if update_solution:
            for i in range(N):
                for d in range(D):
                    s = update_solution[i][d]
                    pheromone[(i, d, s)] += update_quality
        
        # Giới hạn pheromone
        for key in pheromone:
            pheromone[key] = max(tau_min, min(tau_max, pheromone[key]))
        
        # Điều chỉnh giới hạn pheromone theo tiến trình
        if it % 10 == 0 and it > 0:
            ratio = (iterations - it) / iterations
            tau_max = 5.0 * ratio + 3.0 * (1 - ratio)
            tau_min = 0.01 * ratio + 0.05 * (1 - ratio)
    
    if result_queue is not None:
        result_queue.put((best_solution, best_quality, colony_id))
    return best_solution, best_quality

def multi_colony_aco(N, D, A, B, days_off, colonies=3, n_ants=10, iterations=50):
    """ACO với nhiều quần thể (Multi-colony ACO)"""
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    # Các tham số cho mỗi quần thể
    processes = []
    
    # Chạy song song các quần thể
    for c in range(colonies):
        # Chạy mỗi quần thể trên một process riêng
        p = mp.Process(target=max_min_ant_system, 
                      args=(c, N, D, A, B, days_off, n_ants, iterations, result_queue))
        processes.append(p)
        p.start()
    
    # Chờ tất cả các process hoàn thành
    for p in processes:
        p.join()
    
    # Thu thập kết quả
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Chọn lời giải tốt nhất
    results.sort(key=lambda x: x[1], reverse=True)
    best_solution = results[0][0]
    
    return best_solution

def main():
    # Đọc dữ liệu đầu vào
    try:
        N, D, A, B, days_off = read_input()
        
        # Thiết lập tham số dựa trên kích thước bài toán
        if N < 15:
            colonies = 2
            n_ants = 5
            iterations = 30
        elif N < 30:
            colonies = 3
            n_ants = 10
            iterations = 50
        else:
            colonies = 4
            n_ants = 15
            iterations = 80
        
        start_time = time.time()
        
        # Chạy thuật toán với nhiều quần thể
        solution = multi_colony_aco(N, D, A, B, days_off, 
                                   colonies=colonies, 
                                   n_ants=n_ants, 
                                   iterations=iterations)
        
        runtime = time.time() - start_time
        print(f"Thời gian chạy: {runtime:.2f} giây")
        
        # Kiểm tra và sửa chữa lời giải cuối cùng
        solution = local_optimization(solution, N, D, A, B, days_off)
        
        # In kết quả
        for i in range(N):
            print(" ".join(str(solution[i][d]) for d in range(D)))
        # Tính số ca đêm tối đa
        max_night_shifts = max(sum(1 for d in range(D) if solution[i][d] == 4) for i in range(N))
        print(f"Số ca đêm tối đa: {max_night_shifts}")
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()