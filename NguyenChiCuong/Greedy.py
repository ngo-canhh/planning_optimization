import numpy as np
import time
class Heuristic_C:
    def __init__(self):
        pass
        
    def solve(self, N, D, A, B, dayoff):    
        start = time.time()
        x = np.zeros((N+1, D+1))

        # Tính số ca đêm tối thiểu trung bình mỗi nhân viên
        num_of_night_shifts = A * D
        min_night_shifts = 0
        if num_of_night_shifts % N != 0:
            min_night_shifts = num_of_night_shifts // N + 1
        else:
            min_night_shifts = num_of_night_shifts // N

        max_night_per_staff = np.zeros((N+1))
        #print("min_night_shifts: ", min_night_shifts)

        # Phân bổ ca đêm (ca 4)
        for j in range(1, D+1):
            # Đếm số nhân viên có thể làm việc trong ngày j
            available_staff = []
            for i in range(1, N+1):
                if dayoff[i][j] == 0 and x[i][j-1] != 4 and x[i][j] == 0:  
                    available_staff.append(i)
            

            # Kiểm tra tính khả thi
            if len(available_staff) < 4 * A:
                print(f"Không khả thi: Ngày {j} không đủ nhân viên ({len(available_staff)} < {4*A})")
                break

            # Phân bổ ca 4
            nums = 0
            available_staff.sort(key=lambda i: max_night_per_staff[i])
            for i in available_staff:
                if nums < A and max_night_per_staff[i] <= min_night_shifts:
                    x[i][j] = 4
                    nums += 1
                    max_night_per_staff[i] += 1
                if nums == A:
                    break

            # Nếu không đủ A nhân viên cho ca 4, thử phân bổ thêm
            if nums < A:
                remaining_staff = [i for i in available_staff if x[i][j] == 0]
                remaining_staff.sort(key=lambda i: max_night_per_staff[i])
                for i in remaining_staff:
                    if x[i][j] == 0 and nums < A:
                        x[i][j] = 4
                        nums += 1
                        max_night_per_staff[i] += 1
                    if nums == A:
                        break

            if nums < A:
                print(f"Vi phạm ràng buộc: Ngày {j} không đủ {A} nhân viên cho ca 4 (chỉ có {nums})")

        print("max_night_per_staff: ", np.max(max_night_per_staff[1:]))

        # Phân bổ ca ngày (ca 1, 2, 3)
        for j in range(1, D+1):
            # Lấy danh sách nhân viên có thể làm việc
            available_staff = []
            for i in range(1, N+1):
                if dayoff[i][j] == 0 and x[i][j] == 0 and x[i][j-1] != 4:  # Không nghỉ phép và chưa được gán ca 4
                    available_staff.append(i)

            # Phân bổ ca 1, 2, 3
            for shift in range(1, 4):
                nums = 0
                for i in available_staff:
                    if x[i][j] == 0 and nums < A:
                        x[i][j] = shift
                        nums += 1
                    if nums == A:
                        break


                if nums < A:
                    print(f"Vi phạm ràng buộc: Ngày {j}, ca {shift} không đủ {A} nhân viên (chỉ có {nums})")

        def check(x, N, D, A, B, dayoff):
            # Kiểm tra vi phạm ngày nghỉ
            for i in range(1, N + 1):
                for d in range(1, D + 1):
                    if dayoff[i][d] and x[i][d] != 0:
                        return False

            # Kiểm tra vi phạm số lượng nhân viên mỗi ca
            for d in range(1, D + 1):
                for shift in range(1, 5):
                    count = np.sum(x[:, d] == shift)
                    if count < A or count > B:
                        return False

            # Kiểm tra vi phạm ca đêm
            for i in range(1, N + 1):
                for d in range(1, D):
                    if x[i][d] == 4 and x[i][d + 1] != 0:
                        return False

            return True
        if check(x, N, D, A, B, dayoff):
            print("Solution found.")
            print("Lịch làm việc hợp lệ:")
            for i in range(1, N + 1):
                print(f"Nhân viên {i}: ", end='')
                for d in range(1, D + 1):
                    print(int(x[i][d]), end=' ')
                print()
            endtime = time.time()
            print("Execution time: {:.2f} seconds".format(endtime - start))
            return x
        else:
            print("Không tìm thấy lịch làm việc hợp lệ.")
            return None
        
