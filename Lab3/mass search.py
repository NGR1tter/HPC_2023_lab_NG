import csv
import time
import numpy as np
from matplotlib import pyplot as plt
from numba import cuda

# R - матрица
# N - множество подстрок поиска
# Н - буфер поиска
def mass_search_on_cpu(R, N, H):
    for j in range(R.shape[1]):
        for i in range(R.shape[0]):
            n = N[i]
            for k in range(len(n)):
                if n[k] in H[j]:
                    R[i, j - k] -= 1
    return R

@cuda.jit
def mass_search_on_gpu(R, N, H):
    for j in range(cuda.grid(2)[1], R.shape[1], cuda.blockDim.y * cuda.gridDim.y):
        for i in range(cuda.grid(2)[0], R.shape[0], cuda.blockDim.x * cuda.gridDim.x):
            if i < R.shape[0] and j < R.shape[1]:
                n = N[i]
                for k in range(len(n)):
                    for p in range(len(H[j])):
                        if n[k] == H[j][p]:
                            R[i, j - k] -= 1


def save_to_csv(my_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(my_array)


def start_calculation(sizes: np.ndarray):
    all_time_CPU = np.zeros((0, sizes.shape[0]))
    all_time_GPU = np.zeros((0, sizes.shape[0]))

    for n in range(5):
        print(n)
        mas_timeCPU = []
        mas_timeGPU = []
        for size in sizes:
            print("size:", size)
            R = np.zeros((size, 3), dtype=int)  
            N = np.random.randint(len(ABC), size=(size, 3), dtype=np.uint8)
            H = np.random.randint(len(ABC), size=(size, 3), dtype=np.uint8)

            start = time.time()
            result_CPU = mass_search_on_cpu(R.copy(), N, H)
            end = time.time()
            timeCPU = end - start
            
            #print("Результат на CPU:", result_CPU)
            print("time CPU:", timeCPU) 
              
            mas_timeCPU.append(timeCPU)

            result_GPU = np.zeros((size, 3), dtype=int)

            start = time.time()
            result_GPU = cuda.to_device(result_GPU)
            N_GPU = cuda.to_device(N)
            H_GPU = cuda.to_device(H)
            mass_search_on_gpu[blocks_per_grid, threads_per_block](result_GPU, N_GPU, H_GPU)

            result_GPU = result_GPU.copy_to_host()
            end = time.time()
            timeGPU = end - start

            mas_timeGPU.append(timeGPU)
            save_to_csv(result_GPU, f"result_on_GPU_{size}")
            
            #print("Результат на GPU:", result_GPU)
            print("time GPU:", timeGPU) 

            if np.array_equal(result_GPU,result_CPU):
                print('all nice. Result equal', '\n')
            else:
                print('!!!we have problem!!!', '\n')

        all_time_CPU = np.vstack((all_time_CPU, np.array(mas_timeCPU).reshape((1, 10))))
        all_time_GPU = np.vstack((all_time_GPU, np.array(mas_timeGPU).reshape((1, 10))))
    
    all_time_CPU = np.squeeze(all_time_CPU)
    all_time_GPU = np.squeeze(all_time_GPU)
    
    #print(all_time_CPU)
    #print(all_time_GPU)

    mas_CPU_time = [np.mean(all_time_CPU[:, i]) for i in range(all_time_CPU.shape[1])]
    mas_GPU_time = [np.mean(all_time_GPU[:, i]) for i in range(all_time_GPU.shape[1])]
    return mas_CPU_time, mas_GPU_time


threads_per_block = (8, 8)
blocks_per_grid = (16, 16)
max = 10000
sizes = np.linspace(200, max, 10, dtype=int)
ABC = np.arange(256)

mas_CPU_time, mas_GPU_time = start_calculation(sizes)

mas_CPU_time = np.array(mas_CPU_time)
mas_GPU_time = np.array(mas_GPU_time)

#print(mas_CPU_time)
#print(mas_GPU_time)
#print(mas_CPU_time / mas_GPU_time)

plt.plot(sizes, mas_CPU_time / mas_GPU_time, label='boost', color='black')
plt.title('График')
plt.xlabel('size')
plt.ylabel('boost')
plt.xticks(np.linspace(200, max, 10), sizes)
plt.legend()
plt.show()