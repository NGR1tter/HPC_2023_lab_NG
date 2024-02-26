import time
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np

@cuda.jit   # декоратор для работы функии на 

def VectorSum_on_gpu(a, b, c):
    for i in range(cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x, a.shape[0],cuda.blockDim.x * cuda.gridDim.x):
        if i < c.shape[0]:
            c[i] = a[i] + b[i]

def VectorSum_on_cpu(a, b):
    c = np.zeros((a.shape[0],), dtype=float)
    for index in range(a.size):
        c[index] = a[index] + b[index]
    return c


N = 10
threads_per_block = 512
blocks_per_grid = 1024
mas_size = np.linspace(100000, 10000000, N, dtype=int)
all_time_CPU = np.zeros((0, mas_size.shape[0]))
all_time_GPU = np.zeros((0, mas_size.shape[0]))

N1 = 4
for i in range(0, N1):#N
    print (i+1)
    mas_timeCPU = []
    mas_timeGPU = []

    for size in mas_size:
        print("Size: ",size)
        print("TPB: ", threads_per_block, "BPG: ", blocks_per_grid) 

        start = time.time()
        a = np.random.normal(0, 2.5, size)
        b = np.random.normal(0, 2.5, size)
        result_CPU = VectorSum_on_cpu(a, b)
        end = time.time()
        timeCPU = end - start

        #print("Результат на CPU:", result_CPU)
        print("time CPU:", timeCPU) 
        mas_timeCPU.append(timeCPU)

        start = time.time()
        a_on_gpu = cuda.to_device(a)
        b_on_gpu = cuda.to_device(b)
        c_on_gpu = cuda.to_device(np.zeros((size,), dtype=float))

        VectorSum_on_gpu[blocks_per_grid, threads_per_block](a_on_gpu, b_on_gpu, c_on_gpu)
        result_GPU = c_on_gpu.copy_to_host()

        end = time.time()
        timeGPU = end - start
        mas_timeGPU.append(timeGPU)

        #print("Результат на GPU:", result_GPU)
        print("time GPU:", timeGPU)  
        
        if np.array_equal(result_GPU,result_CPU):
            print('all nice. Result equal', '\n')
        else:
            print('we have problem')

    all_time_CPU = np.vstack((all_time_CPU, np.array(mas_timeCPU).reshape((1, N))))
    all_time_GPU = np.vstack((all_time_GPU, np.array(mas_timeGPU).reshape((1, N))))
    
all_time_CPU = np.squeeze(all_time_CPU)
all_time_GPU = np.squeeze(all_time_GPU)

#усреднение
#mas_CPU_time = [np.mean(all_time_CPU[:, i]) for i in range(all_time_CPU.shape[1])]
#mas_GPU_time = [np.mean(all_time_GPU[:, i]) for i in range(all_time_GPU.shape[1])]
#mas_CPU_time = np.array(mas_CPU_time)
#mas_GPU_time = np.array(mas_GPU_time)

mas_CPU_time=all_time_CPU
mas_GPU_time=all_time_GPU

plt.plot(mas_size, mas_CPU_time / mas_GPU_time, label='boost', color='black')
plt.title('График')
plt.xlabel('size')
plt.ylabel('boost')
plt.legend()
plt.show()