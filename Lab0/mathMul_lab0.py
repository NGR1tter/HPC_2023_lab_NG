import time
import matplotlib.pyplot as plt
import numpy as np
import numba
import multiprocessing
from numba import cuda

@cuda.jit  # декоратор для работы функии на 

def mathMul_on_gpu(a, b, c):
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        c[i, j] = 0
        for k in range(a.shape[1]):
            c[i, j] += a[i, k] * b[k, j]

def mathMul_on_cpu(a, b):
    result_matrix = np.matmul(a,b)
    return result_matrix

mas_timeCPU = []
mas_timeGPU = []
boost = []
mas_size = np.linspace(100, 2000, 20, dtype=int)
blocks_per_grid = (512, 512)
threads_per_block = (32, 32)

for size in mas_size:
    a = np.random.randint(-100, 100, size=(size, size))
    b = np.random.randint(-100, 100, size=(size, size))
    c = np.zeros((size, size))
    
    print(size,size)
    start = time.time()
    result_CPU = mathMul_on_cpu(a, b)
    end = time.time()

    timeCPU = end - start
    #print("Результат на CPU:", result_CPU)
    print("time CPU", timeCPU)
    mas_timeCPU.append(timeCPU)
    
    start = time.time()
    a_on_gpu = cuda.to_device(a)
    b_on_gpu = cuda.to_device(b)
    c_on_gpu = cuda.to_device(c)
    mathMul_on_gpu[blocks_per_grid, threads_per_block](a_on_gpu, b_on_gpu, c_on_gpu)
    result_GPU = c_on_gpu.copy_to_host()
    end = time.time()

    timeGPU = end - start
    mas_timeGPU.append(timeGPU)
    print("time GPU", timeGPU)
    #print("Результат на GPU:", result_GPU)
        
    if np.array_equal(result_GPU,result_CPU):
        print('all nice. Result equal')
    else:
        print('we have problem')

    boost_time = timeCPU/timeGPU
    boost.append(boost_time)

plt.plot(mas_size, mas_timeCPU, label='cpu', color='red')
plt.plot(mas_size, mas_timeGPU, label='gpu', color='blue')
plt.xlabel('Размерность матриц')
plt.ylabel('Время в сек')
plt.legend()
plt.grid()
plt.figure()
plt.title("График boost")
plt.plot(mas_size, boost, label = 'boost', color='black')
plt.xlabel("Размерность матриц")
plt.legend()
plt.grid()
plt.show()