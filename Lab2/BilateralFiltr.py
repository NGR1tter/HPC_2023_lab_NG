import math
import time
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numba import cuda

@cuda.jit
def gauss_on_gpu(x):
    return 0.2 * math.exp(-1/10*(x ** 2))

def gauss_on_cpu(x):
    return 0.2 * math.exp(-1/10*(x ** 2))

@cuda.jit  # декоратор для работы функии на GPU
# y = up, down. x = right, left

def bilateral_filter_on_gpu(input, res):
    height, width = res.shape
    for i in range(cuda.grid(2)[0], height, cuda.blockDim.x * cuda.gridDim.x):
        for j in range(cuda.grid(2)[1], width, cuda.blockDim.y * cuda.gridDim.y):
            if i < height and j < width:
                pxl = input[i, j] / 255.0
                min_i = max(0, i - 4)
                max_i = min(height - 1, i + 4)
                min_j = max(0, j - 4)
                max_j = min(width - 1, j + 4)
                k = 0
                value = 0
                for x in range(min_i, max_i + 1):
                    for y in range(min_j, max_j + 1):
                        f = input[x, y] / 255.0
                        r = gauss_on_gpu(f - pxl) 
                        dist_x = (x - i) ** 2
                        dist_y = (y - j) ** 2
                        g = gauss_on_gpu(dist_x + dist_y)
                        value += f * r * g
                        k += g * r
                res[i, j] = 255.0 * value / k


def bilateral_filter_on_cpu(input, kernel=4):
    height, width = input.size
    res = np.zeros((width, height), dtype=np.uint8)
    print(input.size)
    for i in range(height):
        for j in range(width):
            pxl = input.getpixel((i, j)) / 255.0
            min_i = max(0, i - kernel)
            max_i = min(height - 1, i + kernel)
            min_j = max(0, j - kernel)
            max_j = min(width - 1, j + kernel)
            k = 0
            value = 0
            for x in range(min_i, max_i + 1):
                for y in range(min_j, max_j + 1):
                    f = input.getpixel((x, y)) / 255.0
                    r = gauss_on_cpu(f - pxl)
                    dist_x = (x - i) ** 2
                    dist_y = (y - j) ** 2
                    g = gauss_on_cpu(dist_x + dist_y) 
                    value += f * r * g
                    k += g * r
            res[j, i] = 255.0 * value / k
    return res


def start_calc(w_size: np.ndarray, h_size: np.ndarray):
    all_time_CPU = np.zeros((0, w_size.shape[0]))
    all_time_GPU = np.zeros((0, w_size.shape[0]))
    img = Image.open("C:/Users/alien_4.jpg").convert('L')

    for n in range(3):
        print(n, '\n')
        mas_timeCPU = []
        mas_timeGPU = []
        for w, h in zip(w_size, h_size):
            print("Size w and h: ",w, h)
            #print("TPB: ", threads_per_block, "BPG: ", blocks_per_grid) 
            start = time.time()
            new_img = img.resize((w, h))
            result_CPU = bilateral_filter_on_cpu(new_img)
            end = time.time()
            timeCPU = end - start
            
            #print("Результат на CPU:", result_CPU)
            print("time CPU:", timeCPU) 
            
            PIL_img = Image.fromarray(result_CPU, mode='L')
            PIL_img.save(f'C:/Project/HPS/REadyLABS/Nik_labs/Read/Lab2/Result_Bilinear/alien_4_CPU_{str(w)}x{str(h)}.jpg')
            mas_timeCPU.append(timeCPU)

            start = time.time()
            img_array = np.array(new_img)
            GPU_img_array = cuda.to_device(img_array)
            result_GPU = np.zeros((h, w), dtype=np.uint8)
            result_GPU = cuda.to_device(result_GPU)
            bilateral_filter_on_gpu[blocks_per_grid, threads_per_block](GPU_img_array, result_GPU)
            result_GPU = result_GPU.copy_to_host()
            end = time.time()
            timeGPU = end - start
            mas_timeGPU.append(timeGPU)

            #print("Результат на GPU:", result_GPU)
            print("time GPU:", timeGPU) 
        
            if np.array_equal(result_GPU,result_CPU):
                print('all nice. Result equal', '\n')
            else:
                print('we have problem')

            PIL_img = Image.fromarray(result_GPU, mode='L')
            PIL_img.save(f'C:/Project/HPS/REadyLABS/Nik_labs/Read/Lab2/Result_Bilinear/alien_4_GPU_{str(w)}x{str(h)}.jpg')

        all_time_CPU = np.vstack((all_time_CPU, np.array(mas_timeCPU).reshape((1, N))))
        all_time_GPU = np.vstack((all_time_GPU, np.array(mas_timeGPU).reshape((1, N))))
    
    #усреднение
    all_time_CPU = np.squeeze(all_time_CPU)
    all_time_GPU = np.squeeze(all_time_GPU)

    mas_CPU_time = [np.mean(all_time_CPU[:, i]) for i in range(all_time_CPU.shape[1])]
    mas_GPU_time = [np.mean(all_time_GPU[:, i]) for i in range(all_time_GPU.shape[1])]

    #mas_CPU_time=all_time_CPU
    #mas_GPU_time=all_time_GPU

    return mas_CPU_time, mas_GPU_time


threads_per_block = (8, 8)
blocks_per_grid = (16, 16)
img = Image.open("C:/Users/alien_4.jpg").convert('L')
print(img.size)

N = 10
w_size = np.linspace(100, img.size[0], N, dtype=int)
h_size = np.linspace(100, img.size[1], N, dtype=int)
mas_size = [str((i, j)) for i, j in zip(w_size, h_size)]

mas_CPU_time, mas_GPU_time = start_calc(w_size, h_size)

mas_CPU_time = np.array(mas_CPU_time)
mas_GPU_time = np.array(mas_GPU_time)

#print("time CPU:", mas_CPU_time) 
#print("time GPU:", mas_GPU_time) 
#print(mas_CPU_time / mas_GPU_time)

plt.plot(mas_size, mas_CPU_time / mas_GPU_time, label='boost', color='black')
plt.title('График')
plt.xlabel('size')
plt.ylabel('boost')
plt.xticks(np.linspace(0, 10, 10), mas_size)
plt.legend()
plt.show()