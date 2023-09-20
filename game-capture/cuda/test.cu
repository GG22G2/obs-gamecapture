#include "test.h"
#include <stdio.h>
#include <stdint.h>

#include "cuda_utils.h"

//__global__ void hahahaha(
//        uint8_t* src) {
//}

static uint8_t *img_buffer_host = nullptr;
static uint8_t *img_buffer_device = nullptr;

struct AffineMatrix {
    float value[6];
};




__global__ void rgba2rgb(const uchar4 *rgbaBytes, uint8_t *rgbBytes, int width, int height, int length) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= length) return;
    uchar4 pix = rgbaBytes[position];
    int rgbIdx = position * 3;
    rgbBytes[rgbIdx] = pix.x;
    rgbBytes[rgbIdx + 1] = pix.y;
    rgbBytes[rgbIdx + 2] = pix.z;

}

__global__ void calculateSum(const uint8_t *imageData, int width, int height, unsigned long long *result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width * 3 + x * 3;

        unsigned long long sum = 0;
        sum += imageData[index];
        sum += imageData[index + 1];
        sum += imageData[index + 2];
        //sum += imageData[index + 3];
        atomicAdd(result, sum);
        //  result[0]=imageData[index];
    }
}


void rgba2rgbProxy(const uint8_t *rgbaBytes, uint8_t *rgbBytes, int width, int height, cudaStream_t stream) {
    int jobs = width * height;
    int threads = 256;
    int blocks = ceil(jobs / (float) threads);

    rgba2rgb <<<blocks, threads, 0, stream>>>((uchar4 *) rgbaBytes, rgbBytes, width, height, jobs);
}

void
calculateSumProxy(const uint8_t *imageData, int width, int height, unsigned long long *result, cudaStream_t stream) {
    // 计算核心配置
    dim3 blockSize(16, 16);  // 每个线程块的线程数量
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);  // 网格大小

    calculateSum <<<gridSize, blockSize, 0, stream>>>(imageData, width, height, result);
}