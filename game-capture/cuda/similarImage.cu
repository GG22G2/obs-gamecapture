//
// Created by h6706 on 2023/9/18.
//

#include "test.h"
#include <iostream>
#include "cuda_utils.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>


using namespace std;

/**
 *
 * 864 * 416
 * 计算当前图片和上一张图片是否相似
 *
 * 执行的最小单元是线程束，线程束一般是16个线程，假设一个线程访问4字节，也就是一个线程束一次比较32个像素点
 *
 * 一次核函数加载32，比较了4个像素点 ， 那么一个线程束就是4*32个像素点,
 *
 * */
__global__ void imageCompare(const int4 *pix1, const int4 *pix2, unsigned int *sumResult, int length) {

    int position = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    if (position >= length) return;
    int4 v1 = pix1[position];
    int4 v2 = pix2[position];
    int sum = v1.x - v2.x;
    sum = sum + v1.y - v2.y;
    sum = sum + v1.z - v2.z;
    sum = sum + v1.w - v2.w;
    //unsigned int result = pix1[position] - pix2[position];
    atomicAdd(sumResult, sum);
}

const int NX = 10240;            //数组长度
const int ThreadX = 256;        //线程块大小
//使用shared memory和多个线程块
__global__ void
d_SharedMemoryTest(const unsigned int *pix1, const unsigned int *pix2, unsigned int *sumResult, int length) {
    __shared__ int hasNoEqual;
    int i = threadIdx.x;                                    //该线程块中线程索引
    int tid = blockIdx.x * 2 * blockDim.x + threadIdx.x;        //M个包含N个线程的线程块中相对应全局内存数组的索引（全局线程）
    __shared__ unsigned int s_Para[ThreadX];                        //定义固定长度（线程块长度）的共享内存数组
    if (tid < length) {
        int cal = pix1[tid] - pix2[tid];
        //s_Para[i] = cal;
        if (cal == 0) {
            s_Para[i] = cal;
        } else {
            hasNoEqual = 125;
            atomicAdd(sumResult, 0xFFFFFFFF);
        }
    }
    __syncthreads();                                        //(红色下波浪线提示由于VS不识别，不影响运行)同步，等待所有线程把自己负责的元素载入到共享内存再执行下面代码
    if (hasNoEqual == 125) {
        return;
    }

    for (int index = 1; index < blockDim.x; index *= 2)        //归约求和
    {
        __syncthreads();
        if (i % (2 * index) == 0) {
            s_Para[i] += s_Para[i + index];
        }
    }

    if (i == 0) {
        // sumResult[blockIdx.x] = s_Para[0];
        atomicAdd(sumResult, s_Para[0]);
    }
    //求和完成，总和保存在共享内存数组的0号元素中
    //在每个线程块中，将共享内存数组的0号元素赋给全局内存数组的对应元素，即线程块索引*线程块维度+i（blockIdx.x * blockDim.x + i）
}

unsigned int *dresult;
unsigned int *hresult;

// bgra格式像素
bool imageSimilar(unsigned char *dpix1, unsigned char *dpix2, int width, int height, cudaStream_t stream) {
    return false;
    int gridNum = ceil((width * height/2) / (float) ThreadX);

    if (dresult == nullptr){
        cudaMalloc(&dresult, 1 * sizeof(int));
        cudaMallocHost(&hresult, 1 * sizeof(int));
    }

//    unsigned char *hpix1;
//    unsigned char *hpix2;
//    cudaMallocHost(&hpix1, width * height * sizeof(int));
//    cudaMallocHost(&hpix2, width * height * sizeof(int));
//    cudaMemcpy(hpix1, dpix1, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(hpix2, dpix2, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//
//
//    int sum = 0;
//    for (int i = 0; i < width * height * 4; i++) {
//        sum += hpix1[0] - hpix2[0];
//    }
 //   std::cout << sum << endl;
    auto start = std::chrono::system_clock::now();
    cudaMemset(dresult, 0, 4);
    //todo  csgo2 中cpu资源好像已经利用很多了，这个方法延迟特别高，所以得换一种方式来处理
    d_SharedMemoryTest<<<gridNum, ThreadX, 256, stream>>>((unsigned int *) dpix1, (unsigned int *) dpix2, dresult,
                                                          width * height);
    cudaStreamSynchronize(stream);
    cudaMemcpy(hresult, dresult, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::system_clock::now();
    std::cout << "d_SharedMemoryTest time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
              << std::endl;
    bool result = hresult[0] == 0;
    //cudaFree(dresult);

   // cudaFreeHost(hresult);
    return result;
}


void fillArrayWithRandom(unsigned char *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 256;
    }
}

// bgra格式像素
bool imageSimilar234(unsigned char *dpix1, unsigned char *dpix2, int width, int height, cudaStream_t stream) {




    //  int gridNum = ceil((width * height/2) / (float) ThreadX);

    if (hresult == nullptr){
        //  cudaMalloc(&dresult, 4 * sizeof(char));
        cudaMallocHost(&hresult, 1 * sizeof(int));
    }

//    unsigned char *hpix1;
//    unsigned char *hpix2;
//    cudaMallocHost(&hpix1, width * height * sizeof(int));
//    cudaMallocHost(&hpix2, width * height * sizeof(int));
//    cudaMemcpy(hpix1, dpix1, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(hpix2, dpix2, width * height * sizeof(int), cudaMemcpyDeviceToHost);
//
//
//    int sum = 0;
//    for (int i = 0; i < width * height * 4; i++) {
//       // sum += hpix1[0] - hpix2[0];
//        sum += hpix1[0];
//    }
//    std::cout << sum << endl;
//    cudaFreeHost(hpix1);
//    cudaFreeHost(hpix2);

    auto start = std::chrono::system_clock::now();
    cudaMemcpy(hresult, dpix1, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    unsigned  char * t= (  unsigned  char *)hresult;
    auto end = std::chrono::system_clock::now();
//        std::cout << "valid time: "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
//              << std::endl;
    if (t[3]==205){        //旧图片
        return true;
    } else{
        return false;
    }


//    std::cout << hresult[0] << endl;
//    std::cout << hresult[1] << endl;
//    std::cout << hresult[2] << endl;
//    std::cout << hresult[3] << endl;

//    auto start = std::chrono::system_clock::now();
//    cudaMemset(dresult, 0, 4);
//    d_SharedMemoryTest<<<gridNum, ThreadX, 256, stream>>>((unsigned int *) dpix1, (unsigned int *) dpix2, dresult,
//                                                          width * height);
//    cudaStreamSynchronize(stream);
//    cudaMemcpy(hresult, dresult, 1 * sizeof(int), cudaMemcpyDeviceToHost);
//    auto end = std::chrono::system_clock::now();

//    bool result = hresult[0] == 0;

    // cudaMemset(dpix1+3, 254, 1);

    //  return false;
}
