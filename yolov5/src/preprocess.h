#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>


#define RGB   1
#define RGBA  2
#define BGR   3
#define BGRA  4


void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_memory_src_preprocess(uint8_t* src, int src_width, int src_height,
                                float* dst, int dst_width, int dst_height,
                                cudaStream_t stream);

void cuda_gpu_src_preprocess(
        uint8_t *src, int src_width, int src_height,
        float *dst, int dst_width, int dst_height,
        cudaStream_t stream,int pixelFormat);
void cuda_batch_preprocess(std::vector<cv::Mat>& img_batch,
                           float* dst, int dst_width, int dst_height,
                           cudaStream_t stream);

