
#pragma once

#include <cuda_runtime.h>
#include <cstdint>



void calculateSumProxy(const uint8_t* imageData, int width, int height, unsigned long long* result,cudaStream_t stream);
void rgba2rgbProxy(const uint8_t *rgbaBytes, uint8_t *rgbBytes, int width, int height, cudaStream_t stream);
bool imageSimilar(unsigned char *dpix1, unsigned char *dpix2, int width, int height, cudaStream_t stream);
