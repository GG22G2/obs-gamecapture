//
// Created by h6706 on 2023/9/19.
//

#include "similarImage.cu"

int main() {

    int width = 864;
    int height = 416;
    int byteSize = width * height * 4 * sizeof(unsigned char);
    unsigned char *dpix1;
    unsigned char *dpix2;
    unsigned int *dresult;

    unsigned char *hpix = (unsigned char *) malloc(byteSize);
    //cudaMalloc(&rgbBytes, width * height * 3);
    cudaError ret = cudaMalloc(&dpix1, byteSize);
    ret = cudaMalloc(&dpix2, byteSize);


    fillArrayWithRandom(hpix, byteSize);

    ret = cudaMemcpy(dpix1, hpix, byteSize, cudaMemcpyHostToDevice);
    ret = cudaMemcpy(dpix2, hpix, byteSize, cudaMemcpyHostToDevice);


    int gridNum = ceil(width * height / (float) ThreadX);
    unsigned int *hresult;
    ret = cudaMalloc(&dresult, gridNum * sizeof(int));

    cudaMallocHost(&hresult, gridNum * sizeof(int));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        //cudaMemset(dresult, 0, 4);
        //d_SharedMemoryTest<<<gridNum, ThreadX, 256, stream>>>((unsigned int *) dpix1, (unsigned int *) dpix2, dresult,width * height);

        //   imageCompare<<< gridNum / 2, ThreadX, 0, stream >>>((int4 *) dpix1, (int4 *) dpix2, dresult, width * height);


        cudaDeviceSynchronize();
        // ret = cudaMemcpy(hresult, dresult, gridNum * sizeof(int), cudaMemcpyDeviceToHost);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
                  << std::endl;
        std::cout << hresult[0] << " " << hresult[1] << " " << hresult[2] << std::endl;
    }


    return 0;
}