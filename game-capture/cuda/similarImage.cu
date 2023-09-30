//
// Created by h6706 on 2023/9/18.
//


#include <iostream>
#include <cuda_runtime_api.h>



using namespace std;



// bgra格式像素
bool sumTotalPixel(unsigned char *dpix1, int width, int height) {

    unsigned char *hpix1;
    cudaMallocHost(&hpix1, width * height * sizeof(int));
    cudaMemcpy(hpix1, dpix1, width * height * sizeof(int), cudaMemcpyDeviceToHost);


    int sum = 0;
    for (int i = 0; i < width * height * 4; i++) {
        sum += hpix1[0] ;
    }
    std::cout << sum << endl;

    cudaFreeHost(hpix1);

    return false;
}


