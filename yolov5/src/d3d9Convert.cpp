//
// Created by h6706 on 2023/9/18.
//
#include <cuda_d3d9_interop.h>
#include <cuda.h>
#include <iostream>
#include "det_dll_export.h"

cudaGraphicsResource *cudaResource;

//todo 我想直接把ID3D11Texture2D中的像素数据传递给gpu，避免两次拷贝过程，但是虽然实现录，性能却很不好
byte *gpuPointer;
IDirect3DSurface9 *rgbSurface;
cudaArray *cudaArrayPtr;
// 为结果分配CUDA内存

cudaStream_t stream2;

//没写完暂时不用了
byte *get_IDirect3DDevice9_pix(IDirect3DDevice9 *device, IDirect3DSurface9 *surface,int x,int y ,int clipWidth,int clipHeight){
    D3DSURFACE_DESC desc;
    surface->GetDesc(&desc);

    UINT width = clipWidth;
    UINT height = clipHeight;

    cudaError result;
    if (cudaResource == nullptr) {
        cudaStreamCreate(&stream2);

        int ret = device->CreateOffscreenPlainSurface(
                desc.Width, desc.Height,  // 设置可以修改宽高
                D3DFMT_R8G8B8, // 使用 rgb 格式
                D3DPOOL_DEFAULT,
                &rgbSurface,
                nullptr);



        result = cudaGraphicsD3D9RegisterResource(&cudaResource,
                                                  rgbSurface,
                                                   CU_GRAPHICS_REGISTER_FLAGS_NONE);
        if (result != cudaSuccess) {
            std::cout << "cudaGraphicsD3D11RegisterResource failed" << std::endl;
            return nullptr;
        }

        size_t pitch;
        result = cudaMallocPitch((void **) &gpuPointer, &pitch, width * sizeof(uint8_t) * 3, height);


    }
    return nullptr;
}