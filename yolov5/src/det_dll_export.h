//
// Created by h6706 on 2023/9/15.
//

#include <d3d9.h>

#ifndef YOLOV5_DET_DLL_EXPORT_H
#define YOLOV5_DET_DLL_EXPORT_H

#endif //YOLOV5_DET_DLL_EXPORT_H

extern "C"
{
#define API    _declspec(dllexport)
API
int detect_init(const char *engine_name);
API
float *detect_inference(unsigned char *data, int cols, int rows);

API
float *detect_inferenceGpuData(unsigned char *data, int cols, int rows,int type);

API
byte *get_IDirect3DDevice9_pix(IDirect3DDevice9 *device, IDirect3DSurface9 *surface);

API
void detect_release();
}