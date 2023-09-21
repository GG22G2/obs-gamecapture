#include <cuda_runtime_api.h>
#include "Main.h"

//#include <strsafe.h>
#include "GameCapture.h"


#include "graphics-hook-info.h"
#include "det_dll_export.h"
#include <thread>

volatile bool initialized = false;

static HANDLE init_hooks_thread = NULL;
struct graphics_offsets offsets32 = {0};
struct graphics_offsets offsets64 = {0};


extern "C" bool load_graphics_offsets(bool is32bit, bool use_hook_address_cache,
                                      const char *config_path);

using std::this_thread::sleep_for;

int main() {
    int initResult = detect_init(
            "G:\\kaifa_environment\\code\\clion\\csgo-util\\cmake-build-release\\yolov5\\bin\\csgo2.engine");
    std::locale::global(std::locale("en_US.UTF-8"));
    //detect_inferenceGpuData()

    struct game_capture *data = nullptr;
    while (data == nullptr) {
        data = (struct game_capture *) init_csgo_capture("Counter - Strike: Global Offensive - Direct3D 9", "Valve001");
        Sleep(100);
    }

    //先获取一帧
    while (true) {
        byte *data2 = game_capture_tick_cpu(data, 4, 0, 0, 1, 1);
        if (data2 != nullptr) {
            break;
        }
    }
    timeBeginPeriod(1); //todo window系统的休眠精度默认耗时是15-16之间，这个可以调整
    const bool testPureGpuCopy = false;
    const int testCount = 2000;
    std::cout << "截图成功\n";
    int captureWidth = 100;864;
    int captureHeight =100; 416;

    for (int i = 0; i < testCount; i++) {

       // byte *data2 = game_capture_tick_gpu(data, 4, 528, 332, captureWidth, captureHeight);
        auto start = std::chrono::system_clock::now();
        byte *data2 = game_capture_tick_gpu(data, 4, 0, 0, captureWidth, captureHeight);
        if (!data2) {
            continue;
        }
        auto end = std::chrono::system_clock::now();
        std::cout << testCount << "次纯gpu截图预测平均延迟: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 1 << "ms"
                  << std::endl;


        float *result = detect_inferenceGpuData(data2, captureWidth, captureHeight, 1);

       // std::cout << result[0] << "," << result[1] << "," << result[2] << "," << result[03] << "\n";
        //cudaFree(data2);
        if (result[4] < 0.6) {
            sleep_for(std::chrono::milliseconds(1));
        }

    }



    auto start1 = std::chrono::system_clock::now();
    for (int i = 0; i < 0; i++) {
        byte *data2 = game_capture_tick_cpu(data, 4, 528, 332, 864, 416);
        if (!data2) {
            continue;
        }

        int total = 864 * 416;
        uchar4 *char3 = (uchar4 *) data2;
        uchar3 *rgbPixls = (uchar3 *) malloc(total * sizeof(uchar3));
        for (int k = 0; k < total; ++k) {
            uchar4 p = char3[k];
            rgbPixls[k].x = p.x;
            rgbPixls[k].y = p.y;
            rgbPixls[k].z = p.z;
        }

        float *result = detect_inference((byte *) rgbPixls, 864, 416);
        // std::cout << result[0] << "," << result[1] << "," << result[2] << "," << result[03] << "\n";
        //cudaFree(data2);
        free(rgbPixls);
    }
    auto end1 = std::chrono::system_clock::now();
    std::cout << 10000 << "次纯cpu截图预测平均延迟: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1000.0 / 10000 << "ms"
              << std::endl;


    stop_game_capture(data);

    detect_release();

    //  std::cout << config.title;
    return 1;
}



