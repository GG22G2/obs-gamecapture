#pragma once

#include <tchar.h>
#include <dshow.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h> //zhushi
#include <stdint.h>
#include "CommonTypes.h"
#include <chrono>
#include "Logging.h"
#include <dshow.h>
#include <strsafe.h>
#include <tchar.h>
#include <dxgi.h>
#include "graphics-hook-info.h"
#include "bmem.h"
#include "dstr.h"
#include "app-helpers.h"
#include "platform.h"
#include "threading.h"
#include "obfuscate.h"
#include "nt-stuff.h"
#include "inject-library.h"
#include "DibHelper.h"
#include "window-helpers.h"
#include "ipc-util/pipe.h"
#include "libyuv/convert.h"
#include "libyuv/scale.h"
#include "CommonTypes.h"
#include <dxgi.h>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <DXGI1_5.h>
#include <windows/ComPtr.hpp>
#include <d3dcompiler.h>


typedef struct gs_texture gs_texture_t;
typedef struct gs_texture_render gs_texrender_t;


static uint32_t inject_failed_count = 0;
typedef DPI_AWARENESS_CONTEXT(WINAPI* PFN_SetThreadDpiAwarenessContext)(
        DPI_AWARENESS_CONTEXT);
typedef DPI_AWARENESS_CONTEXT(WINAPI* PFN_GetThreadDpiAwarenessContext)(VOID);
typedef DPI_AWARENESS_CONTEXT(WINAPI* PFN_GetWindowDpiAwarenessContext)(HWND);

enum gs_color_format {
	GS_UNKNOWN,
	GS_A8,
	GS_R8,
	GS_RGBA,
	GS_BGRX,
	GS_BGRA,
	GS_R10G10B10A2,
	GS_RGBA16,
	GS_R16,
	GS_RGBA16F,
	GS_RGBA32F,
	GS_RG16F,
	GS_RG32F,
	GS_R16F,
	GS_R32F,
	GS_DXT1,
	GS_DXT3,
	GS_DXT5,
	GS_R8G8,
	GS_RGBA_UNORM,
	GS_BGRX_UNORM,
	GS_BGRA_UNORM,
	GS_RG16,
};
struct game_capture_config {
    char                          *title;
    char                         *klass;
    char                          *executable;
    enum window_priority          priority;
    enum capture_mode             mode;
    uint32_t                      scale_cx;
    uint32_t                      scale_cy;
    bool                          cursor;
    bool                          force_shmem;
    bool                          force_scaling;
    bool                          allow_transparency;
    bool                          limit_framerate;
    bool                          capture_overlays;
    bool                          anticheat_hook;
    HWND						  window;
    enum						  hook_rate hook_rate;
    bool is_10a2_2020pq;
};

struct game_capture {
    volatile int last_tex;
    uint64_t frame_interval;

    HANDLE injector_process;
    uint32_t cx;
    uint32_t cy;
    uint32_t pitch;
    DWORD process_id;
    DWORD thread_id;
    HWND next_window;
    HWND window;
    float retry_time;
    float fps_reset_time;
    float retry_interval;
    struct dstr title;
    struct dstr klass;
    struct dstr executable;
    enum window_priority priority;
    volatile long hotkey_window;
    volatile bool deactivate_hook;
    volatile bool activate_hook_now;
    bool wait_for_target_startup;
    bool showing;
    bool active;
    bool capturing;
    bool activate_hook;
    bool process_is_64bit;
    bool error_acquiring;
    bool dwm_capture;
    bool initial_config;
    bool convert_16bit;
    bool is_app;
    bool cursor_hidden;

    struct game_capture_config config;

    ipc_pipe_server_t pipe;
    gs_texture_t* texture;
    gs_texture_t* extra_texture;
    gs_texrender_t* extra_texrender;
    bool is_10a2_2020pq;
    bool linear_sample;
    struct hook_info* global_hook_info;
    HANDLE keepalive_mutex;
    HANDLE hook_init;
    HANDLE hook_restart;
    HANDLE hook_stop;
    HANDLE hook_ready;
    HANDLE hook_exit;
    HANDLE hook_data_map;
    HANDLE global_hook_info_map;
    HANDLE target_process;
    HANDLE texture_mutexes[2];
    wchar_t* app_sid;
    int retrying;
    float cursor_check_time;

    union {
        struct {
            struct shmem_data* shmem_data;
            uint8_t* texture_buffers[2];
        };

        struct shtex_data* shtex_data;
        void* data;
    };

    bool (*copy_texture)(struct game_capture*);

    PFN_SetThreadDpiAwarenessContext set_thread_dpi_awareness_context;
    PFN_GetThreadDpiAwarenessContext get_thread_dpi_awareness_context;
    PFN_GetWindowDpiAwarenessContext get_window_dpi_awareness_context;
};




bool isReady(void ** data);

extern "C" _declspec(dllexport) void* init_csgo_capture(const char* windowName, const char* windowClassName);
extern "C" _declspec(dllexport) byte* game_capture_tick_cpu(struct game_capture * data, float seconds, int x, int y, int width, int height);
extern "C" _declspec(dllexport) byte* game_capture_tick_gpu(struct game_capture * data, float seconds, int x, int y, int width, int height);
extern "C" _declspec(dllexport) bool stop_game_capture(void* data);

extern "C" _declspec(dllexport) void cudaFreeProxy(void* data);

void * init(LPCWSTR windowClassName, LPCWSTR windowName, game_capture_config *config, uint64_t frame_interval);

void set_fps(void **data, uint64_t frame_interval);
