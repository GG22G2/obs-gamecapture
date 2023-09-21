#include "GameCapture.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cuda.h>

#include "test.h"

using namespace std;

#define STOP_BEING_BAD \
        "This is most likely due to security software" \
        "that the Bebo Capture installation folder is excluded/ignored in the " \
        "settings of the security software you are using."

#define DEFAULT_RETRY_INTERVAL 2.0f
#define ERROR_RETRY_INTERVAL 4.0f
enum hook_rate {
    HOOK_RATE_SLOW,
    HOOK_RATE_NORMAL,
    HOOK_RATE_FAST,
    HOOK_RATE_FASTEST
};

ComPtr<IDXGIFactory1> factory;
ComPtr<IDXGIAdapter1> adapter;
ComPtr<ID3D11Device> device;
ComPtr<ID3D11DeviceContext> context;
ComPtr<ID3D11Texture2D> texture;

extern "C" {
static std::vector<std::string> logged_file;


struct graphics_offsets offsets32 = {0};
struct graphics_offsets offsets64 = {0};
}

enum capture_mode {
    CAPTURE_MODE_ANY,
    CAPTURE_MODE_WINDOW,
    CAPTURE_MODE_HOTKEY
};


static inline int inject_library(HANDLE process, const wchar_t *dll) {
    return inject_library_obf(process, dll,
                              "D|hkqkW`kl{k\\osofj", 0xa178ef3655e5ade7,
                              "[uawaRzbhh{tIdkj~~", 0x561478dbd824387c,
                              "[fr}pboIe`dlN}", 0x395bfbc9833590fd,
                              "\\`zs}gmOzhhBq", 0x12897dd89168789a,
                              "GbfkDaezbp~X", 0x76aff7238788f7db);
}

static inline bool use_anticheat(struct game_capture *gc) {
    return gc->config.anticheat_hook && !gc->is_app;
}

static inline HANDLE open_mutex_plus_id(struct game_capture *gc,
                                        const wchar_t *name, DWORD id) {
    wchar_t new_name[64];
    _snwprintf(new_name, 64, L"%s%lu", name, id);
    return gc->is_app
           ? open_app_mutex(gc->app_sid, new_name)
           : open_mutex(new_name);
}

static inline HANDLE open_mutex_gc(struct game_capture *gc,
                                   const wchar_t *name) {
    return open_mutex_plus_id(gc, name, gc->process_id);
}

static inline HANDLE open_event_plus_id(struct game_capture *gc,
                                        const wchar_t *name, DWORD id) {
    wchar_t new_name[64];
    _snwprintf(new_name, 64, L"%s%lu", name, id);
    return gc->is_app
           ? open_app_event(gc->app_sid, new_name)
           : open_event(new_name);
}

static inline HANDLE open_event_gc(struct game_capture *gc,
                                   const wchar_t *name) {
    return open_event_plus_id(gc, name, gc->process_id);
}

static inline HANDLE open_map_plus_id(struct game_capture *gc,
                                      const wchar_t *name, DWORD id) {
    wchar_t new_name[64];
    _snwprintf(new_name, 64, L"%s%lu", name, id);

    return gc->is_app
           ? open_app_map(gc->app_sid, new_name)
           : OpenFileMappingW(GC_MAPPING_FLAGS, false, new_name);
}

static inline void free_config(struct game_capture_config *config) {
    bfree(config->title);
    bfree(config->klass);
    bfree(config->executable);
    memset(config, 0, sizeof(*config));
}

static inline HANDLE open_hook_info(struct game_capture *gc) {
    return open_map_plus_id(gc, SHMEM_HOOK_INFO, gc->process_id);
}

static inline float hook_rate_to_float(enum hook_rate rate) {
    switch (rate) {
        case HOOK_RATE_SLOW:
            return 2.0f;
        case HOOK_RATE_FAST:
            return 0.5f;
        case HOOK_RATE_FASTEST:
            return 0.1f;
        case HOOK_RATE_NORMAL:
            /* FALLTHROUGH */
        default:
            return 1.0f;
    }
}

static struct game_capture *game_capture_create(game_capture_config *config, uint64_t frame_interval) {
    struct game_capture *gc = (struct game_capture *) bzalloc(sizeof(*gc));
    gc->initial_config = true;
    gc->config.priority = config->priority;
    gc->retry_interval = DEFAULT_RETRY_INTERVAL *
                         hook_rate_to_float(gc->config.hook_rate);

    gc->config.mode = CAPTURE_MODE_WINDOW;// config->mode;
    gc->config.scale_cx = config->scale_cx;
    gc->config.scale_cy = config->scale_cy;
    gc->config.cursor = false; // config->cursor;
    gc->config.force_shmem = config->force_shmem;
    gc->config.force_scaling = config->force_scaling;
    gc->config.allow_transparency = config->allow_transparency;
    gc->config.limit_framerate = config->limit_framerate;
    gc->config.capture_overlays = config->capture_overlays;
    gc->config.anticheat_hook = inject_failed_count > 10 ? true : config->anticheat_hook;
    gc->frame_interval = frame_interval;
    gc->last_tex = -1;
    gc->retry_time = 1;
    gc->initial_config = true;
    gc->priority = config->priority;
    gc->wait_for_target_startup = false;
    gc->window = config->window;

    return gc;
}

static inline enum gs_color_format convert_format(uint32_t format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return GS_RGBA;
        case DXGI_FORMAT_B8G8R8X8_UNORM:
            return GS_BGRX;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            return GS_BGRA;
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return GS_R10G10B10A2;
        case DXGI_FORMAT_R16G16B16A16_UNORM:
            return GS_RGBA16;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return GS_RGBA16F;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return GS_RGBA32F;
    }

    return GS_UNKNOWN;
}

static void close_handle(HANDLE *p_handle) {
    HANDLE handle = *p_handle;
    if (handle) {
        if (handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
        *p_handle = NULL;
    }
}

static inline HMODULE kernel32(void) {
    static HMODULE kernel32_handle = NULL;
    if (!kernel32_handle)
        kernel32_handle = GetModuleHandleW(L"kernel32");
    return kernel32_handle;
}

static inline HANDLE open_process(DWORD desired_access, bool inherit_handle,
                                  DWORD process_id) {
    static HANDLE (WINAPI *open_process_proc)(DWORD, BOOL, DWORD) = NULL;
    if (!open_process_proc)
        open_process_proc = (HANDLE(__stdcall *)(DWORD, BOOL, DWORD)) get_obfuscated_func(kernel32(),
                                                                                          "NuagUykjcxr",
                                                                                          0x1B694B59451ULL);

    return open_process_proc(desired_access, inherit_handle, process_id);
}

static void setup_window(struct game_capture *gc, HWND window) {
    HANDLE hook_restart;
    HANDLE process;

    GetWindowThreadProcessId(window, &gc->process_id);
    if (gc->process_id) {
        process = open_process(PROCESS_QUERY_INFORMATION,
                               false, gc->process_id);
        if (process) {
            gc->is_app = is_app(process);
            if (gc->is_app) {
                gc->app_sid = get_app_sid(process);
            }
            CloseHandle(process);
        }
    }

    /* do not wait if we're re-hooking a process */
    hook_restart = open_event_gc(gc, EVENT_CAPTURE_RESTART);
    if (hook_restart) {
        gc->wait_for_target_startup = false;
        CloseHandle(hook_restart);
    }

    /* otherwise if it's an unhooked process, always wait a bit for the
     * target process to start up before starting the hook process;
     * sometimes they have important modules to load first or other hooks
     * (such as steam) need a little bit of time to load.  ultimately this
     * helps prevent crashes */
    if (gc->wait_for_target_startup) {
        gc->retry_interval = 3.0f;
        gc->wait_for_target_startup = false;
    } else {
        gc->next_window = window;
    }
}

static void get_fullscreen_window(struct game_capture *gc) {
    HWND window = GetForegroundWindow();
    MONITORINFO mi = {0};
    HMONITOR monitor;
    DWORD styles;
    RECT rect;

    gc->next_window = NULL;

    if (!window) {
        return;
    }
    if (!GetWindowRect(window, &rect)) {
        return;
    }

    /* ignore regular maximized windows */
    styles = (DWORD) GetWindowLongPtr(window, GWL_STYLE);
    if ((styles & WS_MAXIMIZE) != 0 && (styles & WS_BORDER) != 0) {
        return;
    }

    monitor = MonitorFromRect(&rect, MONITOR_DEFAULTTONEAREST);
    if (!monitor) {
        return;
    }

    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfo(monitor, &mi)) {
        return;
    }

    if (rect.left == mi.rcMonitor.left &&
        rect.right == mi.rcMonitor.right &&
        rect.bottom == mi.rcMonitor.bottom &&
        rect.top == mi.rcMonitor.top) {
        setup_window(gc, window);
    } else {
        gc->wait_for_target_startup = true;
    }
}

static void get_selected_window(struct game_capture *gc) {
    HWND window;

    if (dstr_cmpi(&gc->klass, "dwm") == 0) {
        wchar_t class_w[512];
        os_utf8_to_wcs(gc->klass.array, 0, class_w, 512);
        window = FindWindowW(class_w, NULL);
    } else {
        window = find_window(INCLUDE_MINIMIZED,
                             gc->priority,
                             gc->klass.array,
                             gc->title.array,
                             gc->executable.array);
    }

    if (window) {
        setup_window(gc, window);
    } else {
        gc->wait_for_target_startup = true;
    }
}

static inline bool hook_direct(struct game_capture *gc,
                               const char *hook_path_rel) {
    wchar_t hook_path_abs_w[MAX_PATH];
    wchar_t *hook_path_rel_w;
    wchar_t *path_ret;
    HANDLE process;
    int ret;

    os_utf8_to_wcs_ptr(hook_path_rel, 0, &hook_path_rel_w);
    if (!hook_path_rel_w) {
        warn("hook_direct: could not convert string");
        return false;
    }

    path_ret = _wfullpath(hook_path_abs_w, hook_path_rel_w, MAX_PATH);
    bfree(hook_path_rel_w);

    if (path_ret == NULL) {
        warn("hook_direct: could not make absolute path");
        return false;
    }

    process = open_process(PROCESS_ALL_ACCESS, false, gc->process_id);
    if (!process) {
        warn("hook_direct: could not open process: %S (%lu) %S, %S",
             gc->config.executable, GetLastError(), gc->config.title, gc->config.klass);
        return false;
    }

    ret = inject_library(process, hook_path_abs_w);
    CloseHandle(process);

    if (ret != 0) {
        error("hook_direct: inject failed: %d, anti_cheat: %d, %S, %S, %S", ret, gc->config.anticheat_hook,
              gc->config.title, gc->config.klass, gc->config.executable);
        if (ret == INJECT_ERROR_UNLIKELY_FAIL) {
            inject_failed_count++;
        }
        return false;
    }

    return true;
}

static const char *blacklisted_exes[] = {
        "explorer",
        "steam",
        "battle.net",
        "galaxyclient",
        "skype",
        "uplay",
        "origin",
        "devenv",
        "taskmgr",
        "chrome",
        "firefox",
        "systemsettings",
        "applicationframehost",
        "cmd",
        "bebo",
        "epicgameslauncher",
        "shellexperiencehost",
        "winstore.app",
        "searchui",
        NULL
};

static bool is_blacklisted_exe(const char *exe) {
    char cur_exe[MAX_PATH];

    if (!exe)
        return false;

    for (const char **vals = blacklisted_exes; *vals; vals++) {
        strcpy(cur_exe, *vals);
        strcat(cur_exe, ".exe");

        if (_strcmpi(cur_exe, exe) == 0)
            return true;
    }

    return false;
}

static bool target_suspended(struct game_capture *gc) {
    return thread_is_suspended(gc->process_id, gc->thread_id);
}

static inline bool is_64bit_windows(void) {
#ifdef _WIN64
    return true;
#else
    BOOL x86 = false;
    bool success = !!IsWow64Process(GetCurrentProcess(), &x86);
    return success && !!x86;
#endif
}

static inline bool is_64bit_process(HANDLE process) {
    BOOL x86 = true;
    if (is_64bit_windows()) {
        bool success = !!IsWow64Process(process, &x86);
        if (!success) {
            return false;
        }
    }

    return !x86;
}

static inline bool open_target_process(struct game_capture *gc) {
    gc->target_process = open_process(
            PROCESS_QUERY_INFORMATION | SYNCHRONIZE,
            false, gc->process_id);
    if (!gc->target_process) {
        warn("could not open process: %S (%lu) %S, %S",
             gc->config.executable, GetLastError(), gc->config.title, gc->config.klass);
        return false;
    }

    gc->process_is_64bit = is_64bit_process(gc->target_process);
    gc->is_app = is_app(gc->target_process);
    if (gc->is_app) {
        gc->app_sid = get_app_sid(gc->target_process);
    }
    return true;
}

static bool check_file_integrity(struct game_capture *gc, const char *file,
                                 const char *name) {
    DWORD error;
    HANDLE handle;
    wchar_t *w_file = NULL;

    if (!file || !*file) {
        warn("Game capture %S not found.", STOP_BEING_BAD, name);
        return false;
    }

    if (!os_utf8_to_wcs_ptr(file, 0, &w_file)) {
        warn("Could not convert file name to wide string");
        return false;
    }

    handle = CreateFileW(w_file, GENERIC_READ | GENERIC_EXECUTE,
                         FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);

    bfree(w_file);

    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
        return true;
    }

    error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND) {
        warn("Game capture file '%S' not found."
                     STOP_BEING_BAD, file);
    } else if (error == ERROR_ACCESS_DENIED) {
        warn("Game capture file '%S' could not be loaded."
                     STOP_BEING_BAD, file);
    } else {
        warn("Game capture file '%S' could not be loaded: %lu."
                     STOP_BEING_BAD, file, error);
    }

    return false;
}

static inline bool create_inject_process(struct game_capture *gc,
                                         const char *inject_path, const char *hook_dll) {
    wchar_t *command_line_w = (wchar_t *) malloc(4096 * sizeof(wchar_t));
    wchar_t *inject_path_w;
    wchar_t *hook_dll_w;
    bool anti_cheat = use_anticheat(gc);
    PROCESS_INFORMATION pi = {0};
    STARTUPINFO si = {0};
    bool success = false;

    os_utf8_to_wcs_ptr(inject_path, 0, &inject_path_w);
    os_utf8_to_wcs_ptr(hook_dll, 0, &hook_dll_w);

    si.cb = sizeof(si);

    swprintf(command_line_w, 4096, L"\"%s\" \"%s\" %lu %lu",
             inject_path_w, hook_dll_w,
             (unsigned long) anti_cheat,
             anti_cheat ? gc->thread_id : gc->process_id);

    success = !!CreateProcessW(inject_path_w, command_line_w, NULL, NULL,
                               false, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    if (success) {
        CloseHandle(pi.hThread);
        gc->injector_process = pi.hProcess;
    } else {
        warn("Failed to create inject helper process: %S (%lu)",
             gc->config.executable, GetLastError());
    }

    free(command_line_w);
    bfree(inject_path_w);
    bfree(hook_dll_w);
    return success;
}

char *concat_str(const char *parent, const char *file_c) {
    char *result = (char *) bmalloc(strlen(parent) + strlen(file_c) + 1);
    sprintf(result, "%s\\%s", parent, file_c);
    return result;
}

static inline bool inject_hook(struct game_capture *gc) {
    bool matching_architecture;
    bool success = false;
    const char *hook_dll;
    char *inject_path;
    char *hook_path;

    char *parent = "D:\\tools\\obs-studio\\data\\obs-plugins\\win-capture\\";
    if (gc->process_is_64bit) {
        hook_dll = "graphics-hook64.dll";
        inject_path = concat_str(parent,
                                 "inject-helper64.exe"); //inject-helper64.exe    bebo_find_file("inject-helper64.exe");
    } else {
        hook_dll = "graphics-hook32.dll";
        inject_path = concat_str(parent,
                                 "inject-helper32.exe");//inject-helper32.exe   bebo_find_file("inject-helper32.exe");
    }
    //todo 还原路径hook_dll对应路径
    hook_path = concat_str(parent, hook_dll); //hook_dll    //  bebo_find_file(hook_dll);

    info("injecting %S with %S into %S", hook_dll, inject_path, gc->config.executable);

    if (!check_file_integrity(gc, inject_path, "inject helper")) {
        goto cleanup;
    }
    if (!check_file_integrity(gc, hook_path, "graphics hook")) {
        goto cleanup;
    }

#ifdef _WIN64
    matching_architecture = gc->process_is_64bit;
#else
    matching_architecture = !gc->process_is_64bit;
#endif

    if (matching_architecture && !use_anticheat(gc)) {
        info("using direct hook");
        success = hook_direct(gc, hook_path);

        if (!success && inject_failed_count > 10) {
            gc->config.anticheat_hook = true;
            info("hook_direct: inject failed for 10th time, retrying with helper (%S hook)", use_anticheat(gc) ?
                                                                                             "compatibility"
                                                                                                               : "direct");
            success = create_inject_process(gc, inject_path, hook_dll);
        }
    } else {
        info("using helper (%S hook)", use_anticheat(gc) ?
                                       "compatibility" : "direct");
        success = create_inject_process(gc, inject_path, hook_dll);
    }

    if (success) {
        inject_failed_count = 0;
    }

    cleanup:
    bfree(inject_path);
    bfree(hook_path);
    return success;
}

static inline bool init_keepalive(struct game_capture *gc) {
    wchar_t new_name[64];
    _snwprintf(new_name, 64, L"%ls%lu", WINDOW_HOOK_KEEPALIVE,
               gc->process_id);

    gc->keepalive_mutex = CreateMutexW(NULL, false, new_name);
    if (!gc->keepalive_mutex) {
        warn("Failed to create keepalive mutex: %lu", GetLastError());
        return false;
    }
    return true;
}

static inline bool init_texture_mutexes(struct game_capture *gc) {
    gc->texture_mutexes[0] = open_mutex_gc(gc, MUTEX_TEXTURE1);
    gc->texture_mutexes[1] = open_mutex_gc(gc, MUTEX_TEXTURE2);

    if (!gc->texture_mutexes[0] || !gc->texture_mutexes[1]) {
        DWORD error = GetLastError();
        if (error == 2) {
            if (!gc->retrying) {
                gc->retrying = 2;
                info("hook not loaded yet, retrying..");
            }
        } else {
            warn("failed to open texture mutexes: %lu",
                 GetLastError());
        }
        return false;
    }

    return true;
}

static void pipe_log(void *param, uint8_t *data, size_t size) {
    struct game_capture *gc = (game_capture *) param;
    if (data && size)
        info("%S", data);
}

static inline bool init_pipe(struct game_capture *gc) {
    char name[64];

    sprintf(name, "%s%lu", PIPE_NAME, gc->process_id);

    if (!ipc_pipe_server_start(&gc->pipe, name, pipe_log, gc)) {
        warn("init_pipe: failed to start pipe");
        return false;
    }

    return true;
}

static inline void reset_frame_interval(struct game_capture *gc) {
    gc->global_hook_info->frame_interval = gc->frame_interval;
}

static inline bool init_hook_info(struct game_capture *gc) {
    gc->global_hook_info_map = open_hook_info(gc);
    if (!gc->global_hook_info_map) {
        warn("init_hook_info: get_hook_info failed: %lu",
             GetLastError());
        return false;
    }

    gc->global_hook_info = (hook_info *) MapViewOfFile(gc->global_hook_info_map,
                                                       FILE_MAP_ALL_ACCESS, 0, 0,
                                                       sizeof(*gc->global_hook_info));
    if (!gc->global_hook_info) {
        warn("init_hook_info: failed to map data view: %lu",
             GetLastError());
        return false;
    }

    gc->global_hook_info->offsets = gc->process_is_64bit ? offsets64 : offsets32;
    gc->global_hook_info->capture_overlay = gc->config.capture_overlays;
    gc->global_hook_info->force_shmem = false;
    gc->global_hook_info->UNUSED_use_scale = false;
    gc->global_hook_info->allow_srgb_alias = true;
    gc->global_hook_info->cx = gc->config.scale_cx;
    gc->global_hook_info->cy = gc->config.scale_cy;
    reset_frame_interval(gc);

    return true;
}

static inline bool init_events(struct game_capture *gc) {
    if (!gc->hook_restart) {
        gc->hook_restart = open_event_gc(gc, EVENT_CAPTURE_RESTART);
        if (!gc->hook_restart) {
            warn("init_events: failed to get hook_restart "
                 "event: %lu", GetLastError());
            return false;
        }
    }

    if (!gc->hook_stop) {
        gc->hook_stop = open_event_gc(gc, EVENT_CAPTURE_STOP);
        if (!gc->hook_stop) {
            warn("init_events: failed to get hook_stop event: %lu",
                 GetLastError());
            return false;
        }
    }

    if (!gc->hook_init) {
        gc->hook_init = open_event_gc(gc, EVENT_HOOK_INIT);
        if (!gc->hook_init) {
            warn("init_events: failed to get hook_init event: %lu",
                 GetLastError());
            return false;
        }
    }

    if (!gc->hook_ready) {
        gc->hook_ready = open_event_gc(gc, EVENT_HOOK_READY);
        if (!gc->hook_ready) {
            warn("init_events: failed to get hook_ready event: %lu",
                 GetLastError());
            return false;
        }
    }

    if (!gc->hook_exit) {
        gc->hook_exit = open_event_gc(gc, EVENT_HOOK_EXIT);
        if (!gc->hook_exit) {
            warn("init_events: failed to get hook_exit event: %lu",
                 GetLastError());
            return false;
        }
    }

    return true;
}

/* if there's already a hook in the process, then signal and start */
static inline bool attempt_existing_hook(struct game_capture *gc) {
    gc->hook_restart = open_event_gc(gc, EVENT_CAPTURE_RESTART);
    if (gc->hook_restart) {
        char *info = gc->config.executable;
        std::cout << info << "\n";
        debug("existing hook found, signaling process: %s",
              gc->config.executable);
        SetEvent(gc->hook_restart);
        return true;
    }

    return false;
}

static bool init_hook(struct game_capture *gc) {
    struct dstr exe = {0};
    bool blacklisted_process = false;

    if (gc->config.mode == CAPTURE_MODE_ANY) {
        if (get_window_exe(&exe, gc->next_window)) {
            info("attempting to hook fullscreen process: %S", exe.array);
        }
    } else {
        info("attempting to hook process: %S", gc->executable.array);
        dstr_copy_dstr(&exe, &gc->executable);
    }

    blacklisted_process = is_blacklisted_exe(exe.array);
    if (blacklisted_process)
        info("cannot capture %S due to being blacklisted", exe.array);
    dstr_free(&exe);

    if (blacklisted_process) {
        return false;
    }
    if (target_suspended(gc)) {
        info("target is suspended");
        return false;
    }
    if (!open_target_process(gc)) {
        return false;
    }
    if (!init_keepalive(gc)) {
        return false;
    }
    if (!init_pipe(gc)) {
        return false;
    }
    if (!attempt_existing_hook(gc)) {
        if (!inject_hook(gc)) {
            return false;
        }
    }
    if (!init_texture_mutexes(gc)) {
        return false;
    }

    if (!init_hook_info(gc)) {
        return false;
    }

    if (!init_events(gc)) {
        return false;
    }

    SetEvent(gc->hook_init);

    gc->window = gc->next_window;
    gc->next_window = NULL;
    gc->active = true;
    gc->retrying = 0;
    return true;
}

static void stop_capture(struct game_capture *gc) {
    ipc_pipe_server_free(&gc->pipe);

    if (gc->hook_stop) {
        SetEvent(gc->hook_stop);
    }

    if (gc->global_hook_info) {
        UnmapViewOfFile(gc->global_hook_info);
        gc->global_hook_info = NULL;
    }

    if (gc->data) {
        UnmapViewOfFile(gc->data);
        gc->data = NULL;
    }

    if (gc->app_sid) {
        LocalFree(gc->app_sid);
        gc->app_sid = NULL;
    }

    close_handle(&gc->hook_restart);
    close_handle(&gc->hook_stop);
    close_handle(&gc->hook_ready);
    close_handle(&gc->hook_exit);
    close_handle(&gc->hook_init);
    close_handle(&gc->hook_data_map);
    close_handle(&gc->keepalive_mutex);
    close_handle(&gc->global_hook_info_map);
    close_handle(&gc->target_process);
    close_handle(&gc->texture_mutexes[0]);
    close_handle(&gc->texture_mutexes[1]);

    if (gc->active)
        info("game capture stopped");

    gc->copy_texture = NULL;
    gc->wait_for_target_startup = false;
    gc->active = false;
    gc->capturing = false;

    if (gc->retrying)
        gc->retrying--;
}

HWND dbg_last_window = NULL;

static void try_hook(struct game_capture *gc) {
    if (0 && gc->config.mode == CAPTURE_MODE_ANY) {
        get_fullscreen_window(gc);
    } else {
        get_selected_window(gc);
    }

    if (gc->next_window) {
        if (gc->next_window != dbg_last_window) {
            info("hooking next window: %X, %S, %S", gc->next_window, gc->config.title, gc->config.klass);
            dbg_last_window = gc->next_window;
        }
        gc->thread_id = GetWindowThreadProcessId(gc->next_window, &gc->process_id);

        // Make sure we never try to hook ourselves (projector)
        if (gc->process_id == GetCurrentProcessId())
            return;

        if (!gc->thread_id && gc->process_id)
            return;

        if (!gc->process_id) {
            warn("error acquiring, failed to get window thread/process ids: %lu",
                 GetLastError());
            gc->error_acquiring = true;
            return;
        }

        if (!init_hook(gc)) {
            stop_capture(gc);
        }
    } else {
        gc->active = false;
    }
}

bool isReady(void **data) {
    if (*data == NULL) {
        return false;
    }
    struct game_capture *gc = (game_capture *) *data;
    //	debug("isReady - data active: %d && retrying %d - %d", gc->active, gc->retrying, gc->active && gc->capturing);
    return gc->active && !gc->retrying;
}

void set_fps(void **data, uint64_t frame_interval) {
    struct game_capture *gc = (game_capture *) *data;

    if (gc == NULL) {
        debug("set_fps: gc==NULL");
        return;
    }
    debug("set_fps: %d", frame_interval);
    gc->global_hook_info->frame_interval = frame_interval;
}

void *init(LPCWSTR windowClassName, LPCWSTR windowName, game_capture_config *config, uint64_t frame_interval) {
    struct game_capture *gc = (game_capture *) bzalloc(sizeof(*gc));

    HWND hwnd = NULL;
    window_priority priority = WINDOW_PRIORITY_EXE;

    if (windowClassName != NULL && lstrlenW(windowClassName) > 0 &&
        windowName != NULL && lstrlenW(windowName) > 0) {
        hwnd = FindWindowW(windowClassName, windowName);
    }

    if (hwnd == NULL &&
        windowClassName != NULL && lstrlenW(windowClassName) > 0) {
        hwnd = FindWindowW(windowClassName, NULL);
        priority = WINDOW_PRIORITY_CLASS;
    }

    if (hwnd == NULL &&
        windowName != NULL && lstrlenW(windowName) > 0) {
        hwnd = FindWindowW(NULL, windowName);
        priority = WINDOW_PRIORITY_TITLE;
    }

    if (hwnd == NULL) {
        return NULL;
    }

    config->window = hwnd;

    gc = game_capture_create(config, frame_interval);

    struct dstr *klass = &gc->klass;
    struct dstr *title = &gc->title;
    struct dstr *exe = &gc->executable;
    get_window_class(klass, hwnd);
    get_window_exe(exe, hwnd);
    get_window_title(title, hwnd);

    gc->config.executable = _strdup(exe->array);
    gc->config.title = _strdup(title->array);
    gc->config.klass = _strdup(klass->array);;

    gc->priority = priority;

    const HMODULE hModuleUser32 = GetModuleHandle(L"User32.dll");
    if (hModuleUser32) {
        PFN_SetThreadDpiAwarenessContext
                set_thread_dpi_awareness_context =
                (PFN_SetThreadDpiAwarenessContext) GetProcAddress(
                        hModuleUser32,
                        "SetThreadDpiAwarenessContext");
        PFN_GetThreadDpiAwarenessContext
                get_thread_dpi_awareness_context =
                (PFN_GetThreadDpiAwarenessContext) GetProcAddress(
                        hModuleUser32,
                        "GetThreadDpiAwarenessContext");
        PFN_GetWindowDpiAwarenessContext
                get_window_dpi_awareness_context =
                (PFN_GetWindowDpiAwarenessContext) GetProcAddress(
                        hModuleUser32,
                        "GetWindowDpiAwarenessContext");
        if (set_thread_dpi_awareness_context &&
            get_thread_dpi_awareness_context &&
            get_window_dpi_awareness_context) {
            gc->set_thread_dpi_awareness_context =
                    set_thread_dpi_awareness_context;
            gc->get_thread_dpi_awareness_context =
                    get_thread_dpi_awareness_context;
            gc->get_window_dpi_awareness_context =
                    get_window_dpi_awareness_context;
        }
    }
    return gc;
}

enum capture_result {
    CAPTURE_FAIL,
    CAPTURE_RETRY,
    CAPTURE_SUCCESS
};

static inline bool init_data_map(struct game_capture *gc, HWND window) {
    //wchar_t new_name[64];
    //_snwprintf(new_name, 64, L"%s%lu", name, id);

    wchar_t name[64];
    swprintf(name, 64, SHMEM_TEXTURE L"_%" "llu" "_",
             (uint64_t) (uintptr_t) window);

    gc->hook_data_map =
            open_map_plus_id(gc, name, gc->global_hook_info->map_id);
    return !!gc->hook_data_map;
}

const static D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
};

static inline enum capture_result init_capture_data(struct game_capture *gc) {
    gc->cx = gc->global_hook_info->cx;
    gc->cy = gc->global_hook_info->cy;
    gc->pitch = gc->global_hook_info->pitch;

    if (gc->data) {
        UnmapViewOfFile(gc->data);
        gc->data = NULL;
    }

    CloseHandle(gc->hook_data_map);

    DWORD error = 0;
    if (!init_data_map(gc, gc->window)) {
        HWND retry_hwnd = (HWND) (uintptr_t) gc->global_hook_info->window;
        error = GetLastError();

        /* if there's an error, just override.  some windows don't play
         * nice. */
        if (init_data_map(gc, retry_hwnd)) {
            error = 0;
        }
    }

    if (!gc->hook_data_map) {
        if (error == 2) {
            return CAPTURE_RETRY;
        } else {
            warn("init_capture_data: failed to open file "
                 "mapping: %lu",
                 error);
        }
        return CAPTURE_FAIL;
    }
    gc->data = MapViewOfFile(gc->hook_data_map, FILE_MAP_ALL_ACCESS, 0, 0,
                             gc->global_hook_info->map_size);
    if (!gc->data) {
        warn("init_capture_data: failed to map data view: %lu",
             GetLastError());
        return CAPTURE_FAIL;
    }

    return CAPTURE_SUCCESS;

    info("init_capture_data successful for %S, %S, %S", gc->config.title, gc->config.klass, gc->config.executable);
    return CAPTURE_SUCCESS;
}

static inline bool is_16bit_format(uint32_t format) {
    return format == DXGI_FORMAT_B5G5R5A1_UNORM ||
           format == DXGI_FORMAT_B5G6R5_UNORM;
}

static int fef = 0;
cudaGraphicsResource *cudaResource;

//todo 我想直接把ID3D11Texture2D中的像素数据传递给gpu，避免两次拷贝过程，但是虽然实现录，性能却很不好
uint8_t *gpuPointer;
uint8_t *gpuPointerOld;  //保存最后两次的图像
ID3D11Texture2D *gpu_texture;
cudaArray *cudaArrayPtr;
// 为结果分配CUDA内存

cudaStream_t stream;

/**
 * 返回的是bgra格式的像素
 * */
static byte *copy_ID3D11Texture2D__to_cuda(ID3D11Texture2D *new_image, D3D11_BOX copyRect) {
    D3D11_TEXTURE2D_DESC desc;
    new_image->GetDesc(&desc);

//    int width = 1280;// desc.Width;
//    int height = 720; //desc.Height;
    UINT width = copyRect.right - copyRect.left;
    UINT height = copyRect.bottom - copyRect.top;
    cudaError result;
    if (cudaResource == nullptr) {
        cudaStreamCreate(&stream);

        texture->GetDesc(&desc);
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        HRESULT hr = device->CreateTexture2D(&desc, NULL, &gpu_texture);


        result = cudaGraphicsD3D11RegisterResource(&cudaResource,
                                                   gpu_texture,
                                                   CU_GRAPHICS_REGISTER_FLAGS_NONE);
        if (result != cudaSuccess) {
            std::cout << "cudaGraphicsD3D11RegisterResource failed" << std::endl;
            return nullptr;
        }

        size_t pitch;
        result = cudaMallocPitch((void **) &gpuPointer, &pitch, width * sizeof(uint8_t) * 4, height);
        result = cudaMallocPitch((void **) &gpuPointerOld, &pitch, width * sizeof(uint8_t) * 4, height);

    }



    // context->CopyResource(gpu_texture, new_image);
    context->CopySubresourceRegion(gpu_texture, 0, 0, 0, 0, new_image, 0, &copyRect);
    result = cudaGraphicsMapResources(1, &cudaResource, 0);

    if (result != cudaSuccess) {
        // 错误处理
        std::cout << "cudaGraphicsMapResources failed" << std::endl;
    }

    result = cudaGraphicsSubResourceGetMappedArray(&cudaArrayPtr, cudaResource, 0, 0);

    if (result != cudaSuccess) {
        // 错误处理
        std::cout << "cudaGraphicsSubResourceGetMappedArray failed" << std::endl;
    }

    result = cudaMemcpyFromArrayAsync(gpuPointer, cudaArrayPtr, 0, 0, width * height * 4, cudaMemcpyDeviceToDevice,
                                      stream);

    if (result != cudaSuccess) {
        // 错误处理
        std::cout << "cudaMemcpyFromArray failed" << result << std::endl;
    }

    bool similar = imageSimilar(gpuPointer, gpuPointerOld, width, height, stream);

    //cudaMemcpy(gpuPointerOld, gpuPointer, width * height * 4, cudaMemcpyDeviceToDevice);

    //cudaMemset(deviceResult, 0, 8);
    // 调用CUDA核函数


    //uint8_t *rgbBytes;
    //cudaMalloc(&rgbBytes, width * height * 3);

    //  rgba2rgbProxy(gpuPointer, rgbBytes, width, height, stream);


    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    if(similar){
       //  std::cout << "相同图片" << std::endl;
        return nullptr;
    }else{
     //   std::cout << "不相同图片" << std::endl;
        auto start = std::chrono::system_clock::now();
        cudaMemcpy(gpuPointerOld, gpuPointer, width * height * 4, cudaMemcpyDeviceToDevice);
        auto end = std::chrono::system_clock::now();
//        std::cout << "gpu copy time: "
//                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms"
//                  << std::endl;
        return gpuPointerOld;
    }


    //calculateSumProxy(rgbBytes, width, height, deviceResult,stream);
//    cudaStreamSynchronize(stream);
//
//
//    cudaMemcpyAsync(test_result, deviceResult, sizeof(unsigned long long), cudaMemcpyDeviceToHost,stream);
//    cudaStreamSynchronize(stream);
//    std::cout << "统计结果:" << (test_result[0]) << std::endl;
//    cudaGraphicsUnmapResources(1, &cudaResource, 0);
//    // cudaGraphicsUnregisterResource(cudaResource);
//
//   // cudaFree(gpuPointer);
//   // cudaFree(deviceResult);
//    cudaFree(rgbBytes);

}

// gpuDate 是否直接返回gpu指针
static byte *copy_shmem_tex(struct game_capture *gc, D3D11_BOX copyRect, bool gpuDate = false) {

    if (gpuDate) {
        return copy_ID3D11Texture2D__to_cuda(texture, copyRect);
    }


    D3D11_TEXTURE2D_DESC frame_desc;
    texture->GetDesc(&frame_desc);

    /**
     * //我的理解是，此时渲染到屏幕上的数据是在gpu中或者是只有gpu能访问的，这个是用来把
     *
     * 先创建一个ID3D11Texture2D，把数据拷贝到这个上边，也就是先拷贝到new_image上
     * */
    ID3D11Texture2D *new_image = NULL;

    UINT width = copyRect.right - copyRect.left;
    UINT height = copyRect.bottom - copyRect.top;
    frame_desc.Width = width;
    frame_desc.Height = height;
    frame_desc.Usage = D3D11_USAGE_STAGING;
    frame_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    frame_desc.BindFlags = 0;
    frame_desc.MiscFlags = 0;
    frame_desc.MipLevels = 1;
    frame_desc.ArraySize = 1;
    frame_desc.SampleDesc.Count = 1;

    HRESULT hr = device->CreateTexture2D(&frame_desc, NULL, &new_image);

    //拷贝部分区域
    context->CopySubresourceRegion(new_image, 0, 0, 0, 0, texture, 0, &copyRect);

    //  context->CopyResource(new_image, texture); //全拷贝


    auto start = std::chrono::system_clock::now();
    IDXGISurface *dxgi_surface = NULL;
    hr = new_image->QueryInterface(__uuidof(IDXGISurface), (void **) (&dxgi_surface));
    new_image->Release();

    DXGI_MAPPED_RECT rect;
    //auto t_start = std::chrono::high_resolution_clock::now();

    hr = dxgi_surface->Map(&rect, DXGI_MAP_READ);

    //auto t_end = std::chrono::high_resolution_clock::now();
    //float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    //std::cout << "copy take: " << total_inf << " ms." << std::endl;

    int total = rect.Pitch * height;
    BYTE *data = rect.pBits;

    dxgi_surface->Unmap();
    dxgi_surface->Release();
    dxgi_surface = nullptr;

    return data;
}

#define VENDOR_ID_INTEL 0x8086
#define IGPU_MEM (512 * 1024 * 1024)

static inline bool init_shmem_capture(struct game_capture *gc) {
    const uint32_t dxgi_format = gc->global_hook_info->format;
    const bool convert_16bit = is_16bit_format(dxgi_format);
    const enum gs_color_format format =
            convert_16bit ? GS_BGRA : convert_format(dxgi_format);

    //gs_texrender_destroy(gc->extra_texrender);
    gc->extra_texrender = NULL;

    gc->extra_texture = NULL;

    gc->texture = NULL;
    //gs_texture_t* const texture =
    //	gs_texture_create(gc->cx, gc->cy, format, 1, NULL, GS_DYNAMIC);

    //bool success = texture != NULL;

    //if (success) {
    const bool linear_sample = format != GS_R10G10B10A2;

    gs_texrender_t *extra_texrender = NULL;
    //if (!linear_sample) {
    //	extra_texrender =
    //		gs_texrender_create(GS_RGBA16F, GS_ZS_NONE);
    //	success = extra_texrender != NULL;
    //	if (!success)
    //		warn("init_shmem_capture: failed to create extra texrender");
    //}

    //if (success) {
    gc->texture_buffers[0] = (uint8_t *) gc->data +
                             gc->shmem_data->tex1_offset;
    gc->texture_buffers[1] = (uint8_t *) gc->data +
                             gc->shmem_data->tex2_offset;
    gc->convert_16bit = convert_16bit;

    //	gc->texture = texture;
    gc->extra_texture = NULL;
    gc->extra_texrender = extra_texrender;
    gc->linear_sample = linear_sample;
    //gc->copy_texture = copy_shmem_tex;
    //	}
    //}
    //else {
    //	warn("init_shmem_capture: failed to create texture");
    //}

    return true;
}

void InitCompiler() {
    char d3dcompiler[40] = {};
    int ver = 49;

    while (ver > 30) {
        sprintf(d3dcompiler, "D3DCompiler_%02d.dll", ver);

        HMODULE module = LoadLibraryA(d3dcompiler);
        if (module) {
            pD3DCompile d3dCompile = (pD3DCompile) GetProcAddress(module,
                                                                  "D3DCompile");

#ifdef DISASSEMBLE_SHADERS
            d3dDisassemble = (pD3DDisassemble)GetProcAddress(
                module, "D3DDisassemble");
#endif
            if (d3dCompile) {
                return;
            }

            FreeLibrary(module);
        }

        ver--;
    }

    throw "Could not find any D3DCompiler libraries. Make sure you've "
          "installed the <a href=\"https://obsproject.com/go/dxwebsetup\">"
          "DirectX components</a> that OBS Studio requires.";
}

static inline bool init_shtex_capture(struct game_capture *gc) {
    InitCompiler();

    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));

    int adapterIdx = 0;
    {
        std::vector<uint32_t> adapterOrder;
        ComPtr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC desc;
        uint32_t iGPUIndex = 0;
        bool hasIGPU = false;
        bool hasDGPU = false;
        int idx = 0;

        while (SUCCEEDED(factory->EnumAdapters(idx, &adapter))) {
            if (SUCCEEDED(adapter->GetDesc(&desc))) {
                if (desc.VendorId == VENDOR_ID_INTEL) {
                    if (desc.DedicatedVideoMemory <= IGPU_MEM) {
                        hasIGPU = true;
                        iGPUIndex = (uint32_t) idx;
                    } else {
                        hasDGPU = true;
                    }
                }
            }

            adapterOrder.push_back((uint32_t) idx++);
        }
        /* Intel specific adapter check for Intel integrated and Intel
 * dedicated. If both exist, then change adapter priority so that the
 * integrated comes first for the sake of improving overall
 * performance */
        if (hasIGPU && hasDGPU) {
            adapterOrder.erase(adapterOrder.begin() + iGPUIndex);
            adapterOrder.insert(adapterOrder.begin(), iGPUIndex);
            adapterIdx = adapterOrder[adapterIdx];
        }
    }

    // 这里被坑死了， 调用OpenSharedResource一直失败，浪费我两个星期，后来debug obs才发现：
    // obs里边虽然adapterIdx是0，但是它获取到的是我的nvida显卡3060， 我用0获取到的是集显，
    // 经测试在我自己电脑上用adapterIdx=1可以获取到3060
    adapterIdx = 0; //哈哈哈  引入cuda的库后这个值变成了1，上次折磨我两星期，这次我直接想到是这里哈哈哈，反正不是0就是1
    //这里是枚举所有设备，因为我这里只有第一个接口可以找到IDXGIOutput，所有就没有枚举
    hr = factory->EnumAdapters1(adapterIdx, &adapter);
    if (FAILED(hr))
        warn("Failed to enumerate DXGIAdapter");

    {
        std::wstring adapterName;
        DXGI_ADAPTER_DESC desc;
        HRESULT hr = 0;
        adapterName = (adapter->GetDesc(&desc) == S_OK) ? desc.Description
                                                        : L"<unknown>";
        int ife = 12;
    }

    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    D3D_FEATURE_LEVEL levelUsed = D3D_FEATURE_LEVEL_10_0;

    hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL,
                           D3D11_CREATE_DEVICE_BGRA_SUPPORT, featureLevels,
                           sizeof(featureLevels) /
                           sizeof(D3D_FEATURE_LEVEL),
                           D3D11_SDK_VERSION, device.Assign(), &levelUsed,
                           context.Assign());

    if (FAILED(hr)) {
        warn("D3D11CreateDevice失败");
    }

    {
        ComQIPtr<IDXGIDevice1> dxgiDevice(device);
        const HRESULT hr = dxgiDevice->SetMaximumFrameLatency(16);
        if (FAILED(hr)) {
            warn("SetMaximumFrameLatency failed");
        }
    }

    hr = device->OpenSharedResource((HANDLE) (uintptr_t) gc->shtex_data->tex_handle, __uuidof(ID3D11Texture2D),
                                    (void **) texture.Assign());

    D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc{};
    memset(&viewDesc, 0, sizeof(viewDesc));
    viewDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    viewDesc.Texture2D.MipLevels = 1;

    ComPtr<ID3D11ShaderResourceView> shaderRes;
    ComPtr<ID3D11ShaderResourceView> shaderResLinear;

    hr = device->CreateShaderResourceView(texture, &viewDesc,
                                          shaderRes.Assign());

    if (FAILED(hr))
        std::cout << "失败";

    D3D11_SHADER_RESOURCE_VIEW_DESC viewDescLinear{};

    viewDescLinear = viewDesc;
    viewDescLinear.Format = DXGI_FORMAT_B8G8R8A8_UNORM;// DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    shaderResLinear = shaderRes;

    D3D11_FEATURE_DATA_D3D11_OPTIONS opts = {};
    hr = device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &opts,
                                     sizeof(opts));
    if (FAILED(hr) || !opts.ExtendedResourceSharing) {
        return CAPTURE_FAIL;
    }

    return true;
}

static bool start_capture(struct game_capture *gc) {
    if (gc->global_hook_info->type == CAPTURE_TYPE_MEMORY) {
        if (!init_shmem_capture(gc)) {
            return false;
        }

        info("memory capture successful for %S, %S, %S", gc->config.title, gc->config.klass, gc->config.executable);
    } else {
        if (!init_shtex_capture(gc)) {
            return false;
        }

        info("shared texture capture successful");
    }

    return true;
}

static inline bool capture_valid(struct game_capture *gc) {
    if (!gc->dwm_capture && !IsWindow(gc->window))
        return false;

    return !object_signalled(gc->target_process);
}

bool stop_game_capture(void *data) {


    struct game_capture *gc = (game_capture *) data;
    ipc_pipe_server_free(&gc->pipe);

    if (gc->hook_stop) {
        SetEvent(gc->hook_stop);
    }

    //stop_capture(gc);
    return true;
}


void cudaFreeProxy(void *data) {
    cudaFree(data);
}


byte *game_capture_tick(struct game_capture *gc, float seconds, boolean gpuDate, D3D11_BOX rect) {
    byte *captureImg{};
    // struct game_capture *gc = (game_capture *) data;

    gc->activate_hook = true;

    if (gc->hook_stop && object_signalled(gc->hook_stop)) {
        debug("hook stop signal received");
        stop_capture(gc);
    }

    if (gc->active && !gc->hook_ready && gc->process_id) {
        gc->hook_ready = open_event_gc(gc, EVENT_HOOK_READY);
    }

    if (gc->injector_process && object_signalled(gc->injector_process)) {
        DWORD exit_code = 0;

        GetExitCodeProcess(gc->injector_process, &exit_code);
        close_handle(&gc->injector_process);

        if (exit_code != 0) {
            warn("inject process failed: %ld", (long) exit_code);
            gc->error_acquiring = true;
        } else if (!gc->capturing) {
            gc->retry_interval =
                    ERROR_RETRY_INTERVAL *
                    hook_rate_to_float(gc->config.hook_rate);
            stop_capture(gc);
        }
    }

    if (gc->hook_ready && object_signalled(gc->hook_ready)) {
        debug("capture initializing!");
        enum capture_result result = init_capture_data(gc);

        if (result == CAPTURE_SUCCESS)
            gc->capturing = start_capture(gc);
        else
                debug("init_capture_data failed");

        if (result != CAPTURE_RETRY && !gc->capturing) {
            gc->retry_interval =
                    ERROR_RETRY_INTERVAL *
                    hook_rate_to_float(gc->config.hook_rate);
            stop_capture(gc);
        }
    }

    gc->retry_time += seconds;

    if (!gc->active) {
        if (!gc->error_acquiring &&
            gc->retry_time > gc->retry_interval) {
            if (gc->config.mode == CAPTURE_MODE_ANY ||
                gc->activate_hook) {
                try_hook(gc);
                gc->retry_time = 0.0f;
            }
        }
    } else {
        if (!capture_valid(gc)) {
            info("capture window no longer exists, "
                 "terminating capture");
            stop_capture(gc);
        } else {
            if (gc->copy_texture) {
                //obs_enter_graphics();
                //gc->copy_texture(gc);
                //obs_leave_graphics();
            }
            if (texture) {
                captureImg = copy_shmem_tex(gc, rect, gpuDate);
            }

            gc->fps_reset_time += seconds;
            if (gc->fps_reset_time >= gc->retry_interval) {
                reset_frame_interval(gc);
                gc->fps_reset_time = 0.0f;
            }
        }
    }

    if (!gc->showing)
        gc->showing = true;


    return captureImg;
}


byte *game_capture_tick_cpu(struct game_capture *gc, float seconds, int x, int y, int width, int height) {
//    if (x < 0 || x > gc->cx) {
//        return nullptr;
//    }
//    if (y < 0 || y > gc->cy) {
//        return nullptr;
//    }
//    width = min(gc->cx - x, width);
//    height = min(gc->cy - y, height);

    // 计算源纹理和目标纹理的区域
    D3D11_BOX srcBox;
    srcBox.left = x;
    srcBox.top = y;
    srcBox.right = x + width;  // 区域宽度
    srcBox.bottom = y + height;  // 区域高度
    srcBox.front = 0;
    srcBox.back = 1;
    return game_capture_tick(gc, seconds, false, srcBox);
}

// 返回的bgra格式 gpu上的指针
byte *game_capture_tick_gpu(struct game_capture *gc, float seconds, int x, int y, int width, int height) {
    if (x < 0 || x > gc->cx) {
        return nullptr;
    }
    if (y < 0 || y > gc->cy) {
        return nullptr;
    }

    width = min(gc->cx - x, width);
    height = min(gc->cy - y, height);


    // 计算源纹理和目标纹理的区域
    D3D11_BOX srcBox;
    srcBox.left = x;
    srcBox.top = y;
    srcBox.right = x + width;  // 区域宽度
    srcBox.bottom = y + height;  // 区域高度
    srcBox.front = 0;
    srcBox.back = 1;
    return game_capture_tick(gc, seconds, true, srcBox);
}


extern "C" bool load_graphics_offsets(bool is32bit, bool use_hook_address_cache,
                                      const char *config_path);

LPCWSTR charToLPCWSTR(const char *c) {
    int len = MultiByteToWideChar(CP_UTF8, 0, c, -1, NULL, 0);
    LPWSTR wideStr = new WCHAR[len];
    MultiByteToWideChar(CP_UTF8, 0, c, -1, wideStr, len);
    LPCWSTR lpcwStr = wideStr;
    return lpcwStr;
}


void *init_csgo_capture(const char *windowName, const char *windowClassName) {
    //const char* windowName, const char* w2
    HRESULT *phr;

    bool result = load_graphics_offsets(false, false, "");
    result = load_graphics_offsets(true, false, "");
    //windowName = "Counter - Strike: Global Offensive - Direct3D 9";
    //windowClassName = "Valve001";



    // windowName = L"Counter - Strike: Global Offensive - Direct3D 9";
    // windowClassName = L"Valve001";
    //
    // windowName = L"Apex Legends";
    // windowClassName = L"Respawn001";
    //
    // windowName = L"守望先锋";
    // windowClassName = L"TankWindowClass";




    game_capture_config config{};

    config.anticheat_hook = 1;
    config.force_shmem = false;

    return init(charToLPCWSTR(windowClassName), charToLPCWSTR(windowName), &config, 60);

}