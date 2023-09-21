#include "main.h"
#include <iostream>
#include <windows.h>
#include <winternl.h>
#include <chrono>
#include <thread>
#pragma comment(lib, "ntdll.lib")

void mouseMove(int x, int y) {
	/* mouse event, move absolutely*/
	INPUT inputs = { 0 };
	inputs.type = INPUT_MOUSE;
	inputs.mi.dx = x;
	inputs.mi.dy = y;
	inputs.mi.dwFlags = MOUSEEVENTF_MOVE;
	SendInput(1, &inputs, sizeof(inputs));
}

void mouseLeftPress(){
	INPUT input = { 0 };
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

	//input[1].type = INPUT_MOUSE;
	//input[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;

	SendInput(1, &input, sizeof(INPUT));
}

void mouseLeftRelease(){
	INPUT input = { 0 };
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
	SendInput(1, &input, sizeof(INPUT));
}


class CreateWindowHandle {
	static void _CreateWindowClass(HINSTANCE moduleHandle, LPCWSTR windowsClassName) {
		WNDCLASSEXW wcex;

		wcex.cbSize = sizeof(WNDCLASSEX);

		wcex.style = CS_HREDRAW | CS_VREDRAW;
		wcex.lpfnWndProc = DefWindowProcW;
		wcex.cbClsExtra = 0;
		wcex.cbWndExtra = 0;
		wcex.hInstance = moduleHandle;
		wcex.hIcon = nullptr;
		wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
		wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
		wcex.lpszMenuName = nullptr;
		wcex.lpszClassName = windowsClassName;
		wcex.hIconSm = nullptr;

		RegisterClassExW(&wcex);
	}

	static HWND _CreateWindow(HINSTANCE moduleHandle, LPCWSTR windowsClassName) {
		auto windowsHandle = CreateWindowExW(0L, windowsClassName, L"Window", WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, moduleHandle, nullptr);

		return windowsHandle;
	}

public:
	static HWND Create() {
		auto moduleHandle = static_cast<HINSTANCE>(GetModuleHandleW(nullptr));

		WCHAR windowsClassName[] = L"fsdfsrewrwegfdgfd";

		CreateWindowHandle::_CreateWindowClass(moduleHandle, windowsClassName);

		return CreateWindowHandle::_CreateWindow(moduleHandle, windowsClassName);
	}
};

void RegisterMouseRawInput(HWND handle) {
	RAWINPUTDEVICE rid;

	rid.usUsagePage = 0x01;
	rid.usUsage = 0x02;
	rid.dwFlags = RIDEV_INPUTSINK;
	rid.hwndTarget = handle;

	RegisterRawInputDevices(&rid, 1, sizeof(rid));
}

void RegisterKeyboardRawInput(HWND handle) {
	RAWINPUTDEVICE rid;

	rid.usUsagePage = 0x01;
	rid.usUsage = 0x06;
	rid.dwFlags = RIDEV_INPUTSINK;
	rid.hwndTarget = handle;

	RegisterRawInputDevices(&rid, 1, sizeof(rid));
}

unsigned int* UsedRawInput(LPARAM lParam) {
	static unsigned int* mouseInfo = new unsigned int[8];

	unsigned long long xy = 0;
	char buffer[1024];
	UINT dwSize = 1024;
	//其实GetRawInputData可以分两步调用,第一次获取输出结构的大小,分配空间后调用第二次,但如果知道大小的话就可以省略第一次,所以我只调用了一次
	GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, buffer, &dwSize, sizeof(RAWINPUTHEADER));

	auto raw = reinterpret_cast<RAWINPUT*>(buffer);

	if (raw->header.dwType == RIM_TYPEMOUSE)
	{
		
		mouseInfo[0] = raw->data.mouse.lLastX;
		mouseInfo[1] = raw->data.mouse.lLastY;
		mouseInfo[2] = raw->data.mouse.ulButtons;

		unsigned long long* temp = (unsigned long long*)mouseInfo;

		temp[2] = (unsigned long long)raw->header.hDevice;
		//mouseInfo[3] = raw->data.mouse.ulRawButtons;
		//mouseInfo[4] = raw->data.mouse.ulExtraInformation;
		//mouseInfo[5] = raw->data.mouse.usFlags;
		//mouseInfo[6] = raw->data.mouse.usButtonData;
		//mouseInfo[7] = raw->data.mouse.usButtonFlags;
	}

	return mouseInfo;
}

void listenerMouseMove() {
	HWND windowHandle = CreateWindowHandle::Create();
	RegisterMouseRawInput(windowHandle);
}

unsigned int* getAbsoluteMove() {
	static MSG msg;
	static unsigned int* temp = new unsigned int[8];
	while (GetMessage(&msg, nullptr, 0, 0))
	{
		if (msg.message == WM_INPUT) {
			unsigned int* xy = UsedRawInput(msg.lParam);
			//如果是在前台获取的原始输入则必须调用DispatchMessage(&msg);来清理
			if (GET_RAWINPUT_CODE_WPARAM(msg.wParam) == RIM_INPUT) {
				DispatchMessage(&msg);
			}
			else {
			}

			return xy;
		}
		else {
			DispatchMessage(&msg);
		}
	}

	return temp;
}

typedef struct {
	char button;
	char x;
	char y;
	char wheel;
	char unk1;
} MOUSE_IO;

#define MOUSE_PRESS 1
#define MOUSE_RELEASE 2
#define MOUSE_MOVE 3
#define MOUSE_CLICK 4

static HANDLE g_input;
static IO_STATUS_BLOCK g_io;

BOOL g_found_mouse;

static BOOL callmouse(MOUSE_IO* buffer)
{
	IO_STATUS_BLOCK block;
	return NtDeviceIoControlFile(g_input, 0, 0, 0, &block, 0x2a2010, buffer, sizeof(MOUSE_IO), 0, 0) == 0L;
}

static NTSTATUS device_initialize(PCWSTR device_name)
{
	UNICODE_STRING name;
	OBJECT_ATTRIBUTES attr;

	RtlInitUnicodeString(&name, device_name);
	InitializeObjectAttributes(&attr, &name, 0, NULL, NULL);

	NTSTATUS status = NtCreateFile(&g_input, GENERIC_WRITE | SYNCHRONIZE, &attr, &g_io, 0,
		FILE_ATTRIBUTE_NORMAL, 0, 3, FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, 0, 0);

	return status;
}

bool mouse_open() {
	NTSTATUS status = 0;

	if (g_input == 0) {
		wchar_t buffer0[] = L"\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}";

		status = device_initialize(buffer0);
		if (NT_SUCCESS(status))
			g_found_mouse = 1;
		else {
			wchar_t buffer1[] = L"\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
			status = device_initialize(buffer1);
			if (NT_SUCCESS(status))
				g_found_mouse = 1;
		}
	}
	return status == 0;
}

void mouse_close() {
	if (g_input != 0) {
		//ZwClose(g_input);
		CloseHandle(g_input);
		g_input = 0;
	}
}

void mouse_option(char button, char x, char y, char wheel)
{
	MOUSE_IO io;
	io.unk1 = 0;
	io.button = button;
	io.x = x;
	io.y = y;
	io.wheel = wheel;

	if (!callmouse(&io)) {
		mouse_close();
		mouse_open();
	}
}

void mouse_move(char x, char y, char button) {
	mouse_option(button, x, y, 0);
}

int main2() {
	HWND windowHandle = CreateWindowHandle::Create();

	//RegisterKeyboardRawInput(windowHandle);

	RegisterMouseRawInput(windowHandle);

	MSG msg;

	while (GetMessage(&msg, nullptr, 0, 0))
	{
		if (msg.message == WM_INPUT) {
			UsedRawInput(msg.lParam);
			//如果是在前台获取的原始输入则必须调用DispatchMessage(&msg);来清理
			if (GET_RAWINPUT_CODE_WPARAM(msg.wParam) == RIM_INPUT) {
				DispatchMessage(&msg);
			}
			else {
			}
		}
		else {
			DispatchMessage(&msg);
		}
	}

	return (int)msg.wParam;
}
using std::this_thread::sleep_for;
int main(){

    bool result = mouse_open();

    for (int i = 0; i < 10; ++i) {
        mouse_move(5,5,0);
        sleep_for(std::chrono::milliseconds(50));
    }


}