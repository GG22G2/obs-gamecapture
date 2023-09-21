#pragma once
#include <windows.h>

#ifndef MOUSE_H
#define MOUSE_H
extern "C" _declspec(dllexport) void mouseMove(int x, int y);
extern "C" _declspec(dllexport) void mouseLeftPress();
extern "C" _declspec(dllexport) void mouseLeftRelease();


extern "C" _declspec(dllexport) void listenerMouseMove();
extern "C" _declspec(dllexport) unsigned int* getAbsoluteMove();

extern "C" _declspec(dllexport) bool mouse_open();
extern "C" _declspec(dllexport) void mouse_close();
extern "C" _declspec(dllexport) void mouse_move(char x, char y, char button);
#endif //MOUSE_H


