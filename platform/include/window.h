#ifndef WINDOWING
#define WINDOWING

#include <SDL3/SDL_video.h>

SDL_Window* createWindow(const char *title, int w, int h, SDL_WindowFlags flags);

#endif // !WINDOWING
