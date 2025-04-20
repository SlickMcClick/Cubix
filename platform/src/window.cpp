#include <SDL3/SDL_video.h>
#include <platform/include/window.h>

#include <iostream>
#include <stdexcept>
#include <string>

SDL_Window* createWindow(const char *title, int w, int h, SDL_WindowFlags flags)
{
    auto window = SDL_Window* SDL_CreateWindow(title, w, h, flags | SDL_WINDOW_VULKAN);
    if (!empty(SDL_GetError()))
    {
        throw std::runtime_error("Failed to create window!");
    }

    return window;
}

