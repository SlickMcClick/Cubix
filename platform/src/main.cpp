#pragma once

#include <SDL3/SDL.h>
#include <SLD_main.h>
#include <SDL3/SDL_init.h>

#include <platform/include/window.h>

#include <vulkan/vulkan.h>

#include <iostream>
#include <string>
#include <stdexcept>

const char* title = "Cubix";
const int width = 800;
const int height = 800;

void cleanup(window)
{
    SDL_DestroyWindow(window);
}

int main()
{
    SDL_SetAppMetadata(const char* "Cubix", const char* "0.0.1", nullptr)
    SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO);
    auto window = createWindow(*title, width, height, nullptr)


    while (!quit)
    {
        SDL_Event event{};
        while (SDL_PollEvent(&event))
        {
            if (event.key.down)
            {
                
            }
        }
    }

    cleanup(window);

    return 0;
}
