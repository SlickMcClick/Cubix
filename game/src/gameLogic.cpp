#include <iostream>
#include <gameLogic.h>

struct GameData
{
    int positionX = 0;
    int positionY = 0;
    int fpsCounter = 0;
    float timer = 0;
};
static GameData data;

bool initGameplay()
{
    data = {};
    return true;
}

//bool gameplayFrame(float deltaTime, int w, int h, Input &input)
//{
//
    //fps counter
    //Sleep(1);
//    data.fpsCounter += 1;
//    if (data.timer > 1)
//    {
//        data.timer -= 1;
//        std::cout << "FPS: " << data.fpsCounter << '\n';
//        data.fpsCounter = 0;
// }
//
//  return true;
//}

void closeGameLogic()
{
}
