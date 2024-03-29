#include <windows.h>

int main() {
    // Get the device context for the entire screen
    HDC hdcScreen = GetDC(NULL);

    // Define the coordinates of the rectangle
    RECT rect = {100, 100, 200, 200};

    // Create a red solid brush
    HBRUSH hBrush = CreateSolidBrush(RGB(255, 0, 0));

    for(int x=0;x<10000;x++){
    // Draw a red rectangle outline
    FrameRect(hdcScreen, &rect, hBrush);
    }

    // Release the device context
    ReleaseDC(NULL, hdcScreen);

    // Delete the brush
    DeleteObject(hBrush);

    return 0;
}
