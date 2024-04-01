#include <windows.h>
#include <conio.h>
#include <ctype.h>
#include <stdio.h>

int main() {
    // Get the device context for the entire screen
    HDC hdcScreen = GetDC(NULL);

    // Define the coordinates of the rectangle
    RECT rect = {100, 100, 200, 200};

    // Create a red solid brush
    HBRUSH hBrush = CreateSolidBrush(RGB(255, 0, 0));

    // Predefine variable for checking if correct cahracter is pressed.
    char c;
    while(1){
    	if (kbhit()){
    		c = tolower(getch());
    		printf("%c", c);
    		if(c == 'q'){
    			break;
    		}
    	}
		// Draw a red rectangle outline
		FrameRect(hdcScreen, &rect, hBrush);
    }

    // Release the device context
    ReleaseDC(NULL, hdcScreen);

    // Delete the brush
    DeleteObject(hBrush);

    return 0;
}
