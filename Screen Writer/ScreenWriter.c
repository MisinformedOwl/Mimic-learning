#include <windows.h>
#include <conio.h>
#include <ctype.h>
#include <stdio.h>
#include <signal.h>

/**
 * Recieves and handles signals sent by boxDrawer.py
 *
 * @param signal The signal being handled. (Primerily termination signal.)
 */
void signal_handler(int, signal){
	if (signal==SIGTERM) {
		exit(0);
	}
}

/**
 * The meat of the application.
 * This just takes the cordinates given, and endlessly draws a rectangle
 *
 * @param argc Amount of arguments
 * @param argv List of sent parameters
 */
int main(int argc, char* argv[]) {
	signal(SIGTERM, signal_handler);
	for (int i = 0; i < argc; i++){
		printf("%s\n", argv[i]);
	}

    // Get the device context for the entire screen
    HDC hdcScreen = GetDC(NULL);

    // Convert to integer
    int x1 = atoi(argv[1]);
    int y1 = atoi(argv[2]);
    int x2 = atoi(argv[3]);
    int y2 = atoi(argv[4]);

    // Define the coordinates of the rectangle
    RECT rect = {x1,y1,x2,y2};

    // Create a red solid brush
    HBRUSH hBrush = CreateSolidBrush(RGB(255, 0, 0));

    // Predefine variable for checking if correct cahracter is pressed.
    char c;
    while(1){
    	sleep(200);
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
