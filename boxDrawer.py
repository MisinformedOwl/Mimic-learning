from subprocess import Popen, PIPE            #Used to run the C program to draw a box to the screen.
import threading
import signal

class ScreenDraw():
    
    '''
    This function is used to neatly flatten the area array into 4 variables.
    Also transforms them into strings for use in console applications like 
    drawBox
    '''
    def areaUnpack(self, area):
        return [str(a) for a in area]
    
    '''
    Function responcible for running the C program. it starts a seperate 
    process
    
    The C program is passed the nesicery cordinates to draw the box.
    
    Then the collection process is finished, triggering a keyboard interrupt
    with ctrl + c will end the process.
    '''
    def drawBox(self, area):
        x1,y1,x2,y2 = self.areaUnpack(area)
        self.process = Popen(['Screen Writer\Screen Writer.exe', x1, y1, x2, y2], shell = False)
        self.process.wait()
    
    def __init__(self, area):
        self.boxDrawThread = threading.Thread(target=self.drawBox, daemon=True, args=(area,))
        self.boxDrawThread.start()
    
    def end(self):
        self.process.send_signal(signal.SIGTERM)
        self.process.wait()
        self.boxDrawThread.join()
        print("Finished drawing")
    