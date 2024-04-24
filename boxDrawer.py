from subprocess import Popen, PIPE            #Used to run the C program to draw a box to the screen.
import threading
import signal

class ScreenDraw():
    '''
    This class contains all the elements for drawing the box to the screen. On
    top of this, it also manages the thread and sending signals to said thread.
    '''
    
    def __init__(self, area):
        self.boxDrawThread = threading.Thread(target=self.drawBox, daemon=True, args=(area,))
        self.boxDrawThread.start()
    
    def areaUnpack(self, area):
        '''
        This function is used to neatly flatten the area array into 4 variables.
        Also transforms them into strings for use in console applications like 
        drawBox

        Parameters:
            area (list of ints): the desginated area for the box
            
        Return:
            list of area converted to strings
        '''
        return [str(a) for a in area]
    
    
    def drawBox(self, area):
        '''
        Function responcible for running the C program. it starts a seperate 
        process
        
        The C program is passed the nesicery cordinates to draw the box.
        
        Then the collection process is finished, triggering a keyboard interrupt
        with ctrl + c will end the process.
        
        Parameters:
            area (list): The desginated area for the box.
        '''
        x1,y1,x2,y2 = self.areaUnpack(area)
        self.process = Popen(['Screen Writer\Screen Writer.exe', x1, y1, x2, y2], shell = False)
        self.process.wait()
    
    def end(self):
        '''
        Ends the thread by sending a terminate signal to the C program.
        '''
        self.process.send_signal(signal.SIGTERM)
        self.process.wait()
        self.boxDrawThread.join()
        print("Finished drawing")
    