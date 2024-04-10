from subprocess import Popen, PIPE            #Used to run the C program to draw a box to the screen.
import pynput as iput                       #Used to capture mouse and keyboard inputs.
from pynput.mouse import Button, Listener   
import threading                            #Used for multiprocessing/threading.
from mss import mss                         #Used to capture a portion of the screen and then save to PNG.
import mss as ms
import os                                   #Used to manage files for ./data management.
import keyboard                             #Used to wait until kill key is pressed to end data gathering.
import sys
import psutil
import pickle
import signal

#%%

'''
This class is used in capturing data for the model to use. 
It is responsible for both data gathering as well as data cleaning and 
processing 
'''
class ScreenCollection():
    
    area = []           #Used to store the cordinates of the area being captures. [topleftX,topleftY,bottomRightX,bottomRightY]
    areagrab = {}       #Mss uses dictionaries to store the locations, they also do not use purely cordinates.
    filelocation = ""   #Holds the file location of the data being captured.
    imageNumber = 0     #Used to count the number of images created for use in image name generation.
    inputs = []         #The variable used to store inputs
    ended = False       #Global flag to cease all listeners
    
    '''
    Initalise function
    
    Responcible for starting up the application by establishing box location as
    well as naming the data file among other responcibilities.
    '''
    def __init__(self):
        self.setFileLocation()
        print("Please select where you want to capture")
        self.boxlistener = iput.mouse.Listener(on_click=self.getBoxLocation) # fix this trash[]
        self.boxlistener.start()
        self.boxlistener.join()
        self.setBoxArea()
        self.collectimages()
    
    '''
    Sets the dictionary based off of area.
    '''
    def setBoxArea(self):
        self.areagrab = {"left": self.area[0], "top": self.area[1], "width": self.area[2]-self.area[0], "height":self.area[3]-self.area[1]}
    
    '''
    Wipes the data file should the user wish to overwrite.
    '''
    def wipeFile(self, location):
        files = os.listdir(location)
        for file in files:
            os.remove(location + "\\" + file)
    
    '''
    Asks the user for the name of the application being captured, 
    if the file exists give the user an option to overwrite it.
    
    If the folder doesn't exist, create a new folder.'
    '''
    def setFileLocation(self):
        name = input("Please enter the name of the application being captured from: ")
        file = f"data\{name}"
        if os.path.isdir(file):
            responce = input("Directory exists, overwrite? [y/n]")
            if responce.lower() == "y":
                print("Folder wiped")
                self.wipeFile(file)
        else:
            os.mkdir(file)
        self.filelocation = file

    '''
    Listener function responding to mouse click events.
    x: how far from the farmost left part of the screen.
    y: how far fro mthe topmost part of the screen.
    button: Which button was interacted with?
    pressed: Is the button being pressed or let go of.
    '''
    def getBoxLocation(self, x,y,button,pressed):
        if pressed == False: #Checks to see if this is a button click, and not the user releasing the button.
            return
        elif button == Button.left:
            self.area.append(x)
            self.area.append(y)
        if len(self.area) == 4: # If 2 sets of cordinates are captured, stop the listener and continue program.
            self.boxlistener.stop()
    
    '''
    This function is used to neatly flatten the area array into 4 variables.
    Also transforms them into strings for use in console applications like 
    drawBox
    '''
    def areaunpack(self):
        return [str(a) for a in self.area]
    
    '''
    Function responcible for running the C program. it starts a seperate 
    process
    
    The C program is passed the nesicery cordinates to draw the box.
    
    Then the collection process is finished, triggering a keyboard interrupt
    with ctrl + c will end the process.
    '''
    def drawBox(self):
        x1,y1,x2,y2 = self.areaunpack()
        self.process = Popen(['Screen Writer\Screen Writer.exe', x1, y1, x2, y2], shell = False)
        self.process.wait()
        print("drawer ended")
        

    '''
    Create seperate function  due to how multiprocessing works this
    function is required to not interfere with the rest of the function.
    '''
    def getImageListener(self):
        listener = iput.mouse.Listener(on_click=self.getImage)
        listener.start()
        listener.join()

    '''
    Listener function for image collection
     if left click is pressed and it is within the designated area
     
     mss will grab an image, and then save it to filelocationq
     image number will increment as to not overwrite or create duplicate image 
     names.
    '''
    def getImage(self,x,y,button,pressed):
        print("ran")
        if pressed == False:
            return
        if self.ended:
            return False
        if button == Button.left:
            if self.withinArea(x,y):
                with mss() as sct:
                    image = sct.grab(self.areagrab)
                    ms.tools.to_png(image.rgb, image.size, output=f"{self.filelocation}\{self.imageNumber}.png")
                    self.imageNumber+=1
                self.inputs.append([(x,y),button])

    '''
    Checks to see if click is within the predetermined area before taking a 
    screenshot.
    '''
    def withinArea(self, x,y):
        if x > self.area[0] and x < self.area[2] and y > self.area[1] and y < self.area[3]:
            print("within area")
            return True
        else:
            print("not within")
            return False
    
    '''
    Used to save all collected inputs to a pickle file for use later.
    '''
    def saveInputs(self):
        with open(f"{self.filelocation}\inputs.pkl", "wb") as file:
            pickle.dump(self.inputs, file)
    
    '''
    Function responcible for managing image collection threads.
    '''
    def collectimages(self):
        boxDrawThread = threading.Thread(target=self.drawBox, daemon=True)
        mouseListenerThread = threading.Thread(target=self.getImageListener, daemon=True)
        boxDrawThread.start()
        mouseListenerThread.start()
        keyboard.wait("q")
        self.ended = True
        self.process.send_signal(signal.SIGTERM)
        self.saveInputs()