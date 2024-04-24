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
import boxDrawer

#%%


class ScreenCollection():
    '''
    This class is used in capturing data for the model to use. 
    It is responsible for both data gathering as well as data cleaning and 
    processing 
    '''
    area = []           #Used to store the cordinates of the area being captures. [topleftX,topleftY,bottomRightX,bottomRightY]
    areagrab = {}       #Mss uses dictionaries to store the locations, they also do not use purely cordinates.
    filelocation = ""   #Holds the file location of the data being captured.
    imageNumber = 0     #Used to count the number of images created for use in image name generation.
    inputs = []         #The variable used to store inputs
    ended = False       #Global flag to cease all listeners
    downsize = False    #Flag to switch mode to reduce image size
    width, height = 0,0 #The width and height of the iamge, used in data normalization.
    
    def __init__(self):
        '''
        Initalise function
        
        Responcible for starting up the application by establishing box location as
        well as naming the data file among other responcibilities.
        '''
        self.setFileLocation()
        print("Please select where you want to capture")
        self.boxlistener = iput.mouse.Listener(on_click=self.getBoxLocation) # fix this trash[]
        self.boxlistener.start()
        self.boxlistener.join()
        self.setBoxArea()
        self.collectimages()
    
    def setBoxArea(self):
        '''
        Sets the dictionary based off of area.
        '''
        self.areagrab = {"left": self.area[0], "top": self.area[1], "width": self.area[2]-self.area[0], "height":self.area[3]-self.area[1]}
    
    def wipeFile(self, location):
        '''
        Wipes the data file should the user wish to overwrite.
        
        Parameters:
            location: The name of the file in the data folder being used.
        '''
        files = os.listdir(location)
        for file in files:
            os.remove(location + "\\" + file)
    
    def setFileLocation(self):
        '''
        Asks the user for the name of the application being captured, 
        if the file exists give the user an option to overwrite it.
        
        If the folder doesn't exist, create a new folder.'
        '''
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

    def getBoxLocation(self, x,y,button,pressed):
        '''
        Listener function responding to mouse click events.
        
        Parameters:
            x (int): how far from the farmost left part of the screen.
            y (int): how far fro mthe topmost part of the screen.
            button (string): Which button was interacted with?
            pressed (bool): Is the button being pressed or let go of.
        '''
        if pressed == False: #Checks to see if this is a button click, and not the user releasing the button.
            return
        elif button == Button.left:
            self.area.append(x)
            self.area.append(y)
        if len(self.area) == 4: # If 2 sets of cordinates are captured, stop the listener and continue program.
            self.width = self.area[2] - self.area[0]
            self.height = self.area[3] - self.area[1]
            self.boxlistener.stop()
        
    def getImageListener(self):
        '''
        Create seperate function  due to how multiprocessing works this
        function is required to not interfere with the rest of the function.
        '''
        listener = iput.mouse.Listener(on_click=self.getImage)
        listener.start()
        listener.join()

    
    def getImage(self,x,y,button,pressed):
        '''
        Listener function for image collection
         if left click is pressed and it is within the designated area
         
         mss will grab an image, and then save it to filelocationq
         image number will increment as to not overwrite or create duplicate image 
         names.
         
         Parameters:
             x (int): how far from the farmost left part of the screen.
             y (int): how far fro mthe topmost part of the screen.
             button (string): Which button was interacted with?
             pressed (bool): Is the button being pressed or let go of.
        '''
        if pressed == False:
            return
        if self.ended:
            return False
        if self.withinArea(x,y):
            with mss() as sct:
                image = sct.grab(self.areagrab)
                ms.tools.to_png(image.rgb, image.size, output=f"{self.filelocation}\{self.imageNumber}.png")
                self.imageNumber+=1
            
            #Normalise values
            x = (x - self.area[0]) / self.width
            y = (y - self.area[1]) / self.height
            
            if button == Button.left:
                self.inputs.append([[x,y],"left"])
            elif button == Button.right:
                self.inputs.append([[x,y],"right"])

    def withinArea(self, x,y):
        '''
        Checks to see if click is within the predetermined area before taking a 
        screenshot.
        
        parameters:
            x (int): horizontal cordinate
            y (int): vertical cordinate
        
        returns: 
            Boolean: True if in the area flase otherwise
        '''
        if x > self.area[0] and x < self.area[2] and y > self.area[1] and y < self.area[3]:
            return True
        else:
            return False
    
    def saveInputs(self):
        '''
        Used to save all collected inputs to a pickle file for use later.
        '''
        with open(f"{self.filelocation}\inputs.pkl", "wb") as file:
            pickle.dump(self.inputs, file)
        with open(f"{self.filelocation}\\area.txt", "w") as file:
            file.writelines(f"{self.area[0]} {self.area[1]} {self.area[2]} {self.area[3]}")
    
    def collectimages(self):
        '''
        Function responcible for managing image collection threads.
        '''
        box = boxDrawer.ScreenDraw(self.area)
        mouseListenerThread = threading.Thread(target=self.getImageListener, daemon=True)
        mouseListenerThread.start()
        keyboard.wait("q")
        self.ended = True
        self.saveInputs()
        box.end()