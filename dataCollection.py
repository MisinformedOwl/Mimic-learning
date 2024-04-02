from subprocess import run, PIPE
import pynput as iput
from pynput.mouse import Button, Listener
import threading
from mss import mss
import os

#%%

class ScreenCollection():
    
    area = []
    filelocation = ""
    imageNumber = 0
    
    def __init__(self):
        self.setFileLocation()
        print("Please select where you want to capture")
        self.boxlistener = iput.mouse.Listener(on_click=self.getBoxLocation)
        self.boxlistener.start()
        self.boxlistener.join()
        self.collectimages()
    
    def setFileLocation(self):
        name = input("Please enter the name of the application being captured from: ")
        if os.path.isdir(f"data\{name}"):
            responce = input("Directory exists, overwrite? [y/n]")
            if responce.lower == "y"
        os.mkdir(f"data\{name}")
        self.filelocation = f"data\{name}"

    def getBoxLocation(self, x,y,button,pressed):
        if pressed == False:
            return
        if button == Button.left:
            self.area.append(x)
            self.area.append(y)
            print(self.area)
        if len(self.area) == 4:
            self.boxlistener.stop()
    
    def unpack(self):
        return [str(a) for a in self.area]
    
    def drawBox(self):
        x1,y1,x2,y2 = self.unpack()
        try:
            run(['Screen Writer\Screen Writer.exe', x1, y1, x2, y2], stdout=PIPE, text=True)
        except KeyboardInterrupt:
            print("aight")
    
    def getImage(self,x,y,button,pressed):
        areagrab = {"left": self.area[0], "top": self.area[1], "width": self.area[2]-self.area[0], "height":self.area[3]-self.area[1]}
        if pressed == False:
            return
        if button == Button.left:
            if self.withinArea(x,y):
                with mss() as sct:
                    image = sct.grab(areagrab)
                    mss.tools.to_jpg(image.rgb, image.size, output=f"{self.filelocation}\{self.imageNumber}")
                    self.imageNumber+=1
            
    
    def withinArea(self, x,y):
        if x > self.area[0] and x < self.area[2] and y > self.area[1] and y < self.area[3]:
            return True
        else:
            return False
    
    def collectimages(self):
        try:
            boxthread = threading.Thread(target=self.drawBox())
            boxthread.start()
            with iput.Listener(on_click=self.getImage) as Listener:
                Listener.join()
        except KeyboardInterrupt:
            Listener.stop()
            boxthread.stop()
            print("Finished collecting images.")