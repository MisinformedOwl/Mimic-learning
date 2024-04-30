import torch                            #Torch is the framework we will be using to create this neural network.
import torchvision
from torchvision import transforms
import torch.nn.functional as F         #Activation functions
from torchvision.utils import draw_bounding_boxes, save_image
import torch.nn.init as init
from torch.utils.data import DataLoader #Dataloaders used in batching data.
from torch import optim, nn             #Imports for optimsers and neural network layers such as linear.
import graphviz                         # Imported as it's needed to be used with torchviz apparently.
from torchviz import make_dot as viz    #Library used to create an image of the current neural network.
import pickle
import glob
import os
import numpy as np
from cv2 import imread, imshow, resize, INTER_AREA
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import keyboard
from mss import mss                         #Reused library. turn into own module?
import boxDrawer
import mouse
import keyboard
import time
from RulesBasedLearning import RulesBased as RBL


class CNNModel(nn.Module):
    '''
    CNNModel class will hold the CNN.
    
    Parameters: 
        nn.Module: A class which contains all nesicery functions to create a neural network.
    '''
    history = []
    criterion = nn.MSELoss()
    batchsize = 8
    learningRate = 0.001
    varyingLearningRate = [0.01,0.001,0.0001]
    testsize = 10 # in %
    def __init__(self):
        '''
        We begin with 512 as it's not too small to not capture detail. However 
        it is also not big enough to cause issues in memory.
        '''
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.to(self.device)
        
        self._SimpleModel()
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                m.weight.data = m.weight.data.to(self.device)
                m.bias.data = m.bias.data.to(self.device)
            elif isinstance(m, nn.BatchNorm2d):
                m = m.to(self.device)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                assert m.weight.sum().item() != 0

        self.optimizer = optim.Adam(self.parameters(), lr = self.learningRate)        
    

    def _SimpleModel(self):
        '''
        Makes a simple version of the model. Which does not have many layers.
        '''
        self.layer1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)               # Collecting image of [512,512,3] and colecting featuresto create a matrix of [512,512,10]
        self.layer2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)              # Collecting image of [512,512,3] and colecting featuresto create a matrix of [512,512,10]
        self.layer3 = nn.MaxPool2d(2,2)                                        #Maxpool halves the size of the image. Image is now [256,256,10]
        self.layer4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)              #Collecting image of [512,512,3] and colecting featuresto create a matrix of [256,256,10]
        self.layer5 = nn.MaxPool2d(2,2)                                        #Maxpool halves the size of the image. Image is now [128,128,10]
        self.layer6 = nn.Conv2d(8, 10, kernel_size=3, padding=1)              #[128,128,10]->[128,128,20]
        self.layer7 = nn.Conv2d(10, 10, kernel_size=3, padding=1)              #[128,128,10]->[128,128,20]
        
        self.flatten = nn.Flatten()
        
        self.lin1 = nn.Linear(10240, 5120)            #124x124x1 = 15,376
        self.lin2 = nn.Linear(5120, 1024)            #124x124x1 = 15,376
        self.lin3 = nn.Linear(1024, 32)            #124x124x1 = 15,376
        self.lin4 = nn.Linear(32, 2)
    
    def forward(self, x):
        '''
        This function is what is activated when actually training/testing the AI.
        This is a useful tool in torch, as it allows me to intercept hidden layer
        and add extra functionality, like checking the shape of the data or even
        add rule based learning which i am planning on doing.
        
        Parameters:
            x (tensor): The images being passed in shape [batchSize, channels, width, height]
        
        returns:
            Tensor list: Of normalized predicted cordinates
        '''
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        
        x = self.flatten(x)
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        
        
        return x
    
    
    def displayCNN(self, x):
        '''
        This method is used to create and render an image of the current neural 
        network should the user desire.
        
        Parameters:
            x (CNNModel): the model
        '''
        viz(x, params=dict(list(self.named_parameters()))).render("cnn_viz", format="png")
    
    
    def grabData(self, datafolder, resizeData=True, test = False):
        '''
        Grabs the data from file and puts it into a dataloader for use in the model.
        
        [This needs reworked to be better.]
        
        Parameters:
            datafolder (String): Containing the folder name. being taken from
            resizeData (bool): Is the data in question being resized?
            test (bool): Is this data for use in testing? if not. use training data.
            
        return:
            DataLoader, Set: The specialised tool used for batch loading in torch. and a set of unique characters
        '''
        if datafolder == None:
            name = input("Name of the application being used: ")
            self.location = f"data/{name}"
        else:
            self.location = f"data/{datafolder}"
        if not os.path.isdir(self.location):
            raise FileNotFoundError
        
        images = glob.glob(f"{self.location}/*.png")
        images = sorted(images, key = len)
        
        with open(f"{self.location}/area.txt", "r") as file:
            self.area = [int(a) for a in file.readlines()[0].split(" ")]
        
        #Grabbing user inputs and locations.
        with open(f"{self.location}/inputs.pkl", "rb") as file:
            inputs = pickle.load(file)
        
        unique = list(set(l for _, l in inputs))
        
        initialSize = len(inputs)
        testIndex = round((len(inputs)/100)*self.testsize)
        if test:
            inputs = inputs[:testIndex]
            images = images[:testIndex]
        else:
            inputs = inputs[testIndex:]
            images = images[testIndex:]
        
        assert initialSize > len(inputs)
        
        data = []
        if resizeData: #If the data is being resized to 512
            for image, i in zip(images, inputs):
                image = imread(image)
                image = image /255
                image = image.round(4)
                assert (image > 1).sum() == 0
                image = self.preprocessImage(image)
                data.append([image, torch.tensor(i[0]).to(self.device), i[1]])
        else:
            for image, i in zip(images, inputs):
                data.append([torch.tensor(np.transpose(imread(image), (2,0,1)), dtype=torch.float).to(self.device), torch.tensor(i[0]).to(self.device), i[1]])
        
        del images, image, i, initialSize, testIndex, inputs
        return DataLoader(data, batch_size=self.batchsize, shuffle=False), unique

    def preprocessImage(self, image):
        '''
        Preprocesses the image resizing the image and changing it's data type to 
        float to suit the model's requirements
        
        Parameters:
            image (numpy array): The image being worked on
        
        Return:
            Tensor: Of the image fully preprocessed.
        '''
        return torch.tensor(np.transpose(resize(image, (128, 128), INTER_AREA), (2,0,1)), dtype=torch.float).to(self.device)

    def createRules(self, unique):
        '''
        Creates the rule based learning for the model.
        
        Parameters:
            unique (set): A set of unique actions that were displayed during data capturing.
        '''
        self.rules = RBL(unique, 20)

    def learn(self, epochs, location=None):
        '''
        Train function used for training the model.
        
        Parameters:
            epochs (int): The amount of times the AI will train on the same training data.
            location (String): The location of the data which will be trained on.
        
        Return:
            float: The overall loss of the model during training.
        '''
        trainloader, unique = self.grabData(location, True, False)
        self.createRules(unique)
        self.train()
        for epoch in tqdm(range(epochs)):
            runningloss = 0
            for image, cords, button in trainloader:
                self.optimizer.zero_grad()
                y_pred = self(image)
                loss = self.criterion(y_pred, cords)
                loss.backward()
                self.optimizer.step()
                runningloss += loss.item()/self.batchsize
                self.rules.weightTrain(y_pred, button)
            self.history.append(runningloss)
        del trainloader, image, cords, button, y_pred, loss
        torch.cuda.empty_cache()
        return runningloss
    
    def denormalise(self, image, cord):
        '''
        Returns the image to it's original sizes and changes model output into 
        a position on the image between 0 and width/height
        AKA, reverts normalisation
        
        Parameters:
            image (tensor): The image to be denormalized
            cord ([int,int]): The output of the model
            
        Return:
            tensor, [int,int]: Denormalised inputs
        '''
        image= (image*255).to(torch.uint8)
        width = image.shape[1]
        height = image.shape[2]
        cord = torch.tensor([[cord[0]*width-10, cord[1]*height-10, cord[0]*width+10, cord[1]*height+10]])
        return image, cord
    
    def graph(self):
        '''
        Graphs the model's history
        '''
        plt.plot(self.history)

    def setBoxArea(self, area): #Repeat code. fix it.
        '''
        Sets the dictionary based off of area.
        
        Return:
            dict: Containing cordinates and size of the box to be drawn on screen.
        '''
        return {"left": area[0], "top": area[1], "width": area[2]-area[0], "height":area[3]-area[1]} 
    
    def accuracy(self, location=None):
        '''
        Used to test the model. Similar to train function. Might change.
        This would be called testing however this clashes with existing nn.Modules methods.
        
        Parameter:
            location (String): The location of the folder being used.
            
        Return:
            float: The overall loss of the model during testing.
        '''
        loader, _ = self.grabData(location, True, True)
        self.eval()
        runningloss = 0
        count = 0
        for image, cords, button in loader:
            y_pred = self(image)
            loss = self.criterion(y_pred, cords)
            runningloss += loss.item()
            for y in range(len(y_pred)):
                img, box = self.denormalise(image[y], y_pred[y])
                self.rules.checkRules(y_pred)
                img=draw_bounding_boxes(img, box, width=2, colors=(255,0,0))
                img = img.permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(img)  # Display the numpy array directly
                plt.axis('off')
                plt.show()
        del loader, image, cords, button, y_pred, loss
        torch.cuda.empty_cache()
        print(self.rules.accuracy())
        return runningloss
    
    def interactWithScreen(self, cords, button):
        '''
        This method is responsible for the models ability to interact with the screen.
        
        Parameters:
            cords ([int,int]): Contains the cordiantes of the mouse to be moved to.
            button (String): Contains the letter or mouse click to be used.
        '''
        mouse.move(cords[0], cords[1])
        if button == "Lclick":
            mouse.click()
        if button == "Rclick":
            mouse.right_click()
    
    def updateLR(self, index):
        '''
        Updates the learningrate for dynamic training [Not yet used]
        
        Parameters:
            index (int): The index of the requested training rate in varyingLearningRate
        '''
        self.optimizer = optim.Adam(self.parameters(), lr = self.learningRate)
    
    def liveTest(self):
        '''
        This method is used in live testing. The original cordinates used in 
        the dataCollection.py process with be collected and a box is drawn on 
        screen. The user is given a chance to setup before the model will begin 
        collecting images and predicting what action should be taken.
        '''
        imageArea = self.setBoxArea(self.area)
        box = boxDrawer.ScreenDraw(self.area)
        input("Place content in red zone and hit enter.")
        timestop = 0
        while timestop < 10:
            with mss() as sct:
                timestop+=1
                image = np.array(sct.grab(imageArea))[:,:,:3] #Repeat code. find a way to fix it
                image = self.preprocessImage(image)
                image = image.unsqueeze(0)
                width, height = image.shape[1], image.shape[2]
            cords = self(image)
            cords[0][0],cords[0][1] = cords[0][1]+imageArea.get("left"), cords[0][1]+imageArea.get("top")
            self.interactWithScreen(cords[0], "Lclick")
            time.sleep(3)
        box.end()

model = CNNModel()

print(model.learn(50, "keyboard"))
print(model.accuracy("keyboard"))