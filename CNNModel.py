import torch                            #Torch is the framework we will be using to create this neural network.
import torchvision
from torchvision import transforms
import torch.nn.functional as F         #Activation functions
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

'''
CNNModel class will hold the CNN.
We call nn.Module to make use of torch's pre built features and backprop.
'''
class CNNModel(nn.Module):
    history = []
    criterion = nn.MSELoss()
    batchsize = 8
    learningRate = 0.01
    def __init__(self):
        super().__init__()
        '''
        We begin with 512 as it's not too small to not capture detail. However 
        it is also not big enough to cause issues in memory.
        '''
        self.layer1 = nn.Conv2d(3, 10, kernel_size=3) # Collecting image of [512,512,3] and colecting featuresto create a matrix of [512,512,10]
        self.layer2 = nn.MaxPool2d(2,2)              #Maxpool halves the size of the image. Image is now [256,256,10]
        self.layer3 = nn.Conv2d(10, 20, kernel_size=3) #[256,256,10]->[256,256,20]
        self.layer4 = nn.MaxPool2d(2,2)              #[256,256,3]->[128,128,3]
        self.layer5 = nn.Conv2d(20, 20, kernel_size=3) #[128,128,20]->[128,128,20]
        self.layer6 = nn.MaxPool2d(2,2)              #[128,128,20]->[64,64,20]
        self.layer7 = nn.Conv2d(20, 20, kernel_size=3) #[64,64,20]->[64,64,10]
        self.layer8 = nn.MaxPool2d(2,2)              #[64,64,10]->[32,32,10]
        self.layer9 = nn.Conv2d(20, 40, kernel_size=3) #[32,32,10]->[32,32,10]
        
        self.flatten = nn.Flatten()                  #128 x 128 x 1 = 
        
        self.lin1 = nn.Linear(31360, 256)            #124x124x1 = 15,376
        self.drop1 = nn.Dropout(0.7)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 32)               #Gradually reduce neurons until we reach 2, the cordinates.
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, 8)
        self.lin6 = nn.Linear(8, 4)
        self.lin7 = nn.Linear(4, 2)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                assert m.weight.sum().item() != 0
        
        self.optimizer = optim.Adam(self.parameters(), lr = self.learningRate) #It is unable to see parameters therefore must be established here.
        
    '''
    This function is what is activated when actually training/testing the AI.
    This is a useful tool in torch, as it allows me to intercept hidden layer
    and add extra functionality, like checking the shape of the data or even
    add rule based learning which i am planning on doing.
    '''
    def forward(self, x):
        s = []
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
        x = F.relu(self.layer7(x))
        x = self.layer8(x)
        x = F.relu(self.layer9(x))
        
        x = self.flatten(x)
        
        x = F.relu(self.lin1(x))
        x = self.drop1(x)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))
        x = self.lin7(x)
        
        return x
    
    '''
    This method is used to create and render an image of the current neural 
    network should the user desire.
    '''
    def displayCNN(self, x):
        viz(x, params=dict(list(self.named_parameters()))).render("cnn_viz", format="png")
    
    '''
    Grabs the data from file and puts it into a dataloader for use in the model.
    '''
    def grabData(self, datafolder, resizeData=True):
        if datafolder == None:
            name = input("Name of the application being used: ")
            self.location = f"data/{name}"
        else:
            self.location = f"data/{datafolder}"
        if not os.path.isdir(self.location):
            raise FileNotFoundError
        #Grabbing images.
        mn = 0 # min of image will always be 0
        mx = 255 # min will always be 255
        images = glob.glob(f"{self.location}/*.png")
        
        
        #Grabbing user inputs and locations.
        with open(f"{self.location}/inputs.pkl", "rb") as file:
            inputs = pickle.load(file)
        
        data = []
        normalizer = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        if resizeData: #If the data is being resized to 512
            for image, i in zip(images, inputs):
                image = imread(image)
                image = image /255
                image = image.round(4)
                assert (image > 2).sum() == 0
                data.append([torch.tensor(np.transpose(resize(image, (512, 512), INTER_AREA), (2,0,1)), dtype=torch.float), torch.tensor(i[0]), i[1]])
        else:
            for image, i in zip(images, inputs):
                data.append([torch.tensor(np.transpose(imread(image), (2,0,1)), dtype=torch.float), torch.tensor(i[0]), i[1]])
            
        return DataLoader(data, batch_size=self.batchsize, shuffle=False)

    '''
    Train function used for training the model.
    
    Main difference is the model is set to train mode.
    
    [Look into encapsulating this for less remaking of code. Is optimizer zero 
    grad required.]
    '''
    def learn(self, epochs, location=None):
        trainloader = self.grabData(location, True)
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
            self.history.append(runningloss)
        return runningloss
    
    '''
    Graphs the model's history
    '''
    def graph(self):
        plt.plot(self.history)

    '''
    Checks to see if the model is currently training.
    No idea what i'm actually going to use this for yet, might get removed. Fels like a good idea at the time.
    '''
    def isTraining(self):
        return self.training
    
    '''
    Used to test the model. Similar to train function. Might change.
    '''
    def accuracy(self):
        loader = self.grabData()
        self.test()
        runningloss = 0
        for image, cords, button in loader:
            self.optimizer.zero_grad()
            y_pred = self(image)
            loss = self.criterion(y_pred, cords)
            self.optimizer.step()
            runningloss += loss.item()
        return runningloss
