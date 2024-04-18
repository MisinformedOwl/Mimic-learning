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

'''
CNNModel class will hold the CNN.
We call nn.Module to make use of torch's pre built features and backprop.
'''
class CNNModel(nn.Module):
    history = []
    criterion = nn.MSELoss()
    batchsize = 32
    learningRate = 0.001
    testsize = 20 # in %
    def __init__(self):
        super().__init__()
        '''
        We begin with 512 as it's not too small to not capture detail. However 
        it is also not big enough to cause issues in memory.
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.to(self.device)
        
        self.layer1 = nn.Conv2d(3, 10, kernel_size=3) # Collecting image of [512,512,3] and colecting featuresto create a matrix of [512,512,10]
        self.layer2 = nn.MaxPool2d(2,2)              #Maxpool halves the size of the image. Image is now [256,256,10]
        self.layer3 = nn.Conv2d(10, 20, kernel_size=3) #[256,256,10]->[256,256,20]
        self.layer4 = nn.MaxPool2d(2,2)              #[256,256,3]->[128,128,3]
        self.layer5 = nn.Conv2d(20, 40, kernel_size=3) #[128,128,20]->[128,128,20]
        self.layer6 = nn.MaxPool2d(2,2)              #[128,128,20]->[64,64,20]
        self.layer7 = nn.Conv2d(40, 40, kernel_size=3) #[64,64,20]->[64,64,10]
        self.layer8 = nn.MaxPool2d(2,2)              #[64,64,10]->[32,32,10]
        self.layer9 = nn.Conv2d(40, 20, kernel_size=3) #[32,32,10]->[32,32,10]
        
        self.norm = nn.BatchNorm2d(20)
        
        self.flatten = nn.Flatten()
        
        self.lin1 = nn.Linear(15680, 2048)            #124x124x1 = 15,376
        self.drop1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(2048, 256)
        self.drop2 = nn.Dropout(0.5)
        self.lin3 = nn.Linear(256, 32)               #Gradually reduce neurons until we reach 2, the cordinates.
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, 8)
        self.lin6 = nn.Linear(8, 4)
        self.lin7 = nn.Linear(4, 2)
        
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
        
        self.optimizer = optim.Adam(self.parameters(), lr = self.learningRate) #It is unable to see parameters therefore must be established here.
        
    '''
    This function is what is activated when actually training/testing the AI.
    This is a useful tool in torch, as it allows me to intercept hidden layer
    and add extra functionality, like checking the shape of the data or even
    add rule based learning which i am planning on doing.
    '''
    def forward(self, x):
        s = []
        x = F.sigmoid(self.layer1(x))
        x = self.layer2(x)
        x = F.sigmoid(self.layer3(x))
        x = self.layer4(x)
        x = F.sigmoid(self.layer5(x))
        x = self.layer6(x)
        x = F.sigmoid(self.layer7(x))
        x = self.layer8(x)
        x = F.sigmoid(self.layer9(x))
        
        x = self.norm(x)
        
        x = self.flatten(x)
        
        x = F.sigmoid(self.lin1(x))
        x = self.drop1(x)
        x = F.sigmoid(self.lin2(x))
        x = self.drop2(x)
        x = F.sigmoid(self.lin3(x))
        x = F.sigmoid(self.lin4(x))
        x = F.sigmoid(self.lin5(x))
        x = F.sigmoid(self.lin6(x))
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
    def grabData(self, datafolder, resizeData=True, test = False):
        if datafolder == None:
            name = input("Name of the application being used: ")
            self.location = f"data/{name}"
        else:
            self.location = f"data/{datafolder}"
        if not os.path.isdir(self.location):
            raise FileNotFoundError
        
        mn = 0
        mx = 255
        images = glob.glob(f"{self.location}/*.png")
        
        
        #Grabbing user inputs and locations.
        with open(f"{self.location}/inputs.pkl", "rb") as file:
            inputs = pickle.load(file)
        
        initialSize = len(inputs)
        testIndex = round(len(inputs)/100)*self.testsize
        if test:
            inputs = inputs[:testIndex+2]
            images = images[:testIndex+2]
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
                assert (image > 2).sum() == 0
                data.append([torch.tensor(np.transpose(resize(image, (512, 512), INTER_AREA), (2,0,1)), dtype=torch.float).to(self.device), torch.tensor(i[0]).to(self.device), i[1]])
        else:
            for image, i in zip(images, inputs):
                data.append([torch.tensor(np.transpose(imread(image), (2,0,1)), dtype=torch.float).to(self.device), torch.tensor(i[0]).to(self.device), i[1]])
            
        return DataLoader(data, batch_size=self.batchsize, shuffle=False)

    '''
    Train function used for training the model.
    
    Main difference is the model is set to train mode.
    '''
    def learn(self, epochs, location=None):
        trainloader = self.grabData(location, True, False)
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
    
    def denormalise(self, image, cord):
        image= (image*255).to(torch.uint8)
        width = image.shape[1]
        height = image.shape[2]
        cord = torch.tensor([[cord[0]*width-10, cord[1]*height-10, cord[0]*width+10, cord[1]*height+10]])
        return image, cord
    
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
    def accuracy(self, location=None):
        loader = self.grabData(location, True, True)
        self.eval()
        runningloss = 0
        count = 0
        for image, cords, button in loader:
            y_pred = self(image)
            loss = self.criterion(y_pred, cords)
            runningloss += loss.item()
            for y in range(len(y_pred)):
                img, box = self.denormalise(image[y], cords[y])
                img=draw_bounding_boxes(img, box, width=2, colors=(255,0,0))
                img = img.permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(img)  # Display the numpy array directly
                plt.axis('off')
                plt.show()
        return runningloss