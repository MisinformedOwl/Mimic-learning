import torch                            #Torch is the framework we will be using to create this neural network.
import torch.nn.functional as F         #Activation functions
from torch import optim, nn             #Imports for optimsers and neural network layers such as linear.
import graphviz                         # Imported as it's needed to be used with torchviz apparently.
from torchviz import make_dot as viz    #Library used to create an image of the current neural network.

'''
CNNModel class will hold the CNN.
We call nn.Module to make use of torch's pre built features and backprop.
'''
class CNNModel(nn.Module):
    
    training = True
    history = []
    criterion = nn.MSELoss()
    def __init__(self):
        super().__init__()
        self.optimizer = optim.Adam(self.parameters(), lr = 0.01, momentum=0.9) #It is unable to see parameters therefore must be established here.
        
        '''
        We begin with 512 as it's not too small to not capture detail. However 
        it is also not big enough to cause issues in memory.
        '''
        self.layer1 = nn.Conv2d(1, 3, kernel_size=3) # Collecting image of [512,512,1] and colecting featuresto create a matrix of [512,512,3]
        self.layer2 = nn.MaxPool2d(2,2)              #Maxpool halves the size of the image. Image is now [256,256,3]
        self.layer3 = nn.Conv2d(3, 3, kernel_size=3) #[256,256,3]->[256,256,3]
        self.layer4 = nn.MaxPool2d(2,2)              #[256,256,3]->[128,128,3]
        self.layer5 = nn.Conv2d(3, 1, kernel_size=3) #[128,128,3]->[128,128,1]
        
        self.flatten = nn.Flatten()                  #128 x 128 x 1 = 
        
        self.lin1 = nn.Linear(15376, 256)            #124x124x1 = 15,376
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)               #Gradually reduce neurons until we reach 2, the cordinates.
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 16)
        self.lin6 = nn.Linear(16, 8)
        self.lin7 = nn.Linear(8, 2)
        
    '''
    This function is what is activated when actually training/testing the AI.
    This is a useful tool in torch, as it allows me to intercept hidden layer
    and add extra functionality, like checking the shape of the data or even
    add rule based learning which i am planning on doing.
    '''
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = F.relu(self.layer5(x))
        
        x = self.flatten(x)
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))
        x = F.relu(self.lin7(x))
        
        return x
    
    '''
    This method is used to create and render an image of the current neural 
    network should the user desire.
    '''
    def displayCNN(self, x):
        viz(x, params=dict(list(self.named_parameters()))).render("cnn_viz", format="png")
        
    def train(self, trainloader, epochs):
        self.switchTrainingMode(True)
        for epoch in epochs:
            runningloss = 0
            for image, labels in trainloader:
                self.optimizer.zero_grad
                y_pred = self(image)
                loss = self.criterion(y_pred, labels)
                self.optimizer.step()
                runningloss += loss.item()
        runningloss/=epochs
        return runningloss
    
    def switchTrainingMode(self, setting):
        if setting == True:
            self.training = True
            self.train()
        else:
            self.training = False
            self.eval()
    
    def isTraining(self):
        return self.training
    
    def test(self, testloader):
        self.switchTrainingMode(False)
        runningloss = 0
        for image, labels in testloader:
            self.optimizer.zero_grad
            y_pred = self(image)
            loss = self.criterion(y_pred, labels)
            self.optimizer.step()
            runningloss += loss.item()
        return runningloss
    
model = CNNModel()
t = torch.rand(512,512,1)
t = t.permute(2,0,1)
y = model(t)

viz(y, params=dict(model.named_parameters())).render("cnn_viz", format="png")