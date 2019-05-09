
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

     def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(1, 10, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(10, 20, 3)
        
        # 20 outputs * the 5*5 filtered/pooled map size
        # 10 output channels (for the 10 classes)
        self.fc1 = nn.Linear(58320, 50)
        
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # 10 outputs 
        self.fc2 = nn.Linear(50,136)
        

    # define the feedforward behavior
     def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = F.relu(self.fc1(x))
        
        x = self.fc1_drop(x)
        
        x = self.fc2(x)
        # a softmax layer to convert the 10 outputs into a distribution of class scores
#         x = F.log_softmax(x, dim=1)
        
        # final output
        return x
    
# from paper https://arxiv.org/pdf/1710.00977.pdf
class NaimishNet(nn.Module):

     def __init__(self):
        super(NaimishNet, self).__init__()
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        
        # input 1*224*224
        
        self.conv1 = nn.Conv2d(1, 32, 4)    # output 32x221x221
        self.pool1 = nn.MaxPool2d(4, 4)     # output 32x55x55
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)   # output 64x53x53
        self.pool2 = nn.MaxPool2d(2, 2)     # output 64x26x26
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)  # output 128x25x25
        self.pool3 = nn.MaxPool2d(2, 2)     # output 128x12x12
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 1) # output 256x12x12
        self.pool4 = nn.MaxPool2d(2, 2)     # output 256x6x6
        self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256*6*6,1000) # output 1000
        self.drop_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1000,1000) # output 1000
        self.drop_fc2 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(1000,68*2) # output 136 (68,2)

    # define the feedforward behavior
     def forward(self, x):
               
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool3(F.elu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool4(F.elu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1) 
        
        x = F.elu(self.fc1(x))
        x = self.drop_fc1(x)
        x = self.fc2(x)
        x = self.drop_fc2(x)
        x = self.fc3(x)
        
        
        # final output
        return x