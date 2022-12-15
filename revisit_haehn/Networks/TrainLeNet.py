import torch.nn as nn
import torch
from . import NetUtils
from util.Config import obj2dic
from util.Config import ConfigObj
from . import Trainer
import logging

#Defining the convolutional neural network
class LeNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(LeNet, self).__init__()
        self.num_classes=num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(18496, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class TrainLeNet(Trainer.Trainer):

    def _createNetwork(self):
        name = self.param.name
        names=['lenet','LeNet']
        if name not in names:
            logging.error("Unresolved res net type %s"%name)
        num_classes = self.param.num_classes
        
        self.network = LeNet(num_classes=num_classes)
        self.lossfunc = nn.MSELoss()
        # logging.info("Use network %s"%name)
        # num_classes = self.param.num_classes
        # method = getattr(LeNet,name)
        # self.network = method(self.param.pretrained,num_classes=num_classes)
        # self.lossfunc = nn.MSELoss()

    def __init__(self,param):
        ConfigObj.default(param,"pretrained",False)
        super(TrainLeNet,self).__init__(param)