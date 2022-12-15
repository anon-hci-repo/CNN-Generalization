import torch.nn as nn
import torch
from . import NetUtils
from util.Config import obj2dic
from util.Config import ConfigObj
from . import Trainer
import torchvision.models.alexnet as alexnet
import logging

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TrainAlexNet(Trainer.Trainer):

    def _createNetwork(self):
        name = self.param.name
        names=['alexnet','AlexNet']
        if name not in names:
            logging.error("Unresolved res net type %s"%name)
        logging.info("Use network %s"%name)
        num_classes = self.param.num_classes
        self.network = AlexNet(num_classes=num_classes)
        self.lossfunc = nn.MSELoss()
        # method = getattr(alexnet,name)
        
        # self.network = method(self.param.pretrained,num_classes=num_classes)
        # self.lossfunc = nn.MSELoss()

    def __init__(self,param):
        ConfigObj.default(param,"pretrained",False)
        super(TrainAlexNet,self).__init__(param)
        