import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        super(model, self).__init__()

        self.visualModel = None
        self.audioModel = None
        self.fusionModel = None
        self.fcModel = None

        self.createVisualModel()
        self.createAudioModel()
        self.createFusionModel()
        self.createFCModel()

    def createVisualModel(self):
         self.visualModel = nn.Sequential(
            transforms.Resize((128, 128)),
            nn.ReLU(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
            nn.ReLU(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.flatten()
        )

    def createAudioModel(self):
        self.audioModel = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.flatten()
        )

    def createFusionModel(self):
        pass

    def createFCModel(self):
        pass
    
    def train_network(self, loader, epoch, **kwargs):
        pass

    def evaluate_network(self, loader, **kwargs):
        pass

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)
        
    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)