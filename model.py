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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
        self.fusionModel = nn.Sequential(
            nn.Linear(256, 128),  
            nn.ReLU()
        )
    def createFCModel(self):
        self.fcModel = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x_visual, x_audio):
        visual_features = self.visualModel(x_visual)
        audio_features = self.audioModel(x_audio)
        fused_features = torch.cat((visual_features, audio_features), dim=1)
        fusion_output = self.fusionModel(fused_features)
        fc_output = self.fcModel(fusion_output)
        return fc_output
    
    def train_network(self, loader, epoch, **kwargs):
      criterion = nn.BCEWithLogitsLoss()
      optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0001)
      self.train()
      for batch_idx, (data, target) in enumerate(loader):
          optimizer.zero_grad()
          data_visual, data_audio = data['visual'], data['audio']
          target = target.float().unsqueeze(1)

          output = self(data_visual, data_audio)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()

          if batch_idx % 10 == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(loader.dataset),
                  100. * batch_idx / len(loader), loss.item()))
    def evaluate_network(self, loader, **kwargs):
        self.eval()
        test_loss, correct = 0, 0
        size = len(loader.dataset)
        num_batches = len(loader)

        with torch.no_grad():
            for data, target in loader:
                data_visual, data_audio = data['visual'], data['audio']
                target = target.float().unsqueeze(1)

                output = self(data_visual, data_audio)
                test_loss += nn.BCEWithLogitsLoss()(output, target).item()
                correct += (output > 0.5).eq(target).sum().item()

        test_loss /= num_batches
        accuracy = correct / size

        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    def saveParameters(self, path):
        torch.save(self.state_dict(), path)
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