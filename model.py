from torchvision.models.resnet import BasicBlock, ResNet18_Weights, ResNet
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as smry

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(27*27*128, 500),
            nn.BatchNorm1d(500),
            nn.Linear(500, 10)
        )

        self.residual1 = nn.Conv2d(3, 32, 1, 1, 0, bias=False)
        self.residual2 = nn.Conv2d(32, 64, 1, 1, 0, bias=False)
        self.residual3 = nn.Conv2d(64, 128, 1, 1, 0, bias=False)
    
    def forward(self, x):
        _, _, H, W = x.shape
        feature1 = F.max_pool2d(self.conv1(x)+self.residual1(x)[:,:,1:H-1,1:W-1], kernel_size=2, stride=1)
        feature2 = F.max_pool2d(self.conv2(feature1)+self.residual2(feature1), kernel_size=2, stride=1)
        feature3 = F.max_pool2d(self.conv3(feature2)+self.residual3(feature2), kernel_size=2, stride=1)
        flat = feature3.view(-1, 27*27*128)
        out = self.classifier(flat)
        
        return out

def get_teacher() -> nn.Module:
    # Get model information
    weights = ResNet18_Weights.verify(ResNet18_Weights.DEFAULT)
    model = ResNet(BasicBlock, [2, 2, 2, 2])

    # Modify some layers...
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    # Load pre-trained weights
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))

    # Last layer --> the number of class is 10
    model.fc = nn.Linear(512, 10, bias=True)

    # Summarize model information
    smry(model, (3, 32, 32), device='cpu')
    return model

def get_student() -> nn.Module:
    model = StudentNet()
    # Summarize model information
    smry(model, (3, 32, 32), device='cpu')
    return model