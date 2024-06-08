
import torch
import torchvision
from torch import nn

def create_model(num_classes:int=100,
                 seed:int=42):
  Weights=torchvision.models.ResNet50_Weights.DEFAULT
  transforms=Weights.transforms()
  model=torchvision.models.resnet50(weights=Weights)


  for param in model.parameters():
    param.require_grad=False

    torch.manual_seed(42)
    model.fc=nn.Sequential(
    torch.nn.Linear(in_features=2048,
                    out_features=1000),
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=1000,
                    out_features=500),
    torch.nn.Dropout(),
    nn.ReLU(),

    torch.nn.Linear(in_features=500,
                    out_features=num_classes, # same number of output units as our number of classes
                    bias=True))

    return model,transforms
