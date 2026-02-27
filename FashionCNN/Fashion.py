import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import torch.nn as nn


transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,),(0.5,))
])
trainin_dataset=datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=transform
  )

# print(trainin_dataset.data.shape)

test_dataset=datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=transform
  )



train_loader=DataLoader(trainin_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)



class FashionNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(1,32,3)
    self.pool=nn.MaxPool2d(2,2)
    self.conv2=nn.Conv2d(32,64,3)
    
    self.fc1=nn.Linear(64*5*5,128)
    self.fc2=nn.Linear(128,10)

  def forward(self,x):
    x=self.pool(torch.relu(self.conv1(x)))
    x=self.pool(torch.relu(self.conv2(x)))
    x=x.view(-1,64*5*5)
    x=torch.relu(self.fc1(x))
    x=self.fc2(x)
    return x
  
model=FashionNN()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)















