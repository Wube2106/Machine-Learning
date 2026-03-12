import torch
from torch.utils.data import random_split,DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,),(0.5,))
])

full_train_dataset=datasets.MNIST(
  root="/data",
  train=True,
  download=True,
  transform=transform
)

test_dataset=datasets.MNIST(
  root="/data",
  train=False,
  download=True,
  transform=transform
)

train_size=50000
eval_size=10000
train_dataset,eval_dataset=random_split(full_train_dataset,[train_size,eval_size])

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
eval_loader=DataLoader(eval_dataset,batch_size=64,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

class FullyConnectedNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(784,64)
    self.fc2=nn.Linear(64,32)
    self.fc3=nn.Linear(32,10)

  def forward(self,x):
    x=torch.flatten(x,1)
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=self.fc3(x)
    return x
model=FullyConnectedNetwork()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

def train(model,loader):
  model.train()
  running_loss=0.0
  correct=0
  total=0
  for images,labels in loader:
    output=model(images)
    loss=criterion(output,labels)
    running_loss+=loss.item()

    _,predicted=torch.max(output,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum().item()
  accuracy=(correct*100)/total
  return running_loss/len(loader),accuracy

def evaluation(model,loader):
  
  model.eval()
  running_loss=0.0
  correct=0.0
  total=0

  with torch.no_grad():
    for images,labels in loader:
      output=model(images)
      loss=criterion(output,labels)
     
      running_loss+=loss.item()
      _,predicted=torch.max(output,1)
      correct+=(predicted==labels).sum().item()
      total+=labels.size(0)
  accuracy=(100*correct)/total
  return running_loss/len(loader),accuracy


for epoch in range(10):
  train_losses=[]
  train_accuracys=[]
  train_loss,train_accuracy=train(model,train_loader)
  eval_loss,eval_accuracy=evaluation(model,eval_loader)
  print(f"Epoch:{epoch+1}")
        
  
  print(f"train loss:{train_loss:.4f} | train accuracy:{train_accuracy:.2f}%")
  print(f"eval loss:{eval_loss:.4f} | eval accuracy:{eval_accuracy:.2f}%")
  
  test_loss,test_accuracy=evaluation(model,test_loader)
  print(f"test accuracy:{test_accuracy:.2f}%")

  
  
  
  

  

