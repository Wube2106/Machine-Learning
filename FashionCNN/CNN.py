import torch
from torch.utils.data import DataLoader,random_split
from torchvision import transforms,datasets
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,),(0.5,))
])

total_train_dataset=datasets.MNIST(
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
val_size=10000
train_dataset,val_dataset=random_split(total_train_dataset,[train_size,val_size])

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=64,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(1,32,3)
    self.conv2=nn.Conv2d(32,64,3)

    self.pool=nn.MaxPool2d(2,2)

    self.fc1=nn.Linear(64*5*5,128)
    self.fc2=nn.Linear(128,10)
  
  def forward(self,x):
    x=self.pool(torch.relu(self.conv1(x)))
    x=self.pool(torch.relu(self.conv2(x)))

    x=torch.flatten(x,1)
    x=torch.relu(self.fc1(x))
    x=self.fc2(x)
    return x
  
model=CNN()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

def train(model,loader):
  running_loss=0.0
  correct=0
  total=0
  for images,labels in loader:
    output=model(images)
    loss=criterion(output,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()
    _,predicted=torch.max(output,1)
    correct+=(predicted==labels).sum().item()
    total+=labels.size(0)

  accuracy=(100*correct)/total
  return running_loss/len(loader),accuracy

predictions=[]
true_labels=[]
def eval(model,loader):
  running_loss=0.0
  correct=0
  total=0
  for images,labels in loader:
    output=model(images)
    loss=criterion(output,labels)
    
    running_loss+=loss.item()
    _,predicted=torch.max(output,1)
    correct+=(predicted==labels).sum().item()
    total+=labels.size(0)
    predictions.extend(predicted.tolist())
    true_labels.extend(labels.tolist())
  accuracy=(100*correct)/total
  eval
  return running_loss/len(loader),accuracy

train_losses=[]
train_accuracys=[]
test_losses=[]
test_accuracys=[]

for epoch in range(10):
  train_loss,train_accuracy=train(model,train_loader)
  eval_loss,eval_accuracy=eval(model,val_loader)

  train_losses.append(train_loss)
  train_accuracys.append(train_accuracy)


  # print(f"epoch:{epoch+1}")
  # print(f"train_accuracy:{train_accuracy}")
  # print(f"train_loss:{train_loss}")
  # print(f"eval_accuracy:{eval_accuracy}")
  # print(f"eval_loss:{eval_loss}")

  test_loss,test_accuracy=eval(model,test_loader)
  test_losses.append(test_loss)
  test_accuracys.append(test_accuracy)
  # print(f"test_accuracy:{test_accuracy}")

cm=confusion_matrix(true_labels,predictions)
# print(cm)
# plt.plot(train_losses,label="Training_loss")
# plt.plot(test_losses,label="Test_loss")
plt.plot(train_accuracys,label="Training_accuracy")
plt.plot(test_accuracys,label="Test_accuracy")
plt.xlabel("Epochs")
plt.xlabel("Losses")
plt.legend()
plt.show()


