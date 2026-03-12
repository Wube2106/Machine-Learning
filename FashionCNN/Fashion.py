import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
    self.bn1=nn.BatchNorm2d(32)
    
    self.conv2=nn.Conv2d(32,64,3)
    self.bn2=nn.BatchNorm2d(64)

    self.pool=nn.MaxPool2d(2,2)
    
    self.fc1=nn.Linear(64*5*5,128)
    self.dropout=nn.Dropout(0.5)

    self.fc2=nn.Linear(128,10)

  def forward(self,x):
    x=self.pool(torch.relu(self.bn1(self.conv1(x))))
    x=self.pool(torch.relu(self.bn2(self.conv2(x))))
    x=torch.flatten(x,1)
    x=torch.relu(self.fc1(x))
    x=self.dropout(x)
    x=self.fc2(x)
    return x
  
model=FashionNN()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

train_losses=[]
eval_losses=[]
for epoch in range(10):
  model.train()
  running_loss=0.0
  for images,labels in train_loader:
    optimizer.zero_grad()
    outputs=model(images)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()
  train_losses.append(running_loss/len(train_loader))
  print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader):.4f}")

  correct=0
  total=0
  model.eval()
  predictions=[]
  true_labels=[]
  val_loss=0.0
  with torch.no_grad():
    for images,labels in test_loader:
      outputs=model(images)
      loss=criterion(outputs,labels)
      val_loss+=loss.item()
      _,predicted=torch.max(outputs,1)
      total+=labels.size(0)
      predictions.extend(predicted.tolist())
      true_labels.extend(labels.tolist())
      correct+=(predicted==labels).sum().item()
  eval_losses.append(val_loss/len(test_loader))
  print(f"Accuracy: {100*correct/total}%")

cm=confusion_matrix(true_labels,predictions)
print(cm)

plt.plot(train_losses,label="Training loss")
plt.plot(eval_losses,label="Evaluation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
  

  











