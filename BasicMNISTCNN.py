#Importing the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#Device(GPU/CPU)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transform
transform = transforms.ToTensor()
#Dataloader
train_dataset = torchvision.datasets.MNIST(root='./data',download=True,train=True,transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data',download=True,train=False,transform=transform)
#Dataset
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

#Neural Network
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*7*7,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=self.pool(x)
        x=torch.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x
#Model
model=BasicCNN().to(device)
#Optimizer
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=5
#For loop
for epoch in range(epochs):
    model.train()
    running_loss=0
    correct=0
    total=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Epoch[{epoch+1}/{epochs}]"
         f"Loss: {running_loss/len(train_loader):.4f}"
         f"Accuracy: {100*correct/total:.2f}%")
#Evaluation
total=0
correct=0
with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Accuracy: {100*correct/total:.2f}%")