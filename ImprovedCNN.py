#Importing the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
#Device(GPU/CPU)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transform
transform = transforms.ToTensor()
#Dataset
train_dataset=torchvision.datasets.MNIST(root='./data',download=True,train=True,transform=transform)
test_dataset=torchvision.datasets.MNIST(root='./data',download=True,train=False,transform=transform)
#Dataloader
train_dataloader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=64)
test_dataloader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=64)
#Neuralnetwork
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
#Model
model=ImprovedCNN().to(device)
#Optimizer
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=5
#Loop
for epoch in range(epochs):
    model.train()
    running_loss=0
    total_loss=0
    correct=0
    for images, labels in train_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total_loss+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Epoch[{epoch + 1}/{epochs}]"
          f"Loss: {running_loss / len(train_dataloader):.4f}"
          f"Accuracy: {100 * correct / total_loss:.2f}%")
#Evaluation
total_loss=0
correct=0
with torch.no_grad():
    for images, labels in test_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total_loss+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Accuracy: {100*correct/total_loss:.2f}%")