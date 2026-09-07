#Importing the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
#Device(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),
                                                                             (0.2023,0.1994,0.2010))])
#Dataset
train_dataset=torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset=torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform,download=True)
#Dataloader
train_dataloader=DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader=DataLoader(test_dataset, batch_size=64, shuffle=False)
#Neural network
class CIFARNN(nn.Module):
    def __init__(self):
        super(CIFARNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
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
            nn.Dropout2d(0.25),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#Model
model=CIFARNN().to(device)
#Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 55
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
model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total_loss+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Accuracy: {100*correct/total_loss:.2f}%")
