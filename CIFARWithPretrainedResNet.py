#Importing the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import  DataLoader
from torchvision import models
#Device(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transform
transform_train=T.Compose(
    [T.Resize((224,224)),T.RandomHorizontalFlip(),T.ToTensor(),
     T.Normalize((0.485,0.456,0.406),
     (0.229,0.224,0.225))]
)
transform_test=T.Compose(
    [T.Resize((224,224)),T.ToTensor(),
     T.Normalize((0.485,0.456,0.406),
     (0.229,0.224,0.225))]
)
#Dataset
train_dataset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_train)
test_dataset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_test)
#Dataloader
train_dataloader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=32)
test_dataloader=DataLoader(dataset=test_dataset,shuffle=False,batch_size=32)
#Load pretrained model
model=models.resnet18(pretrained=True)
num_features=model.fc.in_features
model.fc = nn.Linear(num_features,10)
#Freeze backbone
for param in model.parameters():
    param.requires_grad=False
#Unfreeze classifier
for param in model.fc.parameters():
    param.requires_grad=True
model=model.to(device)
#Optimizer
criterion = nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=0.001)
epochs=5
#Loop
for epoch in range(epochs):
    model.train()
    correct=0
    running_loss=0
    total=0
    for images,labels in train_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=outputs.max(1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()
    print(f"Epoch{epoch+1},Loss:{running_loss/len(train_dataloader):.4f},"
          f"Train Accuracy:{100*correct/total:.2f}%")
#Evaluation
correct=0
total=0
with torch.no_grad():
    model.eval()
    for images,labels in test_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=outputs.max(1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()
print(f"Test Accuracy: {100*correct/total:.2f}%")