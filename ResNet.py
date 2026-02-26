#Importing the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
#Device(GPU/CPU)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transforms
transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),
                                                                             (0.2023,0.1994,0.2010))])
transform_test=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),
                                                                               (0.2023,0.1994,0.2010))])
#Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data',download=True,train=True,transform=transform_train)
test_dataset=torchvision.datasets.CIFAR10(root='./data',download=True,train=False,transform=transform_test)
#Dataloader
train_dataloader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=64)
test_dataloader=DataLoader(dataset=test_dataset,shuffle=False,batch_size=64)
#Neuralnetwork
class Residualblock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Residualblock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm2d(out_channels)
                                        )
    def forward(self,x):
        identity=self.shortcut(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out += identity
        out=self.relu(out)
        return out
#Minireset model
class MiniResetNet(nn.Module):
    def __init__(self,num_classes=10):
        super(MiniResetNet,self).__init__()
        self.initial=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True)
                                   )
        self.layer1=self._make_layer(64,64,stride=1)
        self.layer2=self._make_layer(64,128,stride=2)
        self.layer3=self._make_layer(128,256,stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256,num_classes)
    def _make_layer(self,in_channels,out_channels,stride):
        return nn.Sequential(Residualblock(in_channels,out_channels,stride),
                             Residualblock(out_channels,out_channels)
                             )
    def forward(self,x):
        x=self.initial(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
#Model
model=MiniResetNet().to(device)
#Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
epochs=60
#Loop
for epoch in range(epochs):
    model.train()
    total=0
    correct=0
    running_loss=0
    for images,labels in train_dataloader:
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
        correct+=predicted.eq(labels).sum().item()
    scheduler.step()
    train_acc=100*correct/total
    print(f"Epoch[{epoch+1}/{epochs}], "
          f"Running loss:{running_loss/len(train_dataloader):.4f}, "
          f"Train Accuracy: {train_acc}%]")
#Evalution
model.eval()
total=0
correct=0
with torch.no_grad():
    for images,labels in test_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()
test_acc=100*correct/total
print(f"Test Accuracy: {test_acc:.2f}%")
