#Imprting the modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#Device(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transform
transform = transforms.ToTensor()
#Dataloader
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True,
                                           batch_size=64)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          shuffle=False,
                                          batch_size=64)
#Neuralnetworkmodel
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x


model=NeuralNet().to(device)
#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss {loss.item():.4f}")

#Evaluation
model.eval()
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

accuracy=100*correct/total
print(f"Test accuracy: {accuracy:.2f}%")
#SaveModel
torch.save(model.state_dict(),"mnist_model.pth")