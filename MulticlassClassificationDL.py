#Importing the modules
import torch
import torch.nn as nn
import torch.optim as optim
#Getting the data
x=torch.tensor([[0.0],[0.5],[0.8],[1.2],[1.5],[1.8],[2.2],[2.5],[3.0]])
y=torch.tensor([0,0,0,1,1,1,2,2,2])
#Neural network model
class MulticlassClassificationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(8,3)

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

#Model
model=MulticlassClassificationNN()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.05)
#Training Loop
epochs=1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output=model(x)
    loss=criterion(output,y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch} loss {loss.item()}")