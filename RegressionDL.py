#Importing the modules
import torch
import torch.nn as nn
import torch.optim as optim
#Giving the values
x=torch.linspace(-10,10,100).view(-1,1)
y=3*x+2
#Neural network model
class RegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,16)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(16,8)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(8,1)
        self.relu3=nn.ReLU()

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        return x
#Model initializing
model=RegressionNN()
optimizer=optim.Adam(model.parameters())
#Training loop
epochs=500
for epoch in range(epochs):
    optimizer.zero_grad()
    output=model(x)
    loss=nn.MSELoss()(output,y)
    loss.backward()
    optimizer.step()
    if epoch%50==0:
        print(f"epoch:{epoch},loss:{loss.item():.4f}")

#Prediction
test_input=torch.tensor([[5.0]])
pred=model(test_input)
print("\nOutput for the input of 5 is:",pred.item())