#Import the required modules
import torch
import torch.nn as nn
import torch.optim as optim
#Hyperparameters
input_size=20
hidden_size=50
num_layers=1
num_classes=2
sequence_length=10
batch_size=8
epochs=5
#Dummy dataset
x=torch.randn(100,sequence_length,input_size)
y=torch.randint(0,2,(100,))
#Dataset and loader
dataset=torch.utils.data.TensorDataset(x,y)
loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
#Model
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out,hidden=self.rnn(x)
        out=out[:,-1,:]
        out=self.fc(out)
        return out
#Model
model=RNN()
#Criterion and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
#Training
for epoch in range(epochs):
    for inputs, labels in loader:
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch:{epoch+1},loss:{loss.item():.4f}")