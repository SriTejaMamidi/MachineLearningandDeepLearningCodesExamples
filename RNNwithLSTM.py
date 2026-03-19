#Importing the required modules
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
class LSTMRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTMRNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_=self.lstm(x,(h0,c0))
        out=out[:,-1,:]
        out=self.fc(out)
        return out
#Model
model=LSTMRNN(input_size,hidden_size,num_layers,num_classes)
#Loss and Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
#Loop
for epoch in range(epochs):
    for inputs,labels in loader:
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch:{epoch+1},loss:{loss.item():.4f}")
with torch.no_grad():
    test_input=torch.randn(1,sequence_length,input_size)
    prediction=model(test_input)
    predicted_class=torch.argmax(prediction,dim=1)
    print("Predicted class:",predicted_class.item())