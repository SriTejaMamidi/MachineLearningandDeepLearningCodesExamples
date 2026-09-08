#Importing the modules
import torch
import torch.nn as nn
#Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1,32)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=2,batch_first=True),
            num_layers=1
        )
        self.fc=nn.Linear(32,1)
    def forward(self,x):
        x=self.embedding(x)
        x=self.transformer(x)
        x=x[:,-1,:]
        x=self.fc(x)
        return x
#Dummy Data
x=torch.randn(10,5,1)
y=torch.randn(10,1)
#Model
model=SimpleTransformer()
#Loss and optimizer
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=50
for epoch in range(epochs):
    output=model(x)
    loss=criterion(output,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")
