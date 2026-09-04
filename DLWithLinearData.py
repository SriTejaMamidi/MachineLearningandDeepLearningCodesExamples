#Importing the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils
torch.manual_seed(42)
distances=torch.tensor([[1.0],[2.0],[3.0],[4.0]],dtype=torch.float32)
times=torch.tensor([[6.96],[12.11],[16.77],[22.21]],dtype=torch.float32)
model=nn.Sequential(nn.Linear(1,1))
lossfunction=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.001)
for epoch in range(500):
    optimizer.zero_grad()
    output=model(distances)
    loss=lossfunction(output,times)
    loss.backward()
    optimizer.step()
    if(epoch+1)%50==0:
        print(f"Epoch {epoch+1}:Loss= {loss.item()}")
helper_utils.plot_data(distances,times)
distance_to_predict=7.0
with torch.no_grad():
    new_distance=torch.sensor([[distance_to_predict]],dtype=torch.float32)
    prediction=model(new_distance)
    print(f"Prediction for a {distance_to_predict} mile delivery is  {prediction.item():1f}minutes")
    if prediction.item()>30:
        print("Donot take the job u will not reach the destination in time")
    else:
        print("Will do it")
