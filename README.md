# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1057" height="702" alt="image" src="https://github.com/user-attachments/assets/0560e8cd-f695-456b-956d-2e6612864f95" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Rishab P Doshi

### Register Number:212224240134

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

url="/content/data.csv"
data=pd.read_csv(url)

X=torch.tensor(data.iloc[:,0].values,dtype=torch.float32).view(-1,1)
y=torch.tensor(data.iloc[:,1].values,dtype=torch.float32).view(-1,1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)

def train_model(ai_brain,X_train,y_train,criterion,optimizer,epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch%200==0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain,X,y,criterion,optimizer)

plt.plot(ai_brain.history['loss'])
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Iteration")
plt.show()


ai_brain.eval()

new_input=torch.tensor([[7.5]],dtype=torch.float32)

with torch.no_grad():
    prediction=ai_brain(new_input)

print("prediction:",prediction.item())


```

### Dataset Information
<img width="195" height="267" alt="image" src="https://github.com/user-attachments/assets/4360acdd-10ba-4362-bd50-bf92a315e764" />

### OUTPUT
<img width="307" height="200" alt="image" src="https://github.com/user-attachments/assets/bcd9d09f-92b9-4bfc-ad24-8b386b9daee3" />

### Training Loss Vs Iteration Plot

<img width="643" height="502" alt="image" src="https://github.com/user-attachments/assets/41a2cde9-334a-4f83-802a-390ad1f760dc" />

### New Sample Data Prediction

<img width="265" height="20" alt="image" src="https://github.com/user-attachments/assets/fd5aa745-3436-4879-a18b-691058999c7f" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
