import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training set shape:", x_train.shape)  # (60000, 28, 28)
print("Test set shape:", x_test.shape)      # (10000, 28, 28)
print("One image shape:", x_train[0].shape) # (28, 28)

print("FLATTENING...")
x_train = torch.tensor(x_train).view(60000, -1).float()/255.0
y_train = torch.tensor(y_train).view(-1)
x_test = torch.tensor(x_test).view(10000, -1).float()/255.0
y_test = torch.tensor(y_test).view(-1)


# MODEL

class MLP(nn.Module):
    def __init__(self, layer_2 = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    
n_samples, n_features = x_train.shape # 60000, 28*28
model = MLP()


# LOSS

LEARNING_RATE = 0.02
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)

#TRAINING LOOP
NUM_ITERATIONS = 40000
PRINT_FREQ = 500

for epoch in range(NUM_ITERATIONS):
    
    y_predicted = model(x_train)
    loss = loss_func(y_predicted, y_train)
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % PRINT_FREQ == 0:
        print(f'epoch {epoch+1}: loss = {loss.item(): .4f}')
        
# TRIAL
correct = 0

with torch.no_grad():
    
    y_test_pred = model(x_test)
    
    for pair in zip(y_test_pred, y_test):
        if torch.argmax(pair[0]) == pair[1]:
            correct += 1
            
print(f'Accuracy: {correct / 10000 * 100:.2f}%')

torch.save(model.state_dict(), "mnist_mlp.pth")

        






