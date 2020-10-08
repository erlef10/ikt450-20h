import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(7)
np.random.seed(7)

def get_accuracy(predictions, labels):
    correct = 0

    for i, prediction in enumerate(predictions):
        if labels[i] == round(prediction.item()):
            correct += 1
    
    return correct / len(predictions)

def step(data):
    inputs = data[0:-1]
    labels = data[-1]

    outputs = model(inputs)
    prediction = torch.max(outputs.data)

    return prediction

def predict(dataloader):
    predictions = []

    for i, batch in enumerate(dataloader):
        pred = step(batch)
        predictions.append(pred)
    
    return predictions

# import dataset
dataset = np.loadtxt("ecoli.data", delimiter="\t", dtype=str)

# split the dataset
dataset = [row.split() for row in dataset]

# Set 'im' as 1 and 'cp' as 0
dataset = [row[:-1]+[0 if row[-1] == 'cp' else 1] for row in dataset]

# remove the first column as they are all unique anyhow
dataset = [row[1::] for row in dataset]

# convert the strings to floats
dataset = [[float(x) for x in row] for row in dataset]

# do the shuffle
np.random.shuffle(dataset)

# split the dataset into training and validation datasets
splitratio = 0.8

training_dataset = dataset[:int(len(dataset)*splitratio)]
validation_dataset = dataset[int(len(dataset)*splitratio):]

# Tranform the numpy arrays to Torch Tensors
X = torch.Tensor([i[0:7] for i in training_dataset])
Y = torch.Tensor([i[7] for i in training_dataset])

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(7,3)
        self.fc2 = nn.Linear(3,1)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = Net()
model.train()

# Binary Cross Entropy loss function
criterion = nn.BCELoss()

# Adaptive Moment Estimation optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
accuracies = []

for epoch in range(1000):
    # predict label
    outputs = model(X)

    # compare prediction to actual value
    loss = criterion(outputs, Y)

    # backpropagate through the graph to determine
    # which weights are causing the losses
    loss.backward()
    losses.append(loss.item())

    # step forward in time (epoch)
    optimizer.step()

    accuracy = get_accuracy(outputs, Y)
    accuracies.append(accuracy)

# set model to evaluation after we are done training
model.eval()

# Setup validation data as Torch Tensors
validation = torch.Tensor([i[0:8] for i in validation_dataset])

predictions = predict(validation)
print("Accuracy:", get_accuracy(predictions, [x[-1] for x in validation]))

# Plot the accuracy and losses
plt.plot(accuracies)
plt.plot(losses)
plt.show()