import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)

def get_accuracy(predictions, labels):
    correct = 0

    for i, prediction in enumerate(predictions):
        if labels[i] == round(prediction.item()):
            correct += 1
    
    return correct / len(predictions)


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

np.random.shuffle(dataset)

# split the dataset into training and validation datasets
splitratio = 0.8

training_dataset = dataset[:int(len(dataset)*splitratio)]
validation_dataset = dataset[int(len(dataset)*splitratio):]

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

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

allloss = []
allaccuracy = []

for epoch in range(1000):
    outputs = model(X)
    loss = criterion(outputs,Y)
    loss.backward() # TODO: wtf?
    optimizer.step()
    allloss.append(loss.item())
    allaccuracy.append(get_accuracy(outputs, Y))

    # for parameter in model.parameters():
    #     print(parameter)

validation = torch.Tensor([i[0:8] for i in validation_dataset])
#Y_val = torch.Tensor([i[7] for i in validation_dataset])

model.eval()

prediction_list = []

def step(data):
    inputs = data[0:-1]
    labels = data[-1]

    outputs = model(inputs)
    pred = torch.max(outputs.data)

    res = torch.Tensor([0])

    if pred > 0.5:
        res = torch.Tensor([1])

    return res

def predict(dataloader):
    predictions = []

    for i, batch in enumerate(dataloader):
        pred = step(batch)
        predictions.append(pred)
    
    return predictions


predictions = predict(validation  )

print("Accuracy:", get_accuracy(predictions, [row[-1] for row in validation]))

plt.plot(allaccuracy)
plt.plot(allloss)
plt.show()