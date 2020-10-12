import torch
import torch.nn.functional as F
from glob import glob
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import dataset

import os
import natsort
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt 
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Running on device {}".format(device))

train_size = 0.8
batch_size = 16
num_epochs = 50

classes = (
    'normal',
    'abnormal'
)

class ECGDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        normal_files = glob(os.path.join(directory, 'normal/*.[!ann]'))
        abnormal_files = glob(os.path.join(directory, 'abnormal/*.[!ann]'))
        self.total_files = natsort.natsorted(normal_files + abnormal_files)

    def __len__(self):
        return len(self.total_files)

    def __getitem__(self, index):
        data_loc = self.total_files[index]

        data = np.loadtxt(data_loc, usecols=1)[:75]

        label = data_loc.split('\\')[1]
        label = 0 if label == 'normal' else 1

        if len(data) < 75:
            length = 75 - len(data)
            data = np.pad(data, (0, length), mode='constant')

        return torch.FloatTensor(data), label

ecg_dataset = ECGDataset('ecg')

normal_train_len = int(train_size * len(ecg_dataset))
normal_valid_len = len(ecg_dataset) - normal_train_len
normal_train, normal_test = random_split(ecg_dataset, [normal_train_len, normal_valid_len])

print("Traning set: {}, testing set: {}".format(len(normal_train), len(normal_test)))

data_loaders = {
    'normal_train': DataLoader(
        normal_train, 
        batch_size=batch_size, 
        shuffle=True,
    ),
    'normal_test': DataLoader(
        normal_test, 
        batch_size=batch_size, 
        shuffle=True
    ),
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(75, 75),
            nn.LeakyReLU(),
            nn.Linear(75, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return x

model = Net().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses = []

print("Starting to train the model...")
for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in data_loaders['normal_train']:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(data_loaders['normal_train'].sampler)
    train_losses.append(train_loss)

    if epoch % 10 == 0:
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in data_loaders['normal_test']:
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

