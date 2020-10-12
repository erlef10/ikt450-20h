import torch
from torch.nn.modules.activation import Softmax2d
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import natsort
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import RandomCrop
from autoaugment import ImageNetPolicy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Running on device {}".format(device))

torch.cuda.empty_cache() 

image_size = 32
batch_size = 8
num_epochs = 200

train_losses = []
valid_losses = []

classes = (
    'bread',
    'dairy product',
    'dessert',
    'egg',
    'fried food',
    'meat',
    'noodles/pasta',
    'rice',
    'seafood',
    'soup',
    'vegetable/fruit'
)

def imshow(image):
    plt.imshow(transforms.ToPILImage()(image.cpu().clone()))
    plt.show()

class FoodDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_images = os.listdir(main_dir)
        self.total_images = natsort.natsorted(all_images)

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        image = PIL.Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image).to(device)
        label = int(self.total_images[idx].split('_')[0])

        return tensor_image, label

image_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((image_size, image_size)),
        ImageNetPolicy(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'valid':
    transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'test':
    transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

data = {
    'train': FoodDataSet('food-11/training', transform=image_transforms['train']),
    'valid': FoodDataSet('food-11/validation', transform=image_transforms['valid']),
    'test': FoodDataSet('food-11/evaluation', transform=image_transforms['test'])
}

data_loaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}
class FoodNet(nn.Module):
    def __init__(self, num_classes=11):
        super(FoodNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FoodNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

dataiter = iter(data_loaders['train'])
images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images))


print("Starting to train and evaluate model...")
for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    # Traning the model
    model.train()
    for data, target in data_loaders['train']:
        # Move tensors to GPU
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * data.size(0)

    # Validating the model
    model.eval()
    for data, target in data_loaders['valid']:
        # Move tensors to GPU
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        
        loss = criterion(output, target)
        
        # Update average validation loss
        valid_loss += loss.item() * data.size(0)

    # Calculate average losses
    train_loss = train_loss / len(data_loaders['train'].sampler)
    valid_loss = valid_loss / len(data_loaders['valid'].sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_loaders['test']:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()

# dataiter = iter(test_loader)
# images, labels = dataiter.next()
# outputs2 = net(images)
# _, predicted = torch.max(outputs2, 1)

# print("{:<20}{:<20}".format("Predicted", "Ground Truth"))
# for j in range(16):
#     print("{:<20}{:<20}".format(classes[predicted[j]], classes[labels[j]]))

# #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(16)))
# #print('GroundTruth: ', ' '.join('%5s' % classes[labels]))
# imshow(torchvision.utils.make_grid(images))
