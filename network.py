import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.cuda as cuda
import torch.nn as nn
import torch

from tqdm import tqdm
import numpy as np

import importlib
import time

import argparse

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from PIL import Image


##### Argument Parsing ######
parser = argparse.ArgumentParser(description='NMIST-Practice')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Batch Size')
parser.add_argument('--use_seed', type=bool, default=False,
                    help='Whether to use seed for randomization')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for initialization of random_dataset_split, weight inits')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.4,
                    help='Momentum for network')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='Weight decay for network')
parser.add_argument('--name', type=str, default="results",
                    help='Name to name all the exported files of the network')
parser.add_argument('--normalize', type=bool, default=False,
                    help='Whether to use normalized data')
parser.add_argument('--lr_decay', type=float, default=1.0,
                    help='LR decay')
parser.add_argument('--lr_step_size', type=float, default=10,
                    help='LR decay step size')

args = parser.parse_args()

weight_decay = args.weight_decay
lr_step_size = args.lr_step_size
batch_size = args.batch_size
normalize = args.normalize
use_seed = args.use_seed
momentum = args.momentum
lr_decay = args.lr_decay
epochs = args.epochs
seed = args.seed
name = args.name
lr = args.lr


##### Confirm Cuda Is Available #####
print("Cuda Available:", cuda.is_available(), '\n')


##### Seeding #####
if use_seed:
    torch.manual_seed(seed)
    np.random.seed(seed)


##### Download MNIST Dataset #####
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

test_num = len(testset)


##### Load Datasets #####
train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(testset, batch_size=50, shuffle=True)


##### Neural Network Definition #####
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(64)
        )

        self.fcDropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 192)
        self.fc2 = nn.Linear(192, 96)
        self.fc3 = nn.Linear(96, 24)
        self.fc4 = nn.Linear(24, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fcDropout(x)
        x = F.relu(self.fc2(x))
        x = self.fcDropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


##### Initialize Network #####
net = Net()

if cuda.is_available():
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_decay)

train_losses = []
test_losses = []
test_accuracies = []


##### Training Loop #####
t0 = time.time()

print("Beginning Training:")
for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
    running_loss = []
    for i, (inputs, labels) in enumerate(train_loader):
        if cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        labels = torch.squeeze(labels)

        inputs = inputs.float()
        labels = labels.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    # print statistics
    avg_loss = np.mean(np.array([running_loss]))
    train_losses.append(avg_loss)

    if cuda.is_available():
        cuda.empty_cache()

    running_loss_test = []

    net.eval()
    epoch_correct = 0
    for i, (inputs, labels) in enumerate(test_loader):
        if cuda.is_available():
            cuda.empty_cache()

        labels = torch.squeeze(labels)

        inputs = inputs.float()
        labels = labels.long()

        if cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss_test.append(loss.item())

        for j in range(0, outputs.shape[0]):
            prediction = torch.argmax(outputs[j])
            ground_truth = labels[j]

            if prediction == ground_truth:
                epoch_correct += 1

    test_avg_loss = np.mean(np.array([running_loss_test]))
    test_losses.append(test_avg_loss)
    test_accuracies.append(epoch_correct / test_num)

    print("Epoch:", epoch, "| Avg Loss:", avg_loss,
          "\n         | Test Avg Loss:", test_avg_loss)

    if cuda.is_available():
        cuda.empty_cache()

    scheduler.step()


##### Test Data Evaluation #####
net.eval()

correct = 0
confusion_matrix = np.zeros((10, 10))

for i, (inputs, lebels) in enumerate(test_loader):
    if cuda.is_available():
        cuda.empty_cache()

    labels = torch.squeeze(labels)

    inputs = inputs.float()
    labels = labels.long()

    if cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    outputs = net(inputs)

    for j in range(0, outputs.shape[0]):
        prediction = torch.argmax(outputs[j])
        ground_truth = labels[j]

        confusion_matrix[ground_truth][prediction] += 1

        if prediction == ground_truth:
            correct += 1

##### Results #####
print("Test Accuracy =", (correct / test_num))
print("Time taken =", (time.time() - t0))

print("Saving weight model to : " +
      name + ".pth")

# Save Weights
torch.save(net.state_dict(), name + ".pth")

# Losses Plot
plt.figure()
plt.plot(range(0, epochs),
         train_losses, 'r--', range(0, epochs), test_losses, 'b--')
plt.ylabel('Loss')
plt.xlabel('Number of Iterations')
plt.title("Learning rate =" + str(lr))
# plt.show()
plt.savefig(name + 'Losses.png')

# Test Accuracy Plot

plt.figure()
plt.plot(range(0, epochs),
         test_accuracies, 'r--')
plt.ylabel('Accuracy')
plt.xlabel('Number of Iterations')
plt.title("Test Accuracy")
# plt.show()
plt.savefig(name + 'Accuracy.png')


print(confusion_matrix)

# Confusion Matrix
df_cm = pd.DataFrame(confusion_matrix, 
                     index=['0',
                            '1',
                            '2',
                            '3',
                            '4',
                            '5',
                            '6',
                            '7',
                            '8',
                            '9'],
                     columns=['0',
                              '1',
                              '2',
                              '3',
                              '4',
                              '5',
                              '6',
                              '7',
                              '8',
                              '9'])
plt.figure(figsize=(10, 10))
plt.xlabel("Predicted Number")
plt.ylabel("Ground Truth Number")
sn.heatmap(df_cm, annot=True, cmap='YlGnBu')
plt.savefig(name + 'ConfusionMatrix.png')

# Saves Results text file

txt = "Test Accuracy: " + str((correct / test_num)) + "\n"
txt += "Epochs: " + str(epochs) + "\n"
txt += "LR: " + str(lr) + "\n"
txt += "Momentum: " + str(momentum) + "\n"
txt += "Weight Decay: " + str(weight_decay) + "\n"
txt += "Batch Size: " + str(batch_size) + "\n"
txt += "Seed: " + str(seed) + "\n"
txt += "Use Seed: " + str(use_seed) + "\n"
txt += "Normalize: " + str(normalize) + "\n"
txt += "LR Decay: " + str(lr_decay) + "\n"
txt += "LR Step Size: " + str(lr_step_size) + "\n"

with open(name + ".txt", "w") as file:
    file.write(txt)
