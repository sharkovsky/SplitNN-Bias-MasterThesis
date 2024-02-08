import torch
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
#import tensorflow as tf

import torchvision
import torchvision.transforms as transforms

#--------------------------------------------------------------------------------------------------DATASETS---------------------------------------------------------------------------------#

transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_mnist)


train_subset = Subset(train_dataset, range(1000))
test_subset = Subset(test_dataset, range(100))

batch_size = 10

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)


# Define transformations
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

for images, labels in trainloader:
    print(images.shape)
    break

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Client1(nn.Module):
    def __init__(self, x_train, y_train):
        super(Client1, self).__init__()

        self.num_samples = x_train.shape[0]
        self.x_train = x_train
        self.labels = y_train
        self.batch = 10

        self.model_1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1,1), padding='valid'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding='valid'),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2,2)),
                                   nn.Flatten()
                                   )

        self.model_2 = nn.Sequential(nn.Linear(64*12*12, 10),
                                     nn.Softmax(dim=1))

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, inputs):
        outputs = self.model_1(inputs)
        return outputs
    

    def send(self):
        logits = self.forward(self.x_train)
        return logits

    def forward_2(self, inputs):
        outputs = self.model_2(inputs)
        return outputs

    def loss(self, logits, labels):
        return self.loss_function(logits, labels)


class Client2(nn.Module):
    def __init__(self, x_train, y_train):
        super(Client2, self).__init__()

        self.num_samples = x_train.shape[0]
        self.x_train = x_train
        self.labels = y_train
        self.batch = 10

        self.model_1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1,1), padding='valid'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding='valid'),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2,2)),
                                   nn.Flatten()
                                   )

        self.model_2 = nn.Sequential(nn.Linear(64*12*12, 10),
                                     nn.Softmax(dim=1))

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, inputs):
        outputs = self.model_1(inputs)
        return outputs
    

    def send(self):
        logits = self.forward(self.x_train)
        return logits

    def forward_2(self, inputs):
        outputs = self.model_2(inputs)
        return outputs

    def loss(self, logits, labels):
        return self.loss_function(logits, labels)

#--------------------------------------------------------------------------------------First Client--------------------------------------------------------------------------------------------------#
x_train = []
y_train = []
x_test = []
y_test = []

for images, labels in train_loader:
    x_train.append(images)
    y_train.append(labels)
    
for images_test, labels_test in test_loader:
    x_test.append(images_test)
    y_test.append(labels_test)

# Concatenate the batches to obtain the full datasets
x_train = torch.cat(x_train, dim=0)
y_train = torch.cat(y_train, dim=0)

x_test = torch.cat(x_test, dim=0)
y_test = torch.cat(y_test, dim=0)

#-------------------------------------------------------------------------------------Second Client----------------------------------------------------------------------------------------------------#
x_train = []
y_train = []
x_test = []
y_test = []

for images, labels in train_loader:
    x_train.append(images)
    y_train.append(labels)
    
for images_test, labels_test in test_loader:
    x_test.append(images_test)
    y_test.append(labels_test)

# Concatenate the batches to obtain the full datasets
x_train = torch.cat(x_train, dim=0)
y_train = torch.cat(y_train, dim=0)

x_test = torch.cat(x_test, dim=0)
y_test = torch.cat(y_test, dim=0)

#----------------------------------------------------------------------------------------Training------------------------------------------------------------------------------------------------------#

for epoch in range(num_epochs):
    # Set the model to training mode
    Client1.train()
    Client2.train()
    



    # Iterate over the batches of training data
    for inputs, labels in train_loader1:
        # Zero the gradients
        optimizer_client1.zero_grad()
        optimizer_client2.zero_grad()
        
        # Forward pass
        outputs_1 = Client1(x_train, y_train)
        outputs_2 = Client2(inputs)
        
        # Compute the loss
        loss_1 = client1.loss(outputs_1, labels)
        loss_2 = client1.loss(outputs_1, labels)
        
        # Backward pass
        loss_1.backward()
        loss_2.backward()
        # Update the parameters
        optimizer_client1.step()
        optimizer_client1.step()
