import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Data loading

# MNIST dataset properties
mean_gray = 0.1307
stddev_gray = 0.3081

# transform image to tensor and normalization
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((mean_gray,), (stddev_gray,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms)

random_img = train_dataset[20][0].numpy() * stddev_gray + mean_gray
plt.imshow(random_img.reshape(28, 28), cmap='gray')

batch_size = 100
train_load = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_dataset), len(test_dataset), len(test_load), len(train_load))


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # padding = (filter_size-1)/2
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1) #channel= 1 becuase gray images
        #output size= (input-filtersize +2padding)/stride +1=28
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568, 600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(-1, 1568)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = CNN()

loss_fn = nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.01)

# training

num_epochs = 10
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(num_epochs):
    correct = 0
    iterations = 0
    iter_loss = 0.0

    model.train()

    for i, (inputs, labels) in enumerate(train_load):

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        iterations += 1

    train_loss.append(iter_loss/iterations)
    train_accuracy.append(100* correct / len(train_dataset))

    # Test

    testing_loss = 0.0
    correct = 0
    iterations = 0

    model.eval()

    for i, (inputs, labels) in enumerate(test_load):

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        testing_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        iterations += 1

    test_loss.append(testing_loss / iterations)
    test_accuracy.append(100*correct / len(test_dataset))

    print('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
          .format(epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))

# plotting
f = plt.figure(figsize=(10, 10))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.legend()
plt.show()

f = plt.figure(figsize=(10, 10))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(test_accuracy, label='Testing Accuracy')
plt.legend()
plt.show()
