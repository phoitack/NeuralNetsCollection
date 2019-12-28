import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Net parameters
input_size = 784
hidden_size = 400
out_size = 10
epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
# Network architecture

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # first fully-connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # second fully-connected layer
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

net = Net(input_size, hidden_size, out_size)
CUDA =torch.cuda.is_available()
if CUDA:
    net = net.cuda()

# loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# training


# reshaping the tensor from 1x28x28 to 784
for epoch in range(epochs):
    correct_train = 0
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):  # iteration loop
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        output = net(images)
        _, predicted = torch.max(output.data, 1)  # taking only the second argument
        correct_train += (predicted == labels).sum()

        # loss function
        loss = criterion(output, labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # weights update

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format
                  (epoch + 1, epochs, running_loss / len(train_loader),
                   (100 * correct_train.double() / len(train_dataset))))
print("Done Training")

with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_dataset)))
