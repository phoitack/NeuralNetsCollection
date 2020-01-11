import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# Transfer learning based Resnet18.
# Considering only training and validation.


# Data augmentation by dictionary define.
data_transforms = {
    'train': transforms.Compose([   # transform for the training
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and std for each channel (RGB)
    ]),
    'val': transforms.Compose([     # transform for the validation
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=4,
                                              shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print("Class names: {}".format(class_names))
print("Number of batch in training set: {}".format(len(dataloaders['train'])))
print("Number of batch in test set: {}".format(len(dataloaders['val'])))
print("Number of training images: {}".format(dataset_sizes['train']))
print("Number of test images: {}".format(dataset_sizes['val']))

# model loading
model_conv = torchvision.models.resnet18(pretrained=True)

# Freezing layers
for param in model_conv.parameters():
    param.requires_grad = False
# Number of inputs for the last layers
num_ftrs = model_conv.fc.in_features
# Here we reconstruct the last layer to have the desired number of classes (2)
model_conv.fc = nn.Linear(num_ftrs, 2)

iteration = 0
correct = 0
for inputs, labels in dataloaders['train']:
    if iteration == 1:
        break
    inputs = Variable(inputs)
    labels = Variable(labels)
    print("After one iteration:")
    print("Input shape: ", inputs.shape)
    print("Label shape: ", labels.shape)
    print("Label are: {}".format(labels))
    out = model_conv(inputs)
    print("Output tensor:", out)
    print("Outputs shape", out.shape)
    _, predicted = torch.max(out, 1)
    print("Predicted Shape", predicted)
    print("Predicted Shape", predicted.shape)
    correct += (predicted == labels).sum()
    print("Correct Prediction:", correct)

    iteration +=1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

num_epochs = 10
for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    correct = 0
    for images, labels in dataloaders['train']:
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        out = model_conv(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()

    train_acc = 100 * correct / dataset_sizes['train']
    print('Epoch [{}/{}], loss: {:.4f}, train accuracy: {}%'.format(epoch+1,num_epochs,loss.item(),train_acc))

# Test

model_conv.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for (images, labels) in dataloaders['val']:
        images = Variable(images)
        labels = Variable(labels)
        out = model_conv(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test accuracy: {:.3f} %'.format(100 * correct / total))

# Visualization
fig = plt.figure()
shown_batch = 0
index = 0
with torch.no_grad():
    for (images, labels) in dataloaders['val']:
        if shown_batch == 1:
            break
        shown_batch += 1
        images = Variable(images)
        labels = Variable(labels)

        out = model_conv(images)
        _, preds = torch.max(out, 1)

        for i in range(4):
            index += 1
            ax = plt.subplot(2, 2, index)
            ax.axis('off')
            ax.set_title('Predicted Label: {}'.format(class_names[preds[i]]))
            input_img = images.cpu().data[i]
            inp = input_img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)