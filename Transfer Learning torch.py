import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np

# Based Resnet. Considering only training and validation.


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
    print("Predicted Shape", predicted.shape)
    correct += (predicted == labels).sum()
    print("Correct Prediction:", correct)

    iteration +=1