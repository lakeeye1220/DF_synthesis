import os
#from models.lenet import LeNet5
from resnet_cifar3 import ResNet18
# import models.resnet as resnet
import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import Caltech256
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import argparse
import torchvision.models as models
from tqdm import tqdm

# from sklearn.model_selection import train_test_split
# import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms, datasets, models

import shutil

# Set paths
data_folder = './data/Caltech256'  # Path to the Caltech-256 dataset folder
train_folder = './data/Caltech256/train'  # Destination folder for train data
test_folder = './data/Caltech256/test'  # Destination folder for test data

# Create train and test folders if they don't exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_classes = 257
batch_size = 32
num_epochs = 100
learning_rate = 0.005



def train_1epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    lr_scheduler
):
    epoch_acc = 0.
    epoch_loss = 0.
    model.train()
    for i,  (data, target) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        probs = torch.softmax(output, dim=1)
        preds = probs.argmax(dim=1)
        epoch_acc += torch.sum(preds == target).item()
        epoch_loss += loss.item()
        # if i == len(dataloader) - 1:
        #     print('Train - acc: %f, Loss: %f' % (epoch, i, loss.data.item()))

    lr_scheduler.step()
    epoch_acc = 100 * epoch_acc / len(dataloader.dataset)
    epoch_loss = epoch_loss / len(dataloader)


    return epoch_acc, epoch_loss

def validate_1epoch(
    model,
    dataloader,
    criterion,
    device
):
    epoch_acc = 0.
    epoch_loss = 0.
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            epoch_acc += torch.sum(preds == target).item()
            epoch_loss += criterion(output, target).item()
    epoch_acc = 100 * epoch_acc / len(dataloader.dataset)
    epoch_loss = epoch_loss / len(dataloader)

    return epoch_acc, epoch_loss 

# Data transformations
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



caltech_dataset = datasets.ImageFolder(root='/home/jihwan/DF_synthesis/256_ObjectCategories',transform=data_transform)

# split train and test.
n = len(caltech_dataset)
n_train = int(n * 0.8)
n_test = n - n_train
train_dataset, test_dataset = random_split(caltech_dataset, [n_train, n_test])




# Load the Caltech-256 dataset
# train_dataset = datasets.ImageFolder(root=train_folder, transform=data_transform)
# test_dataset = datasets.ImageFolder(root=test_folder, transform=data_transform)


# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load the DenseNet-201 model
model = models.densenet201(pretrained=True)




# For inference
# model = models.densenet201()
# model.load_state_dict(torch.load('./classifier_fine_tuned_weight/densenet_caltech256_best_acc.pth'),strict=False)



model.to(device)
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)


lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader) * num_epochs, eta_min=1e-6) 

# Training loop
total_steps = len(train_loader)

# loop run epoch
best_loss = float('inf')
best_acc = -float('inf')
# best_accuracy = 0.0


for i in range(num_epochs):
    train_acc, train_loss = train_1epoch(model, train_loader, criterion, optimizer, device, lr_scheduler)
    print('Train - Epoch %d, Accuracy: %f, Loss: %f' % (i, train_acc, train_loss))
    
    val_acc, val_loss = validate_1epoch(model, test_loader, criterion, device)

    print('val - Epoch %d, Accuracy: %f, Loss: %f' % (i, val_acc, val_loss))
    # logging.info(
    #     f'''
    #     epoch:{i+1},
    #     train loss:{train_loss:.2f},
    #     train acc:{train_acc:.2f},
    #     val loss:{val_loss:.2f},
    #     val acc:{val_acc:.2f}
    #     '''
    # )
    # writer.add_scalar('Acc/train', train_acc, i+1)
    # writer.add_scalar('Acc/val', val_acc, i+1)
    # writer.add_scalar('Loss/train', train_loss, i+1)
    # writer.add_scalar('Loss/val', train_loss, i+1)





    if best_acc < val_acc:
        torch.save(model.state_dict(), './classifier_fine_tuned_weight/'+'densenet_caltech256_best_acc.pth')
        best_acc = val_acc
    if best_loss > val_loss:
        torch.save(model.state_dict(), './classifier_fine_tuned_weight/'+'densenet_caltech256_best_acc.pth')
        best_loss = val_loss




# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

#     # Test the model on the validation set
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         accuracy = (correct / total) * 100
#         print(f"Accuracy on validation set: {accuracy:.2f}%")

#         # Save the model if it has the best accuracy so far
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             torch.save(model.state_dict(), "best_model.pt")

#     model.train()

# # Load the best model and test on the test set
# best_model = models.densenet201(pretrained=False)
# best_model.classifier = nn.Linear(best_model.classifier.in_features, num_classes)
# best_model.load_state_dict(torch.load("best_model.pt"))
# best_model.to(device)

# best_model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = best_model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f"Accuracy on test set: {(correct / total) * 100:.2f}%")
