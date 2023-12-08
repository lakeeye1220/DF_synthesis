#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

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
from torch.utils.data import DataLoader 
from torchvision import datasets
import argparse
import torchvision.models as models
from tqdm import tqdm

parser = argparse.ArgumentParser(description='train-teacher-network')
torch.backends.cudnn.enabled = False
# Basic model parameters.
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST','cifar10','cifar100','imagenet','caltech256'])
parser.add_argument('--data', type=str, default='./data/CIFAR10')
parser.add_argument('--output_dir', type=str, default='./')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0
earlyStop=0

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if args.dataset == 'caltech256':

    transform_train = transforms.Compose([
        transforms.Pad(4,padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])




if args.dataset == 'MNIST':
    
    data_train = MNIST(args.data,download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    data_test = MNIST(args.data,download=True,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=128, num_workers=8)

    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
if args.dataset == 'cifar10':
    
    transform_train = transforms.Compose([
        transforms.Pad(4,padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # data_train = CIFAR10(args.data,download=True,
    #                    transform=transform_train)
    data_train = datasets.ImageFolder(root='/home/jihwan/DF_synthesis/ours_nocut_nodenorm_2k_9557_50k', transform=transform_train)
    data_test = CIFAR10(args.data,download=True,
                      train=False,
                      transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=1024, shuffle=True, num_workers=2)
    data_test_loader = DataLoader(data_test, batch_size=512, num_workers=0)

    net = ResNet18().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'cifar100':
    
    transform_train = transforms.Compose([
        transforms.Pad(4,padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # data_train = CIFAR100(args.data,download=True,
    #                    transform=transform_train)
    data_train = datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet_threshold0.94_cifar100_256000_ver2/Fake_ours_resnet_threshold0.94_cifar100_256000_ver1', transform=transform_train)
    data_test = CIFAR100(args.data,download=True,
                      train=False,
                      transform=transform_test)
                      
    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)
    net = ResNet18(num_classes=100).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)


if args.dataset == 'imagenet':
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    '''
    data_train = CIFAR10(args.data,download=True,
                       transform=transform_train)
    data_test = CIFAR10(args.data,download=True,
                      train=False,
                      transform=transform_test)
    '''
    data_train = datasets.ImageFolder(root='./tiny-imagenet-200/train/parsed', transform=transform_train) 
    data_test = datasets.ImageFolder(root='./tiny-imagenet-200/val/parsed', transform=transform_train) 

    data_train_loader = DataLoader(data_train, batch_size=512, shuffle=True, num_workers=1)
    data_test_loader = DataLoader(data_test, batch_size=256, shuffle=False, num_workers=1)

    #net = ResNet34(num_classes=1000).cuda()
    net = models.resnet34(pretrained=True, progress=True)
    set_parameter_requires_grad(net, True)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, 200)
    net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    net.maxpool = torch.nn.Sequential()
    input_size = 64
    net = net.to('cuda')

    print(net)
    #checkpoint = torch.load('./pretrained/resnet34_imagenet.pth')
    #net.load_state_dict(checkpoint)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    # if args.dataset != 'MNIST':
    #     adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    print(len(data_train_loader))
    for i, (images, labels) in tqdm(enumerate(data_train_loader)):
        images, labels = images.cuda(), labels.cuda()
 
        optimizer.zero_grad()
 
        #output,_,_,_,_,_ = net(images)
        output = net(images)
        #output = net(images)
        #output = net(images,labels)
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
 
        loss.backward()
        optimizer.step()
 
 
def test(epoch):
    global acc, acc_best,earlyStop,epoch_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(data_test_loader)):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            #output,_,_,_,_,_ = net(images)
            #output = net(images,labels)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
        epoch_best = epoch
        print("Best test accuracy : ",acc_best )
        #torch.save(net,args.output_dir)
        torch.save(net.state_dict(),args.output_dir+'resnet18_NI_cifar10_best.pt')
        #print("Model save")
    if acc >= 0.9505 and acc < 0.9510:
        print("hihi")
        earlyStop += 1
        torch.save(net.state_dict(), args.output_dir+f'resnet18_NI_cifar10_of_{earlyStop}.pt')
    print('Test Avg. Loss: %f, Accuracy: %f Best Accuracy: %f at epoch %d' % (avg_loss.data.item(), acc, acc_best, epoch_best))
    return acc,earlyStop


def train_and_test(epoch):
    train(epoch)
    _,earlyStop = test(epoch)
    return earlyStop
 
 
def main():
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 200
    for e in range(1, epoch):
        train_and_test(e)
        #if earlyStop==10:
        #    break
    #torch.save(net,args.output_dir + 'teacher')
 
 
if __name__ == '__main__':
    main()
