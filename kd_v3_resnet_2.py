from __future__ import print_function
from __future__ import absolute_import

import os,sys
import time
import shutil
import pandas as pd
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
from termcolor import colored
from datetime import datetime
#from lenet import LeNet5,LeNet5Half
#sys.path.insert(0,os.path.abspath('..'))
#from mnist.lenet import LeNet5,LeNet5Half
from resnet import ResNet18, ResNet34, ResNet50
#from torch_resnet import resnet50
import vgg
#import distiller
#import models
sys.path.append('./mobileNet-v2_cifar10')
#from network import MobileNetV2

parser =argparse.ArgumentParser(description='YujinKim KD framework')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=400,
                    help='# of total epochs to run')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='# of mini-batch size (default: 256)')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--t_arch', type=str ,default='resnet34',
                    help='model to use')
parser.add_argument('--s_arch', type=str ,default='resnet50',
                    help='model to use')
parser.add_argument('--teacher_weights', default='./pretrained/resnet34.pt', type=str, help='path to load weights of the model')
parser.add_argument('--save', type=str, default='./', metavar='PATH',
                    help='path to save model')
parser.add_argument('--resume', type=str, default='', metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=777, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--evaluate', action='store_true', default=True,
                    help='whether to run evaluation')
parser.add_argument('--log_interval', type=int, default=13, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--tracefile_top1', type=str, default='gaussian_1.0_250epoch_top1_KL1.0_DAFL_lr0.1_batch_128', metavar='PATH',
                    help='path to save top-1 tracefile')
parser.add_argument('--tracefile_top5', type=str, default='', metavar='PATH',
                    help='path to save top-5 tracefile')
parser.add_argument('--ngpu', type=str, default='0', metavar='strN',
                    help='device number as a string')
parser.add_argument('--threads', type=int, default=3, metavar='N',
                    help='# of threads')
parser.add_argument('--alpha', type=float, default='1.0', metavar='FN',
                    help='alpha in paper')
parser.add_argument('--gamma', type=float, default='1000', metavar='FN',
                    help='gamma for the update rate of distillation loss')
parser.add_argument('--kd', action='store_true', default=True,
                    help='transfer class probability')
parser.add_argument('--adjust_rate', type = int, default=80,
                    help='transfer class probability')
parser.add_argument('--dir', type = str,
                    help='transfer class probability')
parser.add_argument('--csv_name', type = str,
                    help='transfer class probability')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
temperature = 2

if os.path.isdir(args.save) == False:
    os.mkdir(args.save)

def cprint(print_string, color):
    cmap={"r": "red", "g": "green", "y": "yellow", "b": "blue"}
    print(colored("{}".format(print_string), cmap[color]))

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cprint(args.cuda, "r")

print("Python  version: {}".format(sys.version.replace('\n', ' ')))
print("PyTorch version: {}".format(torch.__version__))
print("cuDNN   version: {}".format(torch.backends.cudnn.version()))
arguments = vars(args)
for key, value in arguments.items():
    print(f"{key} : {value}")


def adjust_lr(optimizer, lr, adjust_rate, epoch):
    lr = lr * (0.5 ** (epoch // adjust_rate))
    for param in optimizer.param_groups:
        param['lr'] = lr
    return lr

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


criterion_CE = nn.CrossEntropyLoss().cuda()
## Top-1, Top-5 Accuracy
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #print(pred)
        #print(target)
        #print('')
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print(correct)
        tot_correct = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            tot_correct.append(correct_k)
    return tot_correct


## Training with KD
def train_KD(t_net, s_net, epoch):
    s_net.train()
    t_net.eval()
    train_loss = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #print(targets)
        batch_size = inputs.shape[0]
        
        optimizer.zero_grad()
        with torch.no_grad():
            out_t = t_net(inputs)
        out_s = s_net(inputs)
        if args.s_arch == 'mnetv2':
            #print(out_s.shape)
            loss_CE = criterion_CE(out_s, targets)
            #loss_KD = nn.KLDivLoss()(F.log_softmax(out_s/temperature, dim = 1), F.softmax(out_t[0].detach()/temperature, dim = 1)) * (temperature*temperature * 3.0)
            loss_KD = nn.KLDivLoss()(F.log_softmax(out_s/temperature, dim = 1), F.softmax(out_t.detach()/temperature, dim = 1)) * (temperature*temperature * 3.0)
        elif args.t_arch == 'vgg11':
            loss_CE = criterion_CE(out_s[0], targets)
            loss_KD = nn.KLDivLoss()(F.log_softmax(out_s[0]/temperature, dim = 1), F.softmax(out_t.detach()/temperature, dim = 1)) * (temperature*temperature * 3.0)
        elif args.t_arch == 'vgg16':
            loss_CE = criterion_CE(out_s[0], targets)
            loss_KD = nn.KLDivLoss()(F.log_softmax(out_s[0]/temperature, dim = 1), F.softmax(out_t.detach()/temperature, dim = 1)) * (temperature*temperature * 3.0)
        else:
            loss_CE = criterion_CE(out_s[0], targets)
            loss_CE = criterion_CE(out_s[0], targets)
            loss_KD = nn.KLDivLoss()(F.log_softmax(out_s[0]/temperature, dim = 1), F.softmax(out_t[0].detach()/temperature, dim = 1)) * (temperature*temperature * 3.0)

        loss = loss_KD + loss_CE
        
        loss.backward()
        optimizer.step()

        train_loss += loss_CE.item()
        b_idx = batch_idx

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(inputs), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss_CE.data.item()))
    return


def test(net, arch):
    net.eval()
    test_loss = 0
    corr1 = 0
    corr5 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            if arch == 'mnetv2' or arch =='vgg11' or arch=='vgg16':
                outputs = net(inputs)
                test_loss += criterion_CE(outputs, targets).item()
                corr1_, corr5_ = accuracy(outputs, targets, topk=(1,1))
            else:
                outputs = net(inputs)
                test_loss += criterion_CE(outputs[0], targets).item()
                corr1_, corr5_ = accuracy(outputs[0], targets, topk=(1,1))

            corr1 += corr1_
            corr5 += corr5_

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\nTop-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, corr1.item(), len(test_loader.dataset),
            100. * float(corr1.item() / len(test_loader.dataset)),
            corr5.item(), len(test_loader.dataset),
            100. * float(corr5.item() / len(test_loader.dataset))))
    return float(test_loss), float(corr1.item()/len(test_loader.dataset)), float(corr5.item()/len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'best.pth.tar'))


if __name__ == "__main__":
    ## Dataset preprocessing and instantiating
    kwargs = {'num_workers': args.threads, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.2023, 0.1994, 0.2010])

        train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(root=f'./{args.dir}', transform=transforms.Compose([
                #datasets.ImageFolder(root=f'./dataset/{args.dir}', transform=transforms.Compose([
                #datasets.ImageFolder(root=f'../data/Inversion_dataset/{args.dir}', transform=transforms.Compose([
                            transforms.Pad(4, padding_mode='reflect'),
                            # transforms.Resize((32,32)),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data/CIFAR10', train=False, download=True,transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Resize((224,224)),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar10_ori':
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data/CIFAR10', train = True, download = True, transform = transforms.Compose([
                            transforms.Pad(4, padding_mode = 'reflect'),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size = args.batch_size, shuffle = True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data/CIFAR10', train=False, download=True,transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset =='mnist':
        trainset = datasets.ImageFolder(root='../mnist/final_images_KL_1.0',transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,),(0.3081,))
                            ]))

        train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,**kwargs)
        test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data/MNIST/processed',train=False,download=True, transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        #transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))
                        ])), 
                    batch_size = args.test_batch_size, shuffle=False, **kwargs)
                
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
                #datasets.ImageFolder(root=f'../data/Inversion_dataset/{args.dir}', transform=transforms.Compose([
                datasets.ImageFolder(f'{args.dir}', transform=transforms.Compose([
                            transforms.Pad(4, padding_mode='reflect'),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
        ''' 
        test_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(f'./dataset/{args.dir}', transform=transforms.Compose([
                            transforms.Pad(4, padding_mode='reflect'),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.batch_size, shuffle=False, **kwargs)
        '''
        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'cifar100_ori':
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data/cifar100', train = True, download = True, transform = transforms.Compose([
                            transforms.Pad(4, padding_mode = 'reflect'),
			    transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'imagenet':
        #traindir = os.path.join(args.data, 'train')
        #testdir = os.path.join(args.data, 'val3')
        traindir=f'./dataset/{args.dir}'
        testdir = './val3'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

        test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

    elif args.dataset == 'cinic10':
        traindir = os.path.join(args.data, 'train')
        testdir = os.path.join(args.data, 'test')
        normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                        std=[0.24205776, 0.23828046, 0.25874835])
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                    transforms.Pad(4),
                                    transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

        test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

    else:
        raise ValueError("No valid dataset is given.")

    ## Model
    #teacher_path = os.path.join(args.resume, 'model_best.pth.tar')
    if args.t_arch =='lenet':
        teacher_net = LeNet5()
        student_net = LeNet5Half()
        checkpoint = torch.load(args.teacher_weights)
        teacher_net.load_state_dict(checkpoint)
        teacher_net.eval()

    elif args.t_arch =='resnet34':
        if args.dataset == 'cifar10' or args.dataset =='cifar10_ori':
            teacher_net = ResNet34()
            checkpoint = torch.load(args.teacher_weights)
            teacher_net.load_state_dict(checkpoint)
            teacher_net.eval()
        elif args.dataset == 'cifar100' or args.dataset =='cifar100_ori':
            teacher_net = ResNet34(num_classes=100)
            checkpoint = torch.load(args.teacher_weights)
            teacher_net.load_state_dict(checkpoint)
            teacher_net.eval()

    elif args.t_arch == 'resnet50':
        teacher_net = resnet50(pretrained=True, progress=True)
        teacher_net.eval()

    elif args.t_arch == 'vgg11':
        teacher_net = vgg.__dict__['vgg11_bn']()
        checkpoint = torch.load(args.teacher_weights)
        teacher_net.load_state_dict(checkpoint)
        teacher_net.eval()
    elif args.t_arch == 'vgg16':
        teacher_net = vgg.__dict__['vgg16_bn'](num_classes=100)
        checkpoint = torch.load(args.teacher_weights)
        teacher_net.load_state_dict(checkpoint)
        teacher_net.eval()

    if args.s_arch == 'resnet18':
        if args.dataset == 'cifar100' or args.dataset =='cifar100_ori':
            student_net = ResNet18(num_classes=100)
        elif args.dataset == 'cifar10' or args.dataset =='cifar10_ori':
            student_net = ResNet18(num_classes=10)
    elif args.dataset=='cifar100' and args.s_arch=='resnet34':
        student_net = ResNet34(num_classes=100)
    elif args.dataset == 'cifar100':
        student_net = ResNet18(num_classes=100)
    elif args.s_arch =='resnet50':
        if args.dataset =='cifar10':
            student_net = ResNet50()
        elif args.dataset == 'cifar100':
            student_net = ResNet50(num_classes=100)

    elif args.s_arch == 'mnetv2':
        if args.dataset == 'cifar10':
            student_net = MobileNetV2(10, alpha=1)
        elif args.dataset == 'cifar100':
            student_net = MobileNetV2(100, alpha=1)

    elif args.s_arch == 'vgg11':
        if args.dataset == 'cifar10':
            student_net = vgg.__dict__['vgg11_bn']()

    if args.cuda:
        #distiller.cuda()
        student_net.cuda()
        teacher_net.cuda()
        cudnn.benchmark = True

    ## Just for test of teacher network
    if args.evaluate:
        print('Performance of teacher network')
        test(teacher_net, args.t_arch)

    acc1=[]
    acc5=[]
    losses = []
    best_prec1=0.

    optimizer = optim.SGD(student_net.parameters(), lr=args.lr, momentum=args.momentum ,weight_decay=args.wd, nesterov=True)
    best_epoch = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_lr(optimizer, args.lr, args.adjust_rate, epoch)
        
        train_KD(teacher_net, student_net, epoch)
        loss, prec1, prec5 = test(student_net, args.s_arch)
        acc1.append(prec1) ## storing accuracy-1
        acc5.append(prec5) ## storing accuracy-5
        losses.append(loss)
        pd.DataFrame(acc1).to_csv(os.path.join("./csv_kd/", f'{args.csv_name}.csv'), index=None)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        if 0:
            best_epoch = epoch
            torch.save(student_net.state_dict(), f'./pretrained/cifar100_resnet34_{str(best_prec1).split(".")[-1]}.pt')

        '''
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath='./csv_kd/cifar10/cw_VR_lambda10_SGD256_lr0.1')
        '''
        print("Best accuracy: "+str(best_prec1))
        print("Finished saving training history")
