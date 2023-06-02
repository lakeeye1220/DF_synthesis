# ResNet34 학습시키기 위한 코드입니다. from scratch 학습 결과 test 35% 정도 나왔습니다.(아쉽네요)

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse

from resnet import ResNet34
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset',default='cifar10',type = str, help ='traning set')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

writer = SummaryWriter(f'result/tb_summary')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset =='cifar10':
    """trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)"""
    
    d_path = './Fake_cifar10(Trans201)'
    dataset = datasets.ImageFolder(d_path, transform=transform_train)
    trainloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
else : 
    trainset = torchvision.datasets.CIFAR100(
            root='./',train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(
            root='./',train=False, download=True, transform=transform_train)
    testloader = torch.utils. data.DataLoader(
            testset, batch_size=100,shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet34()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs,_,_,_,_,_ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, '/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return (train_loss/(batch_idx+1)), 100.*correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_,_,_,_,_ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, '/', len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    acc = 100.*correct/total
    print(f"TEST ACC on {epoch}: {acc}")

    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('pretrained'):
            os.mkdir('pretrained')
        torch.save(state, './pretrained/resnet18_{}_{}.pth'.format(args.dataset,best_acc))
        best_acc = acc

    return (test_loss/(batch_idx+1)), 100.*correct/total

for epoch in range(start_epoch, start_epoch+200):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    writer.add_scalar(f"loss/train_loss", train_loss, epoch)
    writer.add_scalar(f"loss/test_loss", test_loss, epoch)
    writer.add_scalar(f"acc/train_acc", train_acc, epoch)
    writer.add_scalar(f"acc/test_acc", test_acc, epoch)
    scheduler.step()
