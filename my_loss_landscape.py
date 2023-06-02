# 이전에 loss landscape를 위한 코드입니다.
# 한 batch에 대해서 진행하고 있습니다.
# class와 batch 정도를 argument로 전달해주시면 됩니다.

import numpy as np
import torch 
from torchvision import datasets, transforms
import argparse
from PyHessian_master.pyhessian import hessian
from PyHessian_master.density_plot import get_esd_plot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from utils import *
from tqdm import tqdm
import os
from PIL import Image

torch.backends.cudnn.benchmark = True
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)
# torch.cuda.manual_seed_all(100)

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def get_data(data_orig, data_perb, direction, alpha):
    perb_list = []
    for d_orig, d_perb, d in zip(data_orig, data_perb, direction):
        d_perb.data = d_orig.data + alpha * d
        perb_list.append(d_perb)
    perbs = torch.stack(perb_list, 0)
    if torch.equal(perbs, data_orig) and alpha != 0:
        print('origin and perturbed data are euqal')
    return perbs

def denormalize(image_tensor, dataset):
    channel_num = 0
    if dataset == 'cifar10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        channel_num = 3
    elif dataset == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        channel_num = 3

    for c in range(channel_num):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c]*s+m, 0, 1)

    return image_tensor


def save_img(input, file_name):
    grid = vutils.make_grid(input, nrow=10)
    pil_image = transforms.functional.to_pil_image(grid)
    pil_image.save(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='custom loss landscape')

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--model", default="resnet34", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data_path", default="./", type=str)
    parser.add_argument("--to_perturb", default="input", type=str)
    parser.add_argument("--class_idx", default=0, type=int)

    args = parser.parse_args()

    i = 0
    file_path = f"./scape{i}_class{args.class_idx}_batch{args.batch_size}"

    while os.path.exists(file_path):
        file_path = f"./scape{i}_class{args.class_idx}_batch{args.batch_size}"
        i += 1
    os.makedirs(file_path)

    # load model
    net = nn.DataParallel(load_net(model_name=args.model)).to(args.device)
    # net = load_net(model_name=args.model).to(args.device)
    criterion = nn.CrossEntropyLoss()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # real loader
    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
    loaders = {target: DataLoader(subset,batch_size=args.batch_size, shuffle=True) for target, subset in subsets.items()}

    # biggan loader
    fake_trainset = datasets.ImageFolder('./fake_images', transform=transform_train)
    fake_subsets = {target: Subset(fake_trainset, [i for i, (x, y) in enumerate(fake_trainset) if y == target]) for _, target in fake_trainset.class_to_idx.items()}
    fake_loaders = {target: DataLoader(fake_subset,batch_size=args.batch_size, shuffle=True) for target, fake_subset in fake_subsets.items()}

    # NI loader
    NI_trainset = datasets.ImageFolder('/home/dyl9912/workspace/KIST/NaturalInversion-main/images/official/final_images', transform=transform_train)
    NI_subsets = {target: Subset(NI_trainset, [i for i, (x, y) in enumerate(NI_trainset) if y == target]) for _, target in NI_trainset.class_to_idx.items()}
    NI_loaders = {target: DataLoader(NI_subset,batch_size=args.batch_size, shuffle=True) for target, NI_subset in NI_subsets.items()}

    loader = loaders[args.class_idx]
    fake_loader = fake_loaders[args.class_idx]
    NI_loader = NI_loaders[args.class_idx]

    for input, target in loader:
        input = denormalize(input, 'cifar10')
        save_img(input, file_name=f"{file_path}/real.png")
        break

    for f_input, f_target in fake_loader:
        f_input = denormalize(f_input, 'cifar10')
        save_img(f_input, file_name=f"{file_path}/fake.png")
        break

    for n_input, n_target in NI_loader:
        n_input = denormalize(n_input, 'cifar10')
        save_img(n_input, file_name=f"{file_path}/ni.png")
        break

    print("input shapes: ", input.shape, f_input.shape, n_input.shape)
    print("file all saved")

    input = input.to(args.device)
    target = target.to(args.device)

    f_input = f_input.to(args.device)
    f_target = f_target.to(args.device)

    n_input = n_input.to(args.device)
    n_target = n_target.to(args.device)

    if args.to_perturb == "input": # error -> require grad?
        input.requires_grad_(True)
        f_input.requires_grad_(True)
        n_input.requires_grad_(True)
        net.eval()

    hessian_comp = hessian(net, criterion, data=(input, target))
    hessian_comp_fake = hessian(net, criterion, data=(f_input, f_target))
    hessian_comp_ni = hessian(net, criterion, data=(n_input, n_target))

    # get the top1, top2 eigenvectors
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    f_top_eigenvalues, f_top_eigenvector = hessian_comp_fake.eigenvalues(top_n=2)
    n_top_eigenvalues, n_top_eigenvector = hessian_comp_ni.eigenvalues(top_n=2)

    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams1 = np.linspace(-0.5, 0.5, 31).astype(np.float32)
    lams2 = np.linspace(-0.5, 0.5, 31).astype(np.float32)

    print("real eigenvalue: ",top_eigenvalues)
    print("fake eigenvalue: ",f_top_eigenvalues)
    print("NI eigenvalue: ",n_top_eigenvalues)

    print("real loss of org: ", criterion(net(input)[0], target).item())
    print("fake loss of org: ", criterion(net(f_input)[0], f_target).item())
    print("ni loss of org: ", criterion(net(n_input)[0], n_target).item())

    loss_list = []
    f_loss_list = []
    n_loss_list = []

    # create a copy of the model
    model_perb1 = load_net(model_name=args.model).to(args.device)
    model_perb1.eval()

    if args.to_perturb == "input":

        data_perb1 = input.clone()
        data_perb2 = input.clone()

        f_data_perb1 = f_input.clone()
        f_data_perb2 = f_input.clone()

        n_data_perb1 = n_input.clone()
        n_data_perb2 = n_input.clone()
        
        for lam1 in tqdm(lams1, desc='lamda1'):
            for lam2 in lams2:
                data_perb1 = get_data(input, data_perb1, top_eigenvector[0][0], lam1)
                data_perb2 = get_data(data_perb1, data_perb2, top_eigenvector[1][0], lam2)

                f_data_perb1 = get_data(f_input, f_data_perb1, f_top_eigenvector[0][0], lam1)
                f_data_perb2 = get_data(f_data_perb1, f_data_perb2, f_top_eigenvector[1][0], lam2)

                n_data_perb1 = get_data(n_input, n_data_perb1, n_top_eigenvector[0][0], lam1)
                n_data_perb2 = get_data(n_data_perb1, n_data_perb2, n_top_eigenvector[1][0], lam2)
                    
                loss_list.append((lam1, lam2, criterion(model_perb1(data_perb2)[0], target).item()))
                f_loss_list.append((lam1, lam2, criterion(model_perb1(f_data_perb2)[0], f_target).item()))
                n_loss_list.append((lam1, lam2, criterion(model_perb1(n_data_perb2)[0], n_target).item()))
    else:
        pass


    loss_list = np.array(loss_list)
    f_loss_list = np.array(f_loss_list)
    n_loss_list = np.array(n_loss_list)
                         
    fig = plt.figure()
    landscape = fig.gca(projection='3d')
    landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')


    landscape.set_title('Real Loss Landscape')
    landscape.set_xlabel(f'ε_1, {top_eigenvalues[0]:.4f}')
    landscape.set_ylabel(f'ε_2, {top_eigenvalues[1]:.4f}')
    landscape.set_zlabel(f'Loss min: {min(loss_list[:,2]):.4f}, max: {max(loss_list[:,2]):.4f}')

    landscape.view_init(elev=30, azim=45)
    landscape.dist = 9

    # Set the limits of the z-axis
    z_min = 0.0
    z_max = 10.0
    landscape.set_zlim(z_min, z_max)

    plt.show()
    plt.savefig(f'{file_path}/Real_resnet34_cifar10_'+str()+'class_'+str(args.class_idx))

    fig2 = plt.figure()
    landscape2 = fig2.gca(projection='3d')
    landscape2.plot_trisurf(f_loss_list[:,0], f_loss_list[:,1], f_loss_list[:,2],alpha=0.8, cmap='viridis')
                        #cmap=cm.autumn, #cmamp = 'hot')


    landscape2.set_title('BigGAN Loss Landscape')
    landscape2.set_xlabel(f'ε_1, {f_top_eigenvalues[0]:.4f}')
    landscape2.set_ylabel(f'ε_2, {f_top_eigenvalues[1]:.4f}')
    landscape2.set_zlabel(f'Loss min: {min(f_loss_list[:,2]):.4f}, max: {max(f_loss_list[:,2]):.4f}')

    landscape2.view_init(elev=30, azim=45)
    landscape2.dist = 9

    # Set the limits of the z-axis
    landscape2.set_zlim(z_min, z_max)

    plt.show()
    plt.savefig(f'{file_path}/Generator_resnet34_cifar10_'+str()+'class_'+str(args.class_idx))

    fig3 = plt.figure()
    landscape3 = fig3.gca(projection='3d')
    landscape3.plot_trisurf(n_loss_list[:,0], n_loss_list[:,1], n_loss_list[:,2],alpha=0.8, cmap='viridis')
                        #cmap=cm.autumn, #cmamp = 'hot')


    landscape3.set_title('NI Loss Landscape')
    landscape3.set_xlabel(f'ε_1, {n_top_eigenvalues[0]:.4f}')
    landscape3.set_ylabel(f'ε_2, {n_top_eigenvalues[1]:.4f}')
    landscape3.set_zlabel(f'Loss min: {min(n_loss_list[:,2]):.4f}, max: {max(n_loss_list[:,2]):.4f}')

    landscape3.view_init(elev=30, azim=45)
    landscape3.dist = 9

    # Set the limits of the z-axis
    landscape3.set_zlim(z_min, z_max)

    plt.show()
    plt.savefig(f'{file_path}/NI_resnet34_cifar10_'+str()+'class_'+str(args.class_idx))


