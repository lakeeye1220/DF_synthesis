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
from utils import *
import copy

torch.backends.cudnn.benchmark = True
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

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
    perbs = torch.cat(perb_list,0)
    return perbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='custom loss landscape')

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--model", default="resnet34", type=str)
    parser.add_argument("--daaset", default="cifar10", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--to_perturb", default="input", type=str)
    parser.add_argument("--class_idx", default=3, type=int)

    args = parser.parse_args()

    # load model
    net = nn.DataParallel(load_net(model_name=args.model)).to(args.device)
    # net = load_net(model_name=args.model).to(args.device)
    criterion = nn.CrossEntropyLoss()

    # load data
    dataset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transforms.ToTensor())
    subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
    #train_sampler = torch.utils.data.distributed.DistributedSampler(subsets,
    #                                                                num_replicas=4,
    #                                                                rank=None,
    #                                                                shuffle=False,
    #                                                                drop_last=True)

    loaders = {target: DataLoader(subset,batch_size=args.batch_size) for target, subset in subsets.items()}
    loader = loaders[args.class_idx]

    # loader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    
    for input, target in loader:
        break

    input = input.to(args.device)
    target = target.to(args.device)

    if args.to_perturb == "input": # error -> require grad?
        input.requires_grad_(True)
        net.eval()

    hessian_comp = hessian(net, criterion, data=(input, target))

    # get the top1, top2 eigenvectors
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)


    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    lams2 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    print(top_eigenvalues)
    loss_list = []

    if args.to_perturb == "input":
        # create a copy of the model
        model_perb1 = load_net(model_name=args.model).to(args.device)
        model_perb1.eval()

        data_perb1 = input
        data_perb2 = input

        for lam1 in lams1:
            for lam2 in lams2:
                data_perb1 = get_data(input, data_perb1, top_eigenvector[0], lam1)
                data_perb2 = get_data(data_perb1, data_perb2, top_eigenvector[1], lam2)
                    
                loss_list.append((lam1, lam2, criterion(model_perb1(data_perb2)[0], target).item()))
    else:
        pass

    loss_list = np.array(loss_list)
                         
    fig = plt.figure()
    landscape = fig.gca(projection='3d')
    landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')
                        #cmap=cm.autumn, #cmamp = 'hot')


    landscape.set_title('Loss Landscape')
    landscape.set_xlabel('ε_1')
    landscape.set_ylabel('ε_2')
    landscape.set_zlabel('Loss')

    landscape.view_init(elev=30, azim=45)
    landscape.dist = 9
    plt.show()
    plt.savefig('./Real_resnet34_cifar10_'+str()+'class_'+str(args.class_idx))


