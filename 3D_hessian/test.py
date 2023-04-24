import numpy as np
import torch 
from torchvision import datasets, transforms
from utils import * # get the dataset
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from torch.utils.data import Subset, DataLoader
#from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt
from matplotlib import animation
import sys
sys.path.append('../')
from resnet import ResNet34

#model = ptcv_get_model("resnet20_cifar10", pretrained=True)
#model2 = ptcv_get_model("sepreresnet20_cifar10", pretrained=True)

torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

#class index
class_idx = 0
prefix="../DF_synthesis/resnet34_cifar10_CE_entropy_clip05_"+str(class_idx)+"/"
batch_size=256

model = ResNet34()
model.load_state_dict(torch.load('../DF_synthesis/cifar10_resnet34_9557.pt'))
# change the model to eval mode to disable running stats upate
model = model.cuda()
model.eval()
#model2.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

#optimizer = optim.Adam(model.parameters(), lr=LR)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#dist.init_process_group("nccl", rank=0, world_size=2)
trainset = datasets.CIFAR10(root='../../data/CIFAR10', train=True, download=True, transform=transform_train)
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False)

subsets = {target: Subset(trainset, [i for i, (x, y) in enumerate(trainset) if y == target]) for _, target in trainset.class_to_idx.items()}
#train_sampler = torch.utils.data.distributed.DistributedSampler(subsets,
#                                                                num_replicas=4,
#                                                                rank=None,
#                                                                shuffle=False,
#                                                                drop_last=True)

loaders = {target: DataLoader(subset,batch_size=batch_size) for target, subset in subsets.items()}
train_loader = loaders[class_idx]
#inputs,targets = next(iter(train_loader_0))
#print("input shape :",inputs.shape)
# get dataset 
#train_loader, test_loader = class_0_loader

#biggan loadder
fake_trainset = datasets.ImageFolder('./fake_images', transform=transform_train)

fake_subsets = {target: Subset(fake_trainset, [i for i, (x, y) in enumerate(fake_trainset) if y == target]) for _, target in fake_trainset.class_to_idx.items()}

fake_loaders = {target: DataLoader(fake_subset,batch_size=batch_size) for target, fake_subset in fake_subsets.items()}
fake_loader = fake_loaders[class_idx]



#Gaussian noise

noise_inputs = torch.randn((batch_size,3,32,32)).cuda()
noise_targets = torch.LongTensor([class_idx] * batch_size).cuda()


input_list = []
target_list = []
# for illustrate, we only use one batch to do the tutorial

for inputs,targets in train_loader:
    #inputs,targets = d
    print(inputs.shape)
    #print(targets)
    #input_list.append(inputs)
    #target_list.append(targets)
    break

for fake_inputs,fake_targets in fake_loader:
    #inputs,targets = d
    print(fake_inputs.shape)
    #print(fake_targets)
    #input_list.append(inputs)
    #target_list.append(targets)
    break
#inputs = torch.cat(input_list,0)
#targets = torch.cat(target_list,0)
# we use cuda to make the computation fast
inputs, targets = inputs.cuda(), targets.cuda()

hessian_comp = hessian(model, criterion, data=(inputs,targets), dataloader=None, cuda=True) #data = (inputs,targets)
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
print("real_top_eigenvalues: ",top_eigenvalues)


#'''
#BigGAN images

fake_inputs, fake_targets = fake_inputs.cuda(), fake_targets.cuda()
fake_hessian_comp = hessian(model, criterion, data=(fake_inputs,fake_targets), dataloader=None, cuda=True)
fake_top_eigenvalues, fake_top_eigenvector = fake_hessian_comp.eigenvalues(top_n=2)
print("fake_top_eigenvalues: ",fake_top_eigenvalues)

noise_inputs, noise_targets = noise_inputs.cuda(), noise_targets.cuda()
noise_hessian_comp = hessian(model, criterion, data=(noise_inputs,noise_targets), dataloader=None, cuda=True)
noise_top_eigenvalues, noise_top_eigenvector = noise_hessian_comp.eigenvalues(top_n=2)
print("noise_top_eigenvalues: ",noise_top_eigenvalues)


def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def get_imgs(orig,  perb, direction, alpha):
    perb_list = []
    for m_orig, m_perb, d in zip(orig, perb, direction):
        
        m_perb.data = m_orig.data + alpha * d
        perb_list.append(m_perb)
    perbs = torch.cat(perb_list,0)
    return perbs

def init():
    #for ax, data in zip(axs, [data1, data2]):
    ydata = "ε_2"
    ax.set_xlabel("ε_1")
    ax.set_ylabel(ydata)
    ax.set_title("Loss")
    
    ax.scatter(loss_list[:,0], loss_list[:,1], loss_list[:,2], cmap="inferno", s=5, alpha=0.5)
    
    return fig,

def animate(i):
    axs[0].view_init(elev=30., azim=i)
    #axs[1].view_init(elev=30., azim=i)
    return fig,


lams1 = np.linspace(-0.5, 0.5, 31).astype(np.float32)
lams2 = np.linspace(-0.5, 0.5, 31).astype(np.float32)

loss_list = [] #real
loss_list2 = [] #biggan
loss_list3 = []

# create a copy of the model
#model_perb1 = ptcv_get_model("resnet20_cifar10", pretrained=True)
#model_perb1.eval()
#model_perb1 = model_perb1.cuda()

data_preb1 = inputs
data_preb2 = inputs
data_preb1=data_preb1.cuda()
data_preb2=data_preb2.cuda()


#BigGAN image
data2_preb1 = fake_inputs
data2_preb2 = fake_inputs
data2_preb1=data2_preb1.cuda()
data2_preb2=data2_preb2.cuda()

data3_preb1 = noise_inputs
data3_preb2 = noise_inputs
data3_preb1=data3_preb1.cuda()
data3_preb2=data3_preb2.cuda()

#model_perb2 = ptcv_get_model("resnet20_cifar10", pretrained=True)
#model_perb2.eval()
#model_perb2 = model_perb2.cuda()
#model2_perb1 = ptcv_get_model("sepreresnet20_cifar10", pretrained=True)
#model2_perb1.eval()
#model2_perb1 = model2_perb1.cuda()
#model2_perb2 = ptcv_get_model("sepreresnet20_cifar10", pretrained=True)
#model2_perb2.eval()
#model2_perb2 = model2_perb2.cuda()

for lam1 in lams1:
    for lam2 in lams2:
        #model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam1)
        #model_perb2 = get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
        #print("data preb1 shape :",data_preb1.shape) #1 x 3 x 32 x 32
        #print("top eigenvector :",top_eigenvalues[0]) #62.62
        #print("lam 1 : ",lam1) #-0.5
        
        data_perb1 = get_imgs(inputs, data_preb1, top_eigenvector[0], lam1)
        data_perb2 = get_imgs(data_preb1, data_preb2, top_eigenvector[1], lam2)
        out_perb,_,_,_,_,_ = model(data_perb2)
        
        loss_list.append((lam1, lam2, criterion(out_perb, targets).item()))
        
        #BigGAN losss
        
        data2_preb1 = get_imgs(fake_inputs, data2_preb1, fake_top_eigenvector[0], lam1)
        data2_preb2 = get_imgs(data2_preb1, data2_preb2, fake_top_eigenvector[1], lam2)
        out_perb2,_,_,_,_,_ = model(data2_preb2)
        loss_list2.append((lam1, lam2, criterion(out_perb2, fake_targets).item()))
        #model2_perb1 = get_params(model2, model2_perb1, top_eigenvector2[0], lam1)
        #model2_perb2 = get_params(model2_perb1, model2_perb2, top_eigenvector2[1], lam2)
        #loss_list2.append((lam1, lam2, criterion(model2_perb2(inputs), targets).item()))   

        data3_preb1 = get_imgs(noise_inputs, data3_preb1, noise_top_eigenvector[0], lam1)
        data3_preb2 = get_imgs(data3_preb1, data3_preb2, noise_top_eigenvector[1], lam2)
        out_perb3,_,_,_,_,_ = model(data3_preb2)
        loss_list3.append((lam1, lam2, criterion(out_perb3, fake_targets).item()))

loss_list = np.array(loss_list) #Real
loss_list2 = np.array(loss_list2) #BigGAN
loss_list3 = np.array(loss_list3) #BigGAN
print(loss_list2)

#original code
fig = plt.figure()
landscape = fig.gca(projection='3d')
landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')
                    #cmap=cm.autumn, #cmamp = 'hot')
#landscape.plot_trisurf(loss_list2[:,0], loss_list2[:,1], loss_list2[:,2],alpha=0.8, cmap='hot')
                    #cmap=cm.autumn, #cmamp = 'hot')
landscape.set_title('Loss Landscape')
landscape.set_xlabel('ε_1')
landscape.set_ylabel('ε_2')
landscape.set_zlabel('Loss')
#z_min = min(min(loss_list[:,2]),min(loss_list2[:,2]))
#z_min = min(loss_list[:,2])
#landscape.set_zlim(z_min, z_min+13)
landscape.view_init(elev=30, azim=45)
landscape.dist = 9
plt.show()
plt.savefig('./Real_resnet34_cifar10_'+str(class_idx)+'class_'+str(batch_size))


fig2 = plt.figure()
fake_landscape = fig2.gca(projection='3d')
fake_landscape.plot_trisurf(loss_list2[:,0], loss_list2[:,1], loss_list2[:,2],alpha=0.8, cmap='hot')
                    #cmap=cm.autumn, #cmamp = 'hot')

fake_landscape.set_title('Loss Landscape')
fake_landscape.set_xlabel('ε_1')
fake_landscape.set_ylabel('ε_2')
fake_landscape.set_zlabel('Loss')
fake_landscape.view_init(elev=30, azim=45)
fake_landscape.dist = 9
plt.show()
plt.savefig('./Generator_resnet34_cifar10_'+str(class_idx)+'class_'+str(batch_size))

fig4 = plt.figure()
noise_landscape = fig4.gca(projection='3d')
noise_landscape.plot_trisurf(loss_list3[:,0], loss_list3[:,1], loss_list3[:,2],alpha=0.8, cmap='hot')
                    #cmap=cm.autumn, #cmamp = 'hot')

noise_landscape.set_title('Loss Landscape')
noise_landscape.set_xlabel('ε_1')
noise_landscape.set_ylabel('ε_2')
noise_landscape.set_zlabel('Loss')
noise_landscape.view_init(elev=30, azim=45)
noise_landscape.dist = 9
plt.show()
plt.savefig('./noise_resnet34_cifar10_'+str(class_idx)+'class_'+str(batch_size))

fig3 = plt.figure()
landscape = fig3.gca(projection='3d')
landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')
                    #cmap=cm.autumn, #cmamp = 'hot')
landscape.plot_trisurf(loss_list2[:,0], loss_list2[:,1], loss_list2[:,2],alpha=0.8, cmap='hot')
                    #cmap=cm.autumn, #cmamp = 'hot')
                    
landscape.plot_trisurf(loss_list3[:,0], loss_list3[:,1], loss_list3[:,2],alpha=0.8, cmap='cool')
                    #cmap=cm.autumn, #cmamp = 'hot')
landscape.set_title('Loss Landscape')
landscape.set_xlabel('ε_1')
landscape.set_ylabel('ε_2')
landscape.set_zlabel('Loss')
#z_min = min(min(loss_list[:,2]),min(loss_list2[:,2]))
#z_min = min(loss_list[:,2])
#landscape.set_zlim(z_min, z_min+13)
landscape.view_init(elev=30, azim=45)
landscape.dist = 9
plt.show()
plt.savefig('./Real_Fake_resnet34_cifar10_'+str(class_idx)+'class_'+str(batch_size))

#ax = fig.add_subplot(111)
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=360, interval=20, blit=True)
#anim.save('./mpl3d_scatter.gif', fps=30)

#plt.show()
'''
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={"projection":"3d"})
fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
for ax in axs:
    ydata = "Y_real" if ax == axs[0] else "Y_fake"
    ax.set_xlabel("X_real", fontdict=fontlabel, labelpad=16)
    ax.set_ylabel(ydata, fontdict=fontlabel, labelpad=16)
    ax.set_title("Loss", fontdict=fontlabel)
    ax.view_init(elev=30., azim=120)

    landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')
                    #cmap=cm.autumn, #cmamp = 'hot')
    landscape.plot_trisurf(loss_list2[:,0], loss_list2[:,1], loss_list2[:,2],alpha=0.8, cmap='hot')
                    #cmap=cm.autumn, #cmamp = 'hot')
'''