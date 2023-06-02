# embedding 불러온 후 이미지 생성 후 IS, FID score를 계산하던 옛날 코드입니다.

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import numpy as np
import sys
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from gan_metrics_master.pytorch_gan_metrics.utils import get_inception_score_and_fid
from tqdm import tqdm

print(sys.path)
import BigGAN
from utils import get_config

class_idx = 8
prefix="./exprs/TV/cifar10_"+str(class_idx)
save_prefix="./fake_images/"+str(class_idx)+"/"
batch_size=40
num_generations=125

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


def save_final_images(images,targets,num_generations,save_prefix):
    #save_pth = os.path.join(save_prefix,'final_images/s{}'.format(class_id))
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    images = images.data.clone()
    for id in range(images.shape[0]):
        class_id = str(targets[id].item()).zfill(2)
        image = images[id].reshape(3,32,32)
        image = denormalize(image,'cifar10')
        image_np = images[id].data.cpu().numpy()
        pil_images = torch.from_numpy(image_np)
        
        vutils.save_image(image,os.path.join(save_prefix,'{}_output_{}'.format(num_generations,id))+'.png',normalize=True,scale_each=True,nrow=1)

print("Loading the BigGAN generator model...", flush=True)


resolution = 256
config = get_config(resolution)
G = BigGAN.Generator(**config)
G.load_state_dict(
    torch.load("./pretrained_weights/biggan_256_weights.pth"), strict=False
)
G = nn.DataParallel(G).to('cuda')
G = G.cuda()
G.eval()

class_embedding = np.load(f"{prefix}/0.npy")
class_embedding = torch.tensor(class_embedding)
print("class embedding: ",class_embedding.shape)
#print(class_embedding[0])
#print(class_embedding[1])
z_num = batch_size
# repeat_class_embedding = class_embedding.repeat(int(z_num/4), 1).cuda()
repeat_class_embedding = class_embedding.repeat(z_num, 1).cuda()
print("repeat class embedding: ",repeat_class_embedding.shape)

for i in tqdm(range(num_generations), desc="num generations"):
    zs = torch.randn((z_num,140),requires_grad=False).cuda()

    # noise_layer = nn.Linear(140,140)
    # noise_layer.load_state_dict(torch.load(f"{prefix}/0_noise_layer.pth"))

    # noise_layer = noise_layer.cuda()
    # zs = noise_layer(zs)
    with torch.no_grad():
        gan_images_tensor = G(zs, repeat_class_embedding)
        resized_images_tensor = nn.functional.interpolate(
            gan_images_tensor, size=32 #Flower 224, CelebA 128
            )
    targets = torch.LongTensor([class_idx] * batch_size).cuda()
    save_final_images(resized_images_tensor,targets,i,save_prefix)

batch_size=256

transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

# biggan loader
dataset = datasets.ImageFolder('./fake_images', transform=transform_train)
subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
loaders = {target: DataLoader(subsets,batch_size=batch_size, shuffle=True) for target, subsets in subsets.items()}

loader = loaders[class_idx]
stat_path = "cifar10_npz/cifar10_norm/statistics_class" + f"{class_idx}.npz"

print("All Data Loaded")

(IS, IS_std), FID = get_inception_score_and_fid(
        loader,
        stat_path,
        use_torch=True,
        verbose=True)

print(IS, IS_std, FID)