import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import numpy as np
import sys
import os

sys.path.append('/home/work/yujin/DF_synthesis')
print(sys.path)
import BigGAN
from utils2 import get_config

class_idx = 9
prefix="../DF_synthesis/resnet34_cifar10_CE_entropy_clip05_"+str(class_idx)
save_prefix="./fake_images/"+str(class_idx)+"/"
batch_size=128
num_generations=10

def save_final_images(images,targets,num_generations,save_prefix):
    images = images.data.clone()
    for id in range(images.shape[0]):
        class_id = str(targets[id].item()).zfill(2)
        image = images[id].reshape(3,32,32)
        image_np = images[id].data.cpu().numpy()
        pil_images = torch.from_numpy(image_np)

        #save_pth = os.path.join(save_prefix,'final_images/s{}'.format(class_id))
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)

        vutils.save_image(image,os.path.join(save_prefix,'{}_output_{}'.format(num_generations,id))+'.png',normalize=True,scale_each=True,nrow=1)

print("Loading the BigGAN generator model...", flush=True)

resolution = 256
config = get_config(resolution)
G = BigGAN.Generator(**config)
G.load_state_dict(
    torch.load("../DF_synthesis/pretrained_weights/biggan_256_weights.pth"), strict=False
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
repeat_class_embedding = class_embedding.repeat(int(z_num/4), 1).cuda()
print("repeat class embedding: ",repeat_class_embedding.shape)

for i in range(num_generations):
    zs = torch.randn((z_num,140),requires_grad=False).cuda()

    noise_layer = nn.Linear(140,140)
    noise_layer.load_state_dict(torch.load(f"{prefix}/0_noise_layer.pth"))

    noise_layer = noise_layer.cuda()
    zs = noise_layer(zs)
    print("zs shape :",zs.shape)
    with torch.no_grad():
        gan_images_tensor = G(zs, repeat_class_embedding)
        resized_images_tensor = nn.functional.interpolate(
            gan_images_tensor, size=32 #Flower 224, CelebA 128
            )
    targets = torch.LongTensor([class_idx] * batch_size).cuda()
    save_final_images(resized_images_tensor,targets,i,save_prefix)