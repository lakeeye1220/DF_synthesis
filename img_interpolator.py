import itertools
import numpy as np
import random
import time
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from torch import optim
from torchvision.utils import save_image
from utils import *
from torch.autograd import Variable
from resnet import ResNet34
import numpy as np
import math
import random
import copy
import torch.nn as nn
from transformers import pipeline
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
from skimage.util import random_noise
import torchvision.models as models

import torchvision.transforms as transforms


def denormalize(image_tensor, dataset):
    channel_num = 0
    if dataset == 'cifar':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        channel_num = 3

    #elif dataset == 'imagenet':
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        channel_num = 3

    for c in range(channel_num):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c]*s+m, 0, 1)

    return image_tensor


transform = transforms.Compose([transforms.ToTensor()
                                ])


data_root = ['PII_CNN','PII_ViT','CIFAR10_abs_cat_ResNet','CIFAR10_abs_cat_ViT','CIFAR100_ResNet34_abs_1','CIFAR100_ResNet34_abs_2',
             'CIFAR100_vit_abs_1','CIFAR100_vit_abs_2','CIFAR100_vit_abs_3','CIFAR100_vit_abs_4','CIFAR100_vit_abs_5']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



final_dir_names = ['PII_CNN','PII_ViT','CIFAR10_abs_cat_ResNet','CIFAR10_abs_cat_ViT','CIFAR100_ResNet34_abs_1','CIFAR100_ResNet34_abs_2',
             'CIFAR100_vit_abs_1','CIFAR100_vit_abs_2','CIFAR100_vit_abs_3','CIFAR100_vit_abs_4','CIFAR100_vit_abs_5']



with torch.no_grad():
    for j, final_dir_ in enumerate(data_root):
        data_test = torchvision.datasets.ImageFolder(root= final_dir_, transform=transform)
        data_loader = torch.utils.data.DataLoader(data_test, batch_size=1)

        for data in data_loader:
            images, labels = data[0].to(device), data[1]
            final_dir = 'Fake_norm_'+ final_dir_names[j] +'/'+str(labels.data)+'/'

            images = nn.functional.interpolate(
            images, size=32 #Flower 224, CelebA 128 
            )
            for id in range(images.shape[0]):
                image = images[id].reshape(3,32,32)
                # image_np = image.data.cpu().numpy()
                # image_denorm = denormalize(image,'cifar')

                if not os.path.exists(final_dir):
                    os.makedirs(final_dir)

                vutils.save_image(image,os.path.join(final_dir,'output_{}'.format(id))+'.png',normalize=True,scale_each=True,nrow=1)
     



# # final_dir_norm = 'Fake_'+final_dir+'/'+str(target_class)+'/'
# # final_dir_denorm = 'Fake_'+final_dir+'/'+'denorm'+'/'+str(target_class)+'/'
# img_len_per_class = img_len/total_class
# num_generations = int(img_len_per_class/batch_size) #flower 250 cifar100 500 
# for i in range(num_generations):
#     zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
#     # if optim_comps["use_noise_layer"]:
#     #     noise_layer = nn.Linear(dim_z,dim_z)
#     #     noise_layer.load_state_dict(torch.load(f"{class_dir}/{target_class}_noise_layer.pth"))

#     #     noise_layer = noise_layer.cuda()
#     #     zs = noise_layer(zs)

#     with torch.no_grad():
#         repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
#         print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
#         gan_images_tensor = G(zs, repeat_class_embedding)
#         resized_images_tensor = nn.functional.interpolate(
#             gan_images_tensor, size=img_size #Flower 224, CelebA 128
#             )
#     targets = torch.LongTensor([target_class] * batch_size)
#     images = resized_images_tensor.data.clone()

#     if 'cifar' in dataset:
#         images = nn.functional.interpolate(
#             images, size=32 #Flower 224, CelebA 128 
#         )
#     else:
#             images = nn.functional.interpolate(
#             images, size=img_size
#         )
#     for id in range(images.shape[0]):
#         class_id = str(targets[id].item()).zfill(2)
#         if 'cifar' in dataset:
#             image = images[id].reshape(3,32,32)
#         else:
#             image = images[id].reshape(3,img_size,img_size)
#         image_nodenorm = image.data.clone()
#         image_denorm = denormalize(image,dataset)
#         image_np_denorm = image_denorm.data.cpu().numpy()
#         image_np_nodenorm = images[id].data.cpu().numpy()
#         pil_images_denorm = torch.from_numpy(image_np_denorm)
#         pil_images_nodenorm = torch.from_numpy(image_np_nodenorm)

#         if not os.path.exists(final_dir):
#             os.makedirs(final_dir)
#         # if not os.path.exists(final_dir_norm):
#         #     os.makedirs(final_dir_norm)
#         # if not os.path.exists(final_dir_denorm):
#         #     os.makedirs(final_dir_denorm)
        
#         vutils.save_image(image_denorm,os.path.join(final_dir,'{}_output_{}'.format(i,id))+'.png',normalize=True,scale_each=True,nrow=1)