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

batch_size=256

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

# biggan loader
d_path = './Fake_20230530_030743_cifar10_1'
dataset = datasets.ImageFolder(d_path, transform=transform_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"All Data Loaded, {len(dataset)} loaded")

stat_path = "./cifar10_npz/cifar10_norm/cifar10.train.npz"

(IS, IS_std), FID = get_inception_score_and_fid(
        loader,
        stat_path,
        use_torch=True,
        verbose=True)

print(d_path, ": ", IS, FID)