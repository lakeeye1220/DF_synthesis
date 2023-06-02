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

dataset = datasets.ImageFolder('./Fake_cifar10(Trans201)', transform=transform_train)
subsets = {target: Subset(dataset, [i for i, (x, y) in enumerate(dataset) if y == target]) for _, target in dataset.class_to_idx.items()}
loaders = {target: DataLoader(subsets, batch_size=batch_size, shuffle=True) for target, subsets in subsets.items()}

for class_idx in range(0, 10):
        # biggan loader
        loader = loaders[class_idx]
        stat_path = "cifar10_npz/cifar10_norm/statistics_class" + f"{class_idx}.npz"

        (IS, IS_std), FID = get_inception_score_and_fid(
                loader,
                stat_path,
                use_torch=True,
                verbose=True)

        print(f"class {class_idx} IS: {IS}, FID: {FID}")