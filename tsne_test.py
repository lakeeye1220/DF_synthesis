from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from resnet_cifar3_cifar10 import ResNet34
import ssl
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

        






ssl._create_default_https_context = ssl._create_unverified_context

os.environ['CURL_CA_BUNDLE'] = ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# load data - cifar10

transform = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])


# data_set_root_list = ['/home/jihwan/DF_synthesis/Fake_ours_abs_resnet34_seed161_thres0.98_cifar10_10k_vanilaCE',
#                       '/home/jihwan/DF_synthesis/Fake_ours_abs_resnet34_seed161_thres0.98_cifar10_10k_logitnorm',
#                       '/home/jihwan/DF_synthesis/Fake_ours_abs_resnet34_seed161_thres0.98_cifar10_10k_LGAT',
#                       '/home/jihwan/DF_synthesis/Fake_ours_abs_vit_seed10260000_thres0.98_cifar10_10k_vanilaCE',
#                       '/home/jihwan/DF_synthesis/Fake_ours_abs_vit_seed10260000_thres0.98_cifar10_10k_logitnorm',
#                       '/home/jihwan/DF_synthesis/Fake_ours_abs_vit_seed10260000_thres0.98_cifar10_10k_LGAT',
#                       ]


data_set_root_list = ['Fake_90over_10seed_th098_resnet34_cifar10_T_nearest'
                    #   'Fake_ours_abs_resnet34_thres0.98_cifar10_iter5_LGAT',
                    #   'Fake_ours_abs_vit_seed171_thres0.98_cifar10_iter5_vanilaCE',
                    #   'Fake_ours_abs_vit_seed171_thres0.98_cifar10_iter5_logitnorm',
                    #   'Fake_ours_abs_vit_seed171_thres0.98_cifar10_iter5_LGAT'
                      ]

# 코드 진행방식
# n번 class와, 원본 CIFAR10 특정 클래스에 대한 t-SNE를 진행해야함
# 이때, 000번 클래스는 0epoch_._.png부터 9epoch_._.png로 구성이 되어 있음
# 그래서 epoch별로 자르고, CIFAR10 특정 클래스 5000장 이렇게 t_SNE를 구성하고자 함!





# model = torchvision.models.resnet34(pretrained=True).to(device)
model = ResNet34(num_classes=10).to(device)
model.load_state_dict(torch.load('/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar10_resnet34_9557.pt'))
model.eval()

# feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
# model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')

# model = 

# Get CIFAR10 dataset
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform = transform)



# model = tor
# logit, softmax 쓸 떄는 지울 것 !
model.linear = Identity()
# del model.fc


# Get custom image dataset.
def load_custom_images(root_dir, epoch, cifar10_class):
    images = []
    dir_path = os.path.join(root_dir, str(cifar10_class).zfill(3)) # e.g., '000'
    for filename in os.listdir(dir_path):
        if filename.startswith(f'{epoch}epoch'):
            img_path = os.path.join(dir_path, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
    return images


def extract_features(dataset, model,transform,device):
    features = []
    for img in dataset:
        img_tensor = transform(img).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            feature = model(img_tensor.to(device))
        features += feature.cpu().numpy().tolist()
    ret_features = np.array(features)
    
    return ret_features

def extract_features_tensor(dataset, model,transform,device):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    features = []
    for data in data_loader:
        # img_tensor = transform(img).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            feature = model(data[0].to(device))
        features += feature.cpu().numpy().tolist()
    ret_features = np.array(features)
    return ret_features



def plot_tsne(features1, features2, epoch, cifar10_class):
    tsne = TSNE(n_components=2, random_state=42)
    # combined_features = features1 + features2
    combined_features = np.vstack((features1, features2))
    embeddings = tsne.fit_transform(combined_features)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:len(features1), 0], embeddings[:len(features1), 1], color='r')
    plt.scatter(embeddings[len(features1):, 0], embeddings[len(features1):, 1], color='b')
    plt.title(f't-SNE Visualization - Epoch {epoch} for class {cifar10_class}')
    plt.savefig(f"tsne_viszualization_{epoch}_{cifar10_class}.png",dpi=300)
    # plt.show()



for j, data_root in enumerate(data_set_root_list):
    # data_test = torchvision.datasets.ImageFolder(root= data_root, transform=transform)

# original_data = torchvision.datasets.CIFAR10(root='/home/jihwan/DF_synthesis/data/CIFAR10', train=False,
#                                        download=True, transform=transform)



# data_loader = torch.utils.data.DataLoader(original_data, batch_size=16)

    # data_loader = torch.utils.data.DataLoader(data_test, batch_size=16)



    # actual = []
    # deep_features = []


    # model.eval() # resnet34
    # with torch.no_grad():
    #     for data in data_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         features = model(images) # 512 차원
    #         # breakpoint()

    #         # norm = torch.norm(features,p=2,dim=-1,keepdim=True) + 1e-7
    #         # logit_norm = torch.div(features,norm)#/torch.Tensor([0.02]).cuda()
    #         # for softamx
    #         # softmax = torch.nn.functional.softmax(features)
    #         # features
    #         # deep_features += softmax.cpu().numpy().tolist()
    #         deep_features += features.cpu().numpy().tolist()
    #         actual += labels.cpu().numpy().tolist()

    # tsne = TSNE(n_jobs=4) # 사실 easy 함 sklearn 사용하니..
    # cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    # actual = np.array(actual)

    # plt.figure(figsize=(10, 10))
    # cifar = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # for i, label in zip(range(10), cifar):
    #     idx = np.where(actual == i)
    #     plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)

    # # plt.legend()
    # plt.savefig(f"multiseed_tnse_resnet_logitnorm_{j}.png",dpi=300)
    for target_label in [1,2,3,4,5,6,7,8,9]:
        for epoch in range(10):
            custom_images = load_custom_images(data_root, epoch, target_label)
            # cifar10_images = [image for image, label in cifar10_dataset if label == cifar10_class] # Adjust the number as needed
            target_indices = [i for i, (_, label) in enumerate(cifar10_dataset) if label == target_label]
            target_subset = Subset(cifar10_dataset, target_indices)


            custom_features = extract_features(custom_images, model, transform,device)
            
            cifar10_features = extract_features_tensor(target_subset, model, transform,device)
            # breakpoint()
            plot_tsne(custom_features, cifar10_features, epoch, target_label)





