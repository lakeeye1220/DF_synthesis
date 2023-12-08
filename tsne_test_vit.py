from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from resnet_cifar3_cifar10 import ResNet34
import ssl
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import datasets
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel
        






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
                                transforms.Resize((224,224)),
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


data_set_root_list = ['Fake_ours_zo2m_vit_cifar10_mean_thres_0.98_lr_0.01_tv_1.3e_3_5seed_Tnearest',
                        'Fake_ours_zo2m_vit_cifar10_mean_thres_0.98_lr_0.01_tv_1.3e_3_5seed_logitnormnearest',
                        'Fake_ours_zo2m_vit_cifar10_mean_thres_0.98_lr_0.01_tv_1.3e_3_5seed_vanialaCEnearest'
                      #'Fake_ours_zo2m_vit_b16_cifar10_1k_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_1113'
                    #   'Fake_ours_abs_resnet34_thres0.98_cifar10_iter5_LGAT',
                    #   'Fake_ours_abs_vit_seed171_thres0.98_cifar10_iter5_vanilaCE',
                    #   'Fake_ours_abs_vit_seed171_thres0.98_cifar10_iter5_logitnorm',
                    #   'Fake_ours_abs_vit_seed171_thres0.98_cifar10_iter5_LGAT'
                      ]




# feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
# model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')

# model = 
def extract_embeddings(model, feature_extractor, image_name="image"):
    """
    Utility to compute embeddings.
    Args:
        model: huggingface model
        feature_extractor: huggingface feature extractor
        image_name: name of the image column in the dataset
    Returns:
        function to compute embeddings
    """
    device = model.device
    def pp(batch):
        images = batch[image_name]
        inputs = feature_extractor(
            images=[x.convert("RGB") for x in images], return_tensors="pt"
        ).to(device)
        embeddings = model(**inputs).last_hidden_state[:, 0].detach().cpu().numpy()
        return {"embedding": embeddings}
    return pp

def huggingface_embedding(
    df,
    image_name="image",
    modelname="google/vit-base-patch16-224",
    batched=True,
    batch_size=24,
):
    """
    Compute embeddings using huggingface models.
    Args:
        df: dataframe with images
        image_name: name of the image column in the dataset
        modelname: huggingface model name
        batched: whether to compute embeddings in batches
        batch_size: batch size
    Returns:
        new dataframe with embeddings
    """
    # initialize huggingface model
    feature_extractor = AutoFeatureExtractor.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname, output_hidden_states=True)
    model.eval()
    # create huggingface dataset from df
    dataset = datasets.Dataset.from_pandas(df).cast_column(image_name, datasets.Image())
    # compute embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device), feature_extractor, image_name)
    updated_dataset = dataset.map(extract_fn, batched=batched, batch_size=batch_size)
    df_temp = updated_dataset.to_pandas()
    df_emb = pd.DataFrame()
    df_emb["embedding"] = df_temp["embedding"]
    return df_emb






# model = tor
# logit, softmax 쓸 떄는 지울 것 !
# model.linear = Identity()
# del model.fc

for j, data_root in enumerate(data_set_root_list):
    # data_test = torchvision.datasets.ImageFolder(root= data_root, transform=transform)

# original_data = torchvision.datasets.CIFAR10(root='/home/jihwan/DF_synthesis/data/CIFAR10', train=False,
#                                        download=True, transform=transform)

    dataset = datasets.load_dataset("imagefolder", data_dir=data_root,num_proc=2)

    df = dataset["train"].to_pandas()

# data_loader = torch.utils.data.DataLoader(original_data, batch_size=16)

    # data_loader = torch.utils.data.DataLoader(data_test, batch_size=16)
    embeddings_df = huggingface_embedding(
        df,
        modelname="nateraw/vit-base-patch16-224-cifar10",
        batched=True,
        batch_size=24,
    )

    # actual = []
    # deep_features = []
    df["embedding"] = embeddings_df["embedding"]
    # breakpoint()
    df["label_str"] = df["label"].apply(lambda x: dataset["train"].features["label"].int2str(x))


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

    tsne = TSNE(n_jobs=4) # 사실 easy 함 sklearn 사용하니..
    # cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    # Convert the list of embeddings from the DataFrame into a NumPy array
    embeddings = np.stack(df['embedding'].values)

    # Apply t-SNE transformation
    transformed_embeddings = tsne.fit_transform(embeddings)

    # actual = np.array(actual)


    # Assuming 'label_str' column contains string labels for each embedding
    unique_labels = df['label_str'].unique()
    plt.figure(figsize=(10, 10))
    # cifar = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # for i, label in zip(range(10)):
    #     idx = np.where(actual == i)
    #     plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.')
    for label in unique_labels:
        # Filter indices for each label
        indices = df['label_str'] == label
        plt.scatter(transformed_embeddings[indices, 0], transformed_embeddings[indices, 1], label=label)



    # plt.legend()
    plt.savefig(f"vit_multiseed_tnse_{j}.png",dpi=300)




