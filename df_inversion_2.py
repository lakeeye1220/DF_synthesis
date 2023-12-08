import BigGAN
import itertools
import numpy as np
import random
import time
import torch.nn.functional as F
import torch.nn as nn
import yaml

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
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from resnet_cifar3_cifar10 import ResNet34
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import matplotlib
import glob
import csv
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import umap
from sklearn.preprocessing import StandardScaler

os.environ['CURL_CA_BUNDLE'] = ''

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def do_3d_projection(self, renderer=None):
        # Project 3d data space to 2d data space
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        # return super(Arrow3D, self).do_3d_projection(renderer)
        return np.min(zs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_classes=None):
        super(CustomImageFolder, self).__init__(root, transform=transform)
        if target_classes:
            self.samples = [s for s in self.samples if s[0].split('/')[-2] in target_classes]
            self.targets = [s[1] for s in self.samples]
            self.classes = target_classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}



def change_seed():

    seed_z = np.random.randint(1000000)
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)
    return seed_z

def get_initial_embeddings(
    resolution,
    init_method,
    model_name,
    init_num,
    min_clamp,
    max_clamp,
    dim_z,
    G,
    net,
    feature_extractor,
    model,
    target_class,
    noise_std,
    img_size
):
    class_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
    class_embeddings = torch.from_numpy(class_embeddings)
    print("class embedding shape: ",class_embeddings.shape)
    embedding_dim = class_embeddings.shape[-1]

    if init_method == "mean":
        mean_class_embedding = torch.mean(class_embeddings, dim=0)
        std_class_embedding = torch.std(class_embeddings, dim=0)
        print("mean class embedding : ",mean_class_embedding.shape)
        init_embeddings = torch.normal(mean=mean_class_embedding, std=std_class_embedding)
        print("init embedding : ",init_embeddings.shape)

    elif init_method == "top":
        class_embeddings_clamped = torch.clamp(class_embeddings, min_clamp, max_clamp)
        num_samples = 10
        avg_list = []
        final_z = torch.randn((num_samples, dim_z), requires_grad=False)
        for i in range(1000):
            class_embedding = class_embeddings_clamped[i]
            repeat_class_embedding = class_embedding.repeat(num_samples, 1)
            

            with torch.no_grad():
                gan_images_tensor = G(final_z, repeat_class_embedding)
                # resized_images_tensor = nn.functional.interpolate(
                #     gan_images_tensor, size=224
                # )
                resized_images_tensor = nn.functional.interpolate(
                    gan_images_tensor, size=img_size
                )
                if 'vit' in model_name:
                    model = model.to(device)
                    outputs = model(resized_images_tensor)
                    pred_logits = outputs.logits

                elif 'cartoon' in model_name:
                    model = model.to(device)
                    outputs = model(resized_images_tensor)
                    pred_logits = outputs.logits
                    print("pred_logits shape : ",pred_logits.shape)

                else:
                    pred_logits,_,_,_,_,_ = net(resized_images_tensor)
                    # print(resized_images_tensor.shape)
                    # breakpoint()
                    # pred_logits = net(resized_images_tensor)
            
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()
            avg_list.append(avg_target_prob)

        avg_array = np.array(avg_list)
        sort_index = np.argsort(avg_array)

        print(f"The top {init_num} classes: {sort_index[-init_num:]}")

        init_embeddings = class_embeddings[sort_index[-init_num:]]

    elif init_method == "random":
        index_list = random.sample(range(1000), init_num)
        print(f"The {init_num} random classes: {index_list}")
        init_embeddings = class_embeddings[index_list]

    elif init_method == "target":
        target_class = 404
        init_embeddings = (
            class_embeddings[target_class].unsqueeze(0).repeat(init_num, 1)
        )
        init_embeddings += torch.randn((init_num, embedding_dim)) * noise_std

    return init_embeddings

def decompose_latent(optim_comps,target,G,net,z_num,dim_z,final_dir,device,model_name, class_dir,seed_z):

    # Load Class Embedding
    # 파일 패턴에 맞는 파일 목록 가져오기
    file_list = sorted(glob.glob(f"{class_dir}/{target}_embedding_*_{seed_z}.npy"))

    # 파일 목록을 순회하면서 각 파일을 로드
    embeddings = []
    for file in file_list:
        embeddings_file = np.load(file)
        embeddings.append(embeddings_file)



    repeat_optim_embedding = optim_comps["class_embedding"].repeat(1, 1).to(device)
    class_embeddings = np.load(f"biggan_embeddings_256.npy")
    # Get bigGAN class embedding
    class_embeddings = torch.from_numpy(class_embeddings).to(device)

    if optim_comps["use_noise_layer"]:
        optim_comps["noise_layer"].eval()

    optim_imgs = []
    original_imgs = []
    avg_list=[]

    # zs = torch.randn((z_num, dim_z), device=device, requires_grad=False)

    # load zs
    zs_load = np.load(f"{class_dir}/{target}_z_{seed_z}.npy")
    for it in range(zs_load.shape[0]):
        zs = torch.from_numpy(zs_load[it])
        zs = zs.repeat(1,1).to(device)

        # Initialize Embeddings
        embeddings = []
        for file in file_list:
            embeddings_file = np.load(file)
            embeddings.append(embeddings_file)


        # Generate Images from final embeddings -> Get top 5 classes from ImageNet
        final_embedding = optim_comps["class_embedding"]
        repeat_embedding = final_embedding.repeat(1,1).to(device)
        # breakpoint()
        # Generate Images
        gan_embedding_images = G(zs, repeat_embedding)
        vutils.save_image(gan_embedding_images,os.path.join(final_dir,'{}_{}class_optimal_embedding_{}'.format(it,target,seed_z))+'.png',normalize=True,scale_each=True,nrow=10)

        vutils.save_image(gan_embedding_images,os.path.join(class_dir,'{}_{}class_optimal_embedding_{}'.format(it,target,seed_z))+'.png',normalize=True,scale_each=True,nrow=10)
        
        net = models.resnet34(pretrained=True).to(device)
        net.eval()
        '''
        for i in range(1000):
            class_embedding = class_embeddings[i]
            repeat_class_embedding = class_embedding.repeat(z_num, 1)
            final_z = torch.randn((z_num, dim_z), requires_grad=False)

            with torch.no_grad():
                gan_images_tensor = G(final_z, repeat_class_embedding)
                resized_images_tensor = nn.functional.interpolate(
                        gan_images_tensor, size=224
                    )
                if 'vit' in model_name:
                    model = model.to(device)
                    outputs = model(resized_images_tensor)
                    pred_logits = outputs.logits

                elif 'cartoon' in model_name:
                    model = model.to(device)
                    outputs = model(resized_images_tensor)
                    pred_logits = outputs.logits
                    print("pred_logits shape : ",pred_logits.shape)

                else:
                    pred_logits = net(resized_images_tensor)
        '''
        # Get logit
        pred_logits = net(gan_embedding_images)
        pred_probs = nn.functional.softmax(pred_logits, dim=1)
        avg_target_prob = torch.mean(pred_probs,dim=0)
        #print("pred_probs shape : ",avg_target_prob.shape) #expec: 1 x 1000
        #print("avg_target_prob shape : ",avg_target_prob)
        print("max value : ",max(avg_target_prob.cpu().detach()))
        #print("max value index : ",avg_target_prob.cpu().detach().index(max(avg_target_prob.cpu().detach())))
        #avg_list.append(avg_target_prob.item())

        #avg_array = np.array(avg_list)
        #print(avg_array)
        
        # Get top 5 classes
        sort_index = np.argsort(avg_target_prob.cpu().detach())
        print(sort_index)
        top5_indices_prob = sort_index[-5:]

        cos = nn.CosineSimilarity(dim=0)
        # Get Cos similarity
        optim_embedding = torch.zeros_like(final_embedding).to(device)
        cossim_list = []
        # breakpoint()
        for i in range(1000):
            # cuz final_embedding shape:[1,128]
            # cuz class_embeddings[i] shape:[128]
            cos_out = cos(final_embedding.squeeze(0),class_embeddings[i])
            #print("cosine similarity: ",cos_out.item())
            cossim_list.append(cos_out.item())
        cos_sort_index = np.argsort(cossim_list)

        # Get top 5 cos similarity indices
        top5_indices_cos = cos_sort_index[-5:]

        # Get Low 5 cos similarity indices
        low5_indices_cos = cos_sort_index[:5]

        #cossim_list = cossim_list.sort()
        print(cossim_list)

        print("cosine similarity sort top 5: ",top5_indices_cos)


        print("cosine similarity sort Low 5: ",low5_indices_cos)

        # torch.from_numpy(class_embeddings).to(device)
        # To-Do
        # 반복문으로 각 embedding에 대한 cos similarity 구함
        print('Get cos similarity between the embeddings : Max Concept Embeddings')
        # embeddings : Tensor
        cos_list_emb = []
        for i in range(len(embeddings)):
            cos_between_emb_top5 = []
            for j in top5_indices_cos:
                # cos_out = cos(embeddings[i],class_embeddings[j])
                emb_cuda = torch.from_numpy(embeddings[i]).to(device)
                # emb_cuda shape : [1,128]
                # class_embeddings : [128]
                # breakpoint()
                cos_out = cos(emb_cuda.squeeze(0),class_embeddings[j])
                cos_between_emb_top5.append(cos_out.item())
            cos_list_emb.append(cos_between_emb_top5)
        
        with open(f'{class_dir}/{it}_{target}_output_top5_cos_sim_{seed_z}.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(top5_indices_cos)
            writer.writerows(cos_list_emb)



        # To-Do
        # PCA 실험
        # print('PCA Experiment Start!')
        # pca = PCA(n_components = 2)

        # colors = ['red', 'blue', 'green', 'purple', 'orange']


        print('Get cos similarity between the embeddings : Min Concept Embeddings')
        # embeddings : Tensor
        cos_list_emb_low = []
        for i in range(len(embeddings)):
            cos_between_emb_low5 = []
            for j in low5_indices_cos:
                # cos_out = cos(embeddings[i],class_embeddings[j])
                emb_cuda = torch.from_numpy(embeddings[i]).to(device)
                # emb_cuda shape : [1,128]
                # class_embeddings : [128]
                # breakpoint()
                cos_out = cos(emb_cuda.squeeze(0),class_embeddings[j])
                cos_between_emb_low5.append(cos_out.item())
            cos_list_emb_low.append(cos_between_emb_low5)

        with open(f'{class_dir}/{it}_{target}_output_low5_cos_sim_{seed_z}.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(low5_indices_cos)
            writer.writerows(cos_list_emb_low)



        # 100 * 140차원의 Random z 선언
        random_latent = np.random.randn(10, dim_z)
        

        # 각 embedding 마다 더해줌

        # 더해줄 때마다 concatenate

        top5_class_emb_ct = []
        for j in top5_indices_cos:
            # clm_emb.shape() = (128,)
            cls_emb = class_embeddings[j].cpu().numpy()
            reshaped_cls_emb = cls_emb.reshape(1, 128)
            expanded_cls_emb = np.repeat(reshaped_cls_emb, 10, axis=0)
            result = np.concatenate((expanded_cls_emb, random_latent), axis=1)

            top5_class_emb_ct.append(result)

        # top5_class_emb_ct


        top5_class_emb_ct = np.array(top5_class_emb_ct).reshape(-1,dim_z+128)
        
        # Remainig for T-SNE
        top5_class_emb = []
        for j in top5_indices_cos:
            cls_emb = class_embeddings[j].cpu().numpy()
            top5_class_emb.append(cls_emb)
        
        remaining_class_emb = [class_embeddings[i].cpu().numpy() for i in range(len(class_embeddings)) if i not in top5_indices_cos]





         # Remainig for T-SNE
        low5_class_emb = []
        for j in low5_indices_cos:
            cls_emb = class_embeddings[j].cpu().numpy()
            low5_class_emb.append(cls_emb)
        
        # remaining_low_class_emb = [class_embeddings[i].cpu().numpy() for i in range(len(class_embeddings)) if i not in low5_class_emb]
        

        # # Changed PCA Sequence
        # transform_embeddings = pca.fit_transform(embeddings)
        # plt.figure(figsize=(10, 6))
        # plt.scatter(transform_embeddings[:, 0], transform_embeddings[:, 1], c=range(len(transform_embeddings)),cmap='jet')
        # plt.plot(transform_embeddings[:, 0], transform_embeddings[:, 1], '-o', color='gray', alpha=0.5)
        # plt.colorbar().set_label('Sequence')

        # # class embedding PCA
        # for idx, data in enumerate(top5_class_emb):
        #     reshaped_data = np.array(data).reshape(1,-1)
        #     transformed_data = pca.transform(reshaped_data)
        #     plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color=colors[idx], label=top5_indices_cos[idx], s=50)


        
        
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.title('PCA Sequential Traces on 128-dimensional data for 5 Groups')
        # plt.legend()
        # plt.show()
        # plt.savefig(f'{class_dir}/{target}_PCA_top5_embeddings_{seed}.png')

        # print("image generated finished!")

        print('PCA Experiment Start!')
        
        # 모든 데이터를 2D 배열로 결합
        # breakpoint()
        squeezed_final_embedding = final_embedding.detach().to('cpu').numpy()


        squeezed_final_embedding_repeat = np.repeat(squeezed_final_embedding,10,axis=0)

        squeezed_final_embedding_ct = np.concatenate((squeezed_final_embedding_repeat, random_latent), axis=1)


        # breakpoint()
        squeezed_embeddings = np.squeeze(embeddings, axis = 1)
        
        squeezed_embeddings_ct = []
        for j in range(squeezed_embeddings.shape[0]):
            # clm_emb.shape() = (128,)
            # cls_emb = class_embeddings[j].cpu().numpy()
            reshaped_sq_emb = squeezed_embeddings[j].reshape(1, 128)
            expanded_sq_emb = np.repeat(reshaped_sq_emb, 10, axis=0)
            result = np.concatenate((expanded_sq_emb, random_latent), axis=1)

            squeezed_embeddings_ct.append(result)


        # squeezed_embeddings_reshape = squeezed_embeddings.resh
        squeezed_embeddings_ct = np.array(squeezed_embeddings_ct).reshape(-1,dim_z+128)
        # squeezed_embedding_ct = np.concatenate((squeezed_embeddings, random_latent), axis=1)
        # (n* 128) 차원짜리 embedding이라고 볼 수 있음
        all_data = np.vstack([top5_class_emb, squeezed_final_embedding, squeezed_embeddings,remaining_class_emb])


        # breakpoint()

        all_data_ct = np.vstack([top5_class_emb_ct, squeezed_final_embedding_ct, squeezed_embeddings_ct])


        # 2차원 PCA
        pca2D = PCA(n_components=2)
        pca2D.fit(all_data)

        # 데이터 변환
        transformed_top5_2D = pca2D.transform(top5_class_emb)
        transform_final_2D = pca2D.transform(squeezed_final_embedding)
        transform_embeddings_2D = pca2D.transform(squeezed_embeddings)
        transform_remaining_2D = pca2D.transform(remaining_class_emb)

        
        plt.figure(figsize=(10, 6))

        # 첫 번째 데이터 그룹: transformed_top5 표시
        colors = ['lightseagreen', 'yellowgreen', 'green', 'mediumseagreen', 'skyblue']
        for idx, transformed_data in enumerate(transformed_top5_2D):
            plt.scatter(transformed_data[0], transformed_data[1], color=colors[idx], label=top5_indices_cos[idx], s=20)

        plt.scatter(transform_final_2D[0][0], transform_final_2D[0][1], c='darkblue', label = 'optimal embedding', s=60,marker='*')
        plt.scatter(transform_embeddings_2D[:, 0], transform_embeddings_2D[:, 1], c=range(len(transform_embeddings_2D)), cmap='spring',s=40)
        
        # plt.plot(transform_embeddings[:, 0], transform_embeddings[:, 1], 
        #         '-o', color='gray', alpha=0.5)
        plt.colorbar().set_label('Sequence')

        for idx, transformed_data in enumerate(transform_remaining_2D):
            plt.scatter(transformed_data[0], transformed_data[1], color='grey', s=10, alpha=0.5)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA on 128-dimensional data for 5 Groups')
        plt.legend()
        plt.show()
        plt.savefig(f'{class_dir}/{it}_{target}_2D_PCA_top5_embeddings_{seed_z}.png')

        print("2d pca image generated finished!")

        plt.clf()


        pca2D_ct = PCA(n_components=2)
        pca2D_ct.fit(all_data_ct)

        # 데이터 변환
        transformed_top5_2D_ct = pca2D_ct.transform(top5_class_emb_ct)
        transform_final_2D_ct = pca2D_ct.transform(squeezed_final_embedding_ct)
        transform_embeddings_2D_ct = pca2D_ct.transform(squeezed_embeddings_ct)
        # transform_remaining_2D = pca2D.transform(remaining_class_emb)

        
        plt.figure(figsize=(10, 6))

        # 첫 번째 데이터 그룹: transformed_top5 표시
        colors = ['lightseagreen', 'yellowgreen', 'green', 'mediumseagreen', 'skyblue']
        # for idx, transformed_data in enumerate(transformed_top5_2D_ct):
        #     plt.scatter(transformed_data[0], transformed_data[1], color=colors[int(idx/100)], label=top5_indices_cos[int(idx/100)], s=70)

        plt.scatter(transform_final_2D_ct[0][0], transform_final_2D_ct[0][1], c='darkblue', label = 'optimal embedding', s=70,marker='*')
        
        plt.scatter(transformed_top5_2D_ct[:, 0], transformed_top5_2D_ct[:, 1],c='lightgrey',label='top 5 embedding',s=50,alpha=0.4)
        plt.scatter(transform_embeddings_2D_ct[:, 0], transform_embeddings_2D_ct[:, 1], c=range(len(transform_embeddings_2D_ct)), cmap='spring',s=30,alpha=0.2)
        # plt.plot(transform_embeddings[:, 0], transform_embeddings[:, 1], 
        #         '-o', color='gray', alpha=0.5)
        plt.colorbar().set_label('Sequence')

        # for idx, transformed_data in enumerate(transform_remaining_2D):
        #     plt.scatter(transformed_data[0], transformed_data[1], color='grey', s=10, alpha=0.5)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA on 128-dimensional data for 5 Groups')
        plt.legend()
        plt.show()
        plt.savefig(f'{class_dir}/{it}_{target}_2D_PCA_top5_embeddings_CT_{seed_z}.png')

        print("2d pca CT image generated finished!")

        plt.clf()



        # plt.figure(figsize=(10, 6))

        print("3d image generated start!")
        # 3차원 PCA
        pca3D = PCA(n_components=3)
        pca3D.fit(all_data)


        # 데이터 변환
        transformed_top5_3D = pca3D.transform(top5_class_emb)
        transform_final_3D = pca3D.transform(squeezed_final_embedding)
        transform_embeddings_3D = pca3D.transform(squeezed_embeddings)
        transform_remaining_3D = pca3D.transform(remaining_class_emb)

        # 3차원 PCA 플롯
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for idx, transformed_data in enumerate(transformed_top5_3D):
            ax.scatter(transformed_data[0], transformed_data[1], transformed_data[2], color=colors[idx], label=top5_indices_cos[idx], s=40)
        
        ax.scatter(transform_embeddings_3D[:, 0], transform_embeddings_3D[:, 1], transform_embeddings_3D[:, 2], c=range(len(transform_embeddings_3D)), cmap='spring',s=40)
        # ax.scatter(transform_final_2D[0][0], transform_final_2D[0][1], c='darkblue', label = 'optimal embedding', s=60,marker='*')
        for idx, transformed_data in enumerate(transform_remaining_3D):
            ax.scatter(transformed_data[0], transformed_data[1], transformed_data[2], color='grey', s=10, alpha=0.5)
        ax.scatter(transform_final_3D[0][0], transform_final_3D[0][1],transform_final_3D[0][2], c='darkblue', label = 'optimal embedding', s=60,marker='*')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('3D PCA on 128-dimensional data for 5 Groups')
        plt.legend()
        plt.show()
        plt.savefig(f'{class_dir}/{it}_{target}_3D_PCA_top5_embeddings_{seed_z}.png')

        print("Image generation finished!")


        plt.clf()




        print("3d CT image generated start!")
        # 3차원 PCA
        pca3D_ct = PCA(n_components=3)
        pca3D_ct.fit(all_data_ct)


        # 데이터 변환
        transformed_top5_3D_ct = pca3D_ct.transform(top5_class_emb_ct)
        transform_final_3D_ct = pca3D_ct.transform(squeezed_final_embedding_ct)
        transform_embeddings_3D_ct = pca3D_ct.transform(squeezed_embeddings_ct)
        # transform_remaining_3D = pca3D.transform(remaining_class_emb)

        # 3차원 PCA 플롯
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # for idx, transformed_data in enumerate(transformed_top5_3D_ct):
        #     ax.scatter(transformed_data[0], transformed_data[1], transformed_data[2], color=colors[int(idx/100)], label=top5_indices_cos[int(idx/100)], s=70)
        # ax.scatter(transform_final_3D_ct[0][0], transform_final_3D_ct[0][1], transform_final_3D_ct[0][2], c='darkblue', label = 'optimal embedding', s=70,marker='*')
        ax.scatter(transform_final_3D_ct[0][0], transform_final_3D_ct[0][1], transform_final_3D_ct[0][2], c='darkblue', label = 'optimal embedding', s=70,marker='*')
        ax.scatter(transform_embeddings_3D_ct[:, 0], transform_embeddings_3D_ct[:, 1], transform_embeddings_3D_ct[:, 2], c=range(len(transform_embeddings_3D_ct)), cmap='spring',s=50)
        ax.scatter(transformed_top5_3D_ct[:, 0], transformed_top5_3D_ct[:, 1], transformed_top5_3D_ct[:, 2], c='lightgrey',label='top 5 embedding',s=20,alpha=0.4)
        # for idx, transformed_data in enumerate(transform_remaining_3D):
        #     ax.scatter(transformed_data[0], transformed_data[1], transformed_data[2], color='grey', s=10, alpha=0.5)

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('3D PCA on 128-dimensional data for 5 Groups')
        plt.legend()
        plt.show()
        plt.savefig(f'{class_dir}/{it}_{target}_3D_PCA_top5_embeddings_ct_{seed_z}.png')

        print("3D CT PCA Image generation finished!")


        plt.clf()


        




        # 저장해둔 임베딩 벡터들에 대한 이미지 generation
        print("Intermediate embedding Image Generation Start")
        for i in range(len(embeddings)):
            print(f'{i+1}/{len(embeddings)}')
            ebd_tensor = torch.from_numpy(embeddings[i]).to(device)
            repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(zs, repeat_embedding)
            # vutils.save_image(gan_embedding_images,os.path.join(final_dir,'{}class_optimal_embedding'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)

            vutils.save_image(gan_embedding_images,os.path.join(class_dir,f'{it}_{target}_class_intermediate_embedding_{i}_{seed_z}')+'.png',normalize=True,scale_each=True,nrow=10)
        
        # 저장해둔 top 5 class embedding vector들에 대한 이미지 generation
        print("Intermediate embedding Image Generation Done...")
        print("Top 5 class embedding Image Generation Start")
        for i in range(len(top5_class_emb)):
            print(f"{i+1}/{len(top5_class_emb)}")
            ebd_tensor = torch.from_numpy(top5_class_emb[i]).to(device)
            repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(zs, repeat_embedding)
            # vutils.save_image(gan_embedding_images,os.path.join(final_dir,'{}class_optimal_embedding'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)

            vutils.save_image(gan_embedding_images,os.path.join(class_dir,f'{it}_{target}_top5_embedding_index_{i}_{seed_z}')+'.png',normalize=True,scale_each=True,nrow=10)
        print("Top 5 class embedding Image Generation Done...")

        # 저장해둔 low 5 class embedding vector들에 대한 이미지 generation
        print("Low 5 class embedding Image Generation Start")
        for i in range(len(low5_class_emb)):
            print(f"{i+1}/{len(low5_class_emb)}")
            ebd_tensor = torch.from_numpy(low5_class_emb[i]).to(device)
            repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(zs, repeat_embedding)
            # vutils.save_image(gan_embedding_images,os.path.join(final_dir,'{}class_optimal_embedding'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)

            vutils.save_image(gan_embedding_images,os.path.join(class_dir,f'{it}_{target}_low5_embedding_index_{i}_{seed_z}')+'.png',normalize=True,scale_each=True,nrow=10)
        print("Low 5 class embedding Image Generation Done...")












        # cosine similarity map 기준으로 embedding vector 더했을 때 image generation
        
        # for i in range(len(top5_class_emb)):
        #     ebd_tensor = torch.from_numpy(top5_class_emb[i]).to(device)
        #     repeat_embedding = ebd_tensor.repeat(1,1).to(device)
        #     with torch.no_grad():
        #     # Generate Images
        #         gan_embedding_images = G(zs, repeat_embedding)
        #     # vutils.save_image(gan_embedding_images,os.path.join(final_dir,'{}class_optimal_embedding'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)

        #     vutils.save_image(gan_embedding_images,os.path.join(class_dir,f'{target}_top5_embedding_index_{i}')+'.png',normalize=True,scale_each=True,nrow=10)
        

        cls_embedding = torch.from_numpy(top5_class_emb[0])
        print("Average class embedding Image Generation Start")
        #cosine similarity map 기준으로 embedding vector 더했을때
        for i in range(len(top5_class_emb) - 1 ):
            cls_embedding += torch.from_numpy(top5_class_emb[i+1])
        # if optim_comps["use_noise_layer"]:
        #      with torch.no_grad():
        #         z_hats = optim_comps["noise_layer"](zs)
        # else:
            # z_hats = zs
        # repeat_optimal_embedding = optim_embedding.repeat(z_num, 1).to(device)
        cls_embedding /= len(top5_class_emb)
        repeat_embedding = cls_embedding.repeat(1,1).to(device)
        with torch.no_grad():
            gan_images = G(zs, repeat_embedding)
            # output = cos(embedding, optim_embedding)
        vutils.save_image(gan_images,os.path.join(class_dir,f'{it}_{target}_top5_average_embedding_{seed_z}')+'.png',normalize=True,scale_each=True,nrow=10)
        print("Average class embedding Image Generation Done...")
        # if not os.path.exists(final_dir):
        #     os.makedirs(final_dir)
        # #cossim_list = cossim_list.sort()
        # vutils.save_image(gan_images,os.path.join(final_dir,'{}_cosine_similarity_latent_decompose'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)
        # np.save(f"{final_dir}/{target}_cosine_similarity_decompose_embedding.npy", optim_embedding.detach().cpu().numpy())
        # np.savetxt(os.path.join(f"{final_dir}/{target}",'_class_index_sortedby_cosinesim.csv'), cos_sort_index, fmt='%.4f')
        # np.savetxt(os.path.join(f"{final_dir}/{target}",'_sort_softmax_sortedby_cosinesim.csv'), avg_target_prob[cos_sort_index].cpu().detach(), fmt='%.3f')
        # np.savetxt(os.path.join(f"{final_dir}/{target}",'_cosine_similarity_sortedby_cosinesim.csv'),cossim_list, fmt='%.3f')


        # t-SNE 실험!!
        # imagenet embedding images
        # remaining_class_emb


        
        model = ResNet34(num_classes=10).to(device)
        model.load_state_dict(torch.load('/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar10_resnet34_9557.pt'),strict=True)
        model.linear = Identity()

        # Option -> For ImageNet pretrained CIFAR10
        # model = models.resnet34(pretrained=True).to(device)

        # model.fc = Identity()
        model.eval()


        # breakpoint()

        # model = ResNet34(num_classes=100).to(device)
        # model.load_state_dict(torch.load('/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar100_resnet34_7802.pth'),strict=True)
        # model.linear = Identity()
        # model.eval()


        # Make zs for image generation

        random_100_zs = torch.randn(100,1,dim_z).to(device)


        # class_embeedings_cpu = [class_embeddings[i].cpu().numpy() for i in range(len(class_embeddings))]



        imagenet_features = []
        print("Imagenet feature generation Start")
        for i in range(len(remaining_class_emb)):
            rem_class_emb = remaining_class_emb[i]
            rem_class_emb = torch.from_numpy(rem_class_emb).to(device)
            repeat_class_embedding = rem_class_emb.repeat(1, 1).to(device)

            with torch.no_grad():
                gan_images_tensor = G(zs, repeat_class_embedding)
                resized_images_tensor = nn.functional.interpolate(
                        gan_images_tensor, size=224
                    )
                # resized_images_tensor = nn.functional.interpolate(
                #         gan_images_tensor, size=32
                #     )
                feature = model(resized_images_tensor)
                imagenet_features.append(feature.squeeze().cpu().numpy())

        print("Imagenet feature generation Done")
        print(f"imagenet array shape:{np.array(imagenet_features).shape}")

        print("top5 feature generation Start")
        top5_features=[]
        for i in range(len(top5_class_emb)):
            # print(f"{i+1}/{len(top5_class_emb)}")
            ebd_tensor = torch.from_numpy(top5_class_emb[i]).to(device)
            repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(zs, repeat_embedding)
                resized_images_tensor = nn.functional.interpolate(
                        gan_embedding_images , size=224
                    )
                # resized_images_tensor = nn.functional.interpolate(
                #         gan_images_tensor, size=32
                #     )
                feature = model(resized_images_tensor)
                top5_features.append(feature.squeeze().cpu().numpy())

        print("top5 feature generation Done...")
        print(f"top5 array shape:{np.array(top5_features).shape}")

        print("low5 feature generation Start")
        low5_features=[]
        for i in range(len(low5_class_emb)):
            # print(f"{i+1}/{len(top5_class_emb)}")
            ebd_tensor = torch.from_numpy(low5_class_emb[i]).to(device)
            repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(zs, repeat_embedding)
                resized_images_tensor = nn.functional.interpolate(
                        gan_embedding_images , size=224
                    )
                # resized_images_tensor = nn.functional.interpolate(
                #         gan_images_tensor, size=32
                #     )
                feature = model(resized_images_tensor)
                low5_features.append(feature.squeeze().cpu().numpy())

        print("low5 feature generation Done...")
        print(f"low5 array shape:{np.array(low5_features).shape}")


        
        print("our embedding feature generation Start")
        intermediate_features=[]

        # breakpoint()
        # embeddings.shape -> (n,1,128)
        # final_embedding.shape -> torch.Size([1,128])
        embeddings.append(final_embedding.detach().to('cpu').numpy())
        for i in range(len(embeddings)):
            # print(f'{i+1}/{len(embeddings)}')
            ebd_tensor = torch.from_numpy(embeddings[i]).to(device)
            repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(zs, repeat_embedding)
                resized_images_tensor = nn.functional.interpolate(
                        gan_embedding_images , size=224
                    )
                # resized_images_tensor = nn.functional.interpolate(
                #         gan_images_tensor, size=32
                #     )
                feature = model(resized_images_tensor)
                intermediate_features.append(feature.squeeze().cpu().numpy())
        
        print("our embedding feature generation Done...")
        print(f"our embedding array shape:{np.array(intermediate_features).shape}")


        print("top5 100 features generation Start")
        top5_100_features=[]
        # len(top5_class_emb) = 5
        for i in range(len(top5_class_emb)):
            for z in random_100_zs:
                # print(f"{i+1}/{len(top5_class_emb)}")
                z_vec = z.to(device)
                ebd_tensor = torch.from_numpy(top5_class_emb[i]).to(device)
                repeat_embedding = ebd_tensor.repeat(1,1).to(device)
                with torch.no_grad():
                # Generate Images
                    gan_embedding_images = G(z_vec, repeat_embedding)
                    resized_images_tensor = nn.functional.interpolate(
                            gan_embedding_images , size=224
                        )
                    # resized_images_tensor = nn.functional.interpolate(
                    #         gan_images_tensor, size=32
                    #     )
                    feature = model(resized_images_tensor)
                    top5_100_features.append(feature.squeeze().cpu().numpy())

        print("top5 100 features generation Done...")
        # top5_100_features shape : 500* 512
        print(f"top5 100 features array shape:{np.array(top5_100_features).shape}")


        print("low5 100 features generation Start")
        low5_100_features=[]
        # len(top5_class_emb) = 5
        for i in range(len(low5_class_emb)):
            for z in random_100_zs:
                # print(f"{i+1}/{len(top5_class_emb)}")
                z_vec = z.to(device)
                ebd_tensor = torch.from_numpy(low5_class_emb[i]).to(device)
                repeat_embedding = ebd_tensor.repeat(1,1).to(device)
                with torch.no_grad():
                # Generate Images
                    gan_embedding_images = G(z_vec, repeat_embedding)
                    resized_images_tensor = nn.functional.interpolate(
                            gan_embedding_images , size=224
                        )
                    # resized_images_tensor = nn.functional.interpolate(
                    #         gan_images_tensor, size=32
                    #     )
                    feature = model(resized_images_tensor)
                    low5_100_features.append(feature.squeeze().cpu().numpy())

        print("low5 100 features generation Done...")
        # top5_100_features shape : 500* 512
        print(f"low5 100 features array shape:{np.array(low5_100_features).shape}")




        print("our Initial 100 features generation Start")
        init_features=[]

        # breakpoint()
        # embeddings.shape -> (n,1,128)
        # final_embedding.shape -> torch.Size([1,128])
        init_emb = torch.from_numpy(embeddings[0]).to(device)
        repeat_init_emb = init_emb.repeat(1,1).to(device)



        for z in random_100_zs:
            # print(f'{i+1}/{len(embeddings)}')
            # ebd_tensor = torch.from_numpy(embeddings[i]).to(device)
            # repeat_embedding = ebd_tensor.repeat(1,1).to(device)
            z_vec = z.to(device)

            with torch.no_grad():
            # Generate Images
                gan_embedding_images = G(z_vec, repeat_init_emb)
                resized_images_tensor = nn.functional.interpolate(
                        gan_embedding_images , size=224
                    )
                # resized_images_tensor = nn.functional.interpolate(
                #         gan_images_tensor, size=32
                #     )
                feature = model(resized_images_tensor)
                init_features.append(feature.squeeze().cpu().numpy())
        
        print("our Initial 100 features generation Done...")
        print(f"our Initial 100 features array shape:{np.array(init_features).shape}")





        transform_cifar10 = transforms.Compose(
        [
        # transforms.
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        
        transform_cifar100 = transforms.Compose(
            [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761)),
            ])

        print("CIFAR10 feature generation Start")   
        trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar10)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

        # print("CIFAR100 feature generation Start")

        # "/home/jihwan/DF_synthesis/cifar100_png/train/chair/"
        # "/home/jihwan/DF_synthesis/cifar100_png/train/chair/0013.png"
        # cifar100 load 21 classes

        # class별 embedding mean을 찍기 위함이었음 !


        # cifar100_target = ['apple', 'boy','bus','cloud','dolphin','lawn_mower','lobster','maple_tree',
        # 'oak_tree','palm_tree','pear','pine_tree','poppy','raccoon','rose','shrew',
        # 'sunflower','sweet_pepper','trout','tulip','willow_tree']


        # dataset = CustomImageFolder(
        #     root='/home/jihwan/DF_synthesis/cifar10_png/train',
        #     transform=transform_cifar10
        #     # target_classes=cifar100_target
        # )


        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) 


        # feature_sum = {label: torch.zeros(512).to('cuda' if torch.cuda.is_available() else 'cpu') for label in set(dataset.targets)}
        # class_counts = {label: 0 for label in set(dataset.targets)}



        # with torch.no_grad():
        #     for data, labels in dataloader:
        #         data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
        #         features = model(data).squeeze()
        #         for feature, label in zip(features, labels):
        #             feature_sum[label.item()] += feature
        #             class_counts[label.item()] += 1


        # # # breakpoint()

        # # # 클래스별 평균 feature 계산
        # average_feature = {label: feature_sum[label] / class_counts[label] for label in set(dataset.targets)}


        # features_tensor = torch.stack([average_feature[label] for label in sorted(average_feature.keys())])


        # cifar_features = features_tensor.cpu().numpy()

        

        # make embedding (CIFAR10 embeddings)
        images_class_1 = []
        for images, labels in trainloader:
            for i in range(len(labels)):
                if labels[i] == target and len(images_class_1) < 100:
                    images_class_1.append(images[i])
        # torch.stack: 3,32,32 * 200 -> [200,3,32,32]
        images_class_1 = torch.stack(images_class_1).to(device)
        with torch.no_grad():
            cifar_features_100 = model(images_class_1)
        cifar_features_100 = cifar_features_100.cpu().numpy()

        print("CIFAR10 average embedding feature generation Done...")
        print(f"CIFAR10 embedding array shape:{np.array(cifar_features_100).shape}")

        # feature map의 평균을 계산합니다.
        feature_map_mean = np.mean(cifar_features_100, axis=0)

        print(feature_map_mean.shape)  # 이 코드의 출력은 (512,)이 됩니다.

        # feature_map_mean을 (1, 512) 차원으로 변형합니다.
        feature_map_mean_reshaped = feature_map_mean.reshape(1, -1)

        print(feature_map_mean_reshaped.shape)  # 출력은 (1, 512)가 됩니다.


        # Our trace

        # lr = 0.01, iter = 1000


        

        # resnet에 태운다

        # embedding을 얻는다

        # cifar10 1번 class images

        # resnet에 태운다
        
        # embedding을 얻는다

        # ours images 

        # resnet에 태운다

        # embedding을 얻는다

        # tsne plot을 찍는다
        # imgnet : gray, CIFAR : blue, ours : red 계열, top 5 : 남색 계열


        print("tsne experiment start!")
        print("tsne for imagenet 1000 classes!")
        # dim of imagenet_features : n *512
        # dim of top5_features : n *512
        # dim of intermediate_features : n *512
        # dim of cifar_features : n *512

        imagenet_features = np.array(imagenet_features)
        top5_features = np.array(top5_features)
        intermediate_features = np.array(intermediate_features)
        cifar_features = np.array(cifar_features_100)
        top5_100_features = np.array(top5_100_features)
        init_features = np.array(init_features)
        cifar_mean_features = np.array(feature_map_mean_reshaped)




        low5_features = np.array(low5_features)
        low5_100_features = np.array(low5_100_features)
        # cifar_features_100 = np.array(cifar_features_100)




        # all_embeddings = np.vstack([imagenet_features, top5_features, intermediate_features, cifar_features])

        # all_embeddings = np.vstack([imagenet_features, intermediate_features, cifar_features,cifar_features_100])

        # all_embeddings = np.vstack([imagenet_features, intermediate_features, cifar_features])



        # all_embeddings = np.vstack([init_features, top5_100_features, intermediate_features, cifar_features])

        # all_embeddings = np.vstack([imagenet_features,top5_features,intermediate_features,cifar_mean_features])
        all_embeddings = np.vstack([init_features, intermediate_features, cifar_features])

        tsne = TSNE(n_jobs=4)

        tsne_embeddings = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(10, 10))


        # 색상 지정
        # Convert 'grey' to RGBA tuple
        # grey_rgba = matplotlib.colors.to_rgba('grey')
        # # colors = [grey_rgba] * imagenet_features.shape[0] 
        # colors = [grey_rgba] 


        # top5_colors = ['darkolivegreen', 'lawngreen', 'green', 'springgreen', 'mediumaquamarine']

        # # Convert string colors to RGBA tuples
        # top5_rgba = [matplotlib.colors.to_rgba(color) for color in top5_colors]
        # colors += top5_rgba
        # start_value = 0.3
        # colors += [plt.cm.Reds(i/len(intermediate_features)) for i in range(len(intermediate_features))]
        # # colors += [plt.cm.Reds(i/len(intermediate_features)) for i in range(len(intermediate_features))]
        # blue_rgba = matplotlib.colors.to_rgba('blue')
        
        # colors += [blue_rgba] * cifar_features.shape[0] 
        # skyblue_rgba = matplotlib.colors.to_rgba('skyblue')

        # colors += [skyblue_rgba] * cifar_features_100.shape[0] 
        # colors = ['grey'] * imagenet_features.shape[0] + top5_colors + intermediate_reds + cifar_blues

        # 라벨 지정
        # labels = ['Imagenet 995'] * imagenet_features.shape[0] + ['Imagenet 5'] * top5_features.shape[0] + ['CIFAR10 Class 1'] * intermediate_features.shape[0] + ['Custom Embedding'] * cifar_features.shape[0]

        # labels = ['Init Concept'] * init_features.shape[0] 
        # labels += [f'{top5_indices_cos[0]}'] * (int(top5_100_features.shape[0]/5))
        # labels += [f'{top5_indices_cos[1]}'] * (int(top5_100_features.shape[0]/5))
        # labels += [f'{top5_indices_cos[2]}'] * (int(top5_100_features.shape[0]/5))
        # labels += [f'{top5_indices_cos[3]}'] * (int(top5_100_features.shape[0]/5))
        # labels += [f'{top5_indices_cos[4]}'] * (int(top5_100_features.shape[0]/5))
        # labels += ['Concept Trace'] * intermediate_features.shape[0]
        # labels += ['Target Concept'] * cifar_features.shape[0]
        # labels += [f'Cifar100 {target}'] * cifar_features_100.shape[0]



        # colors = {
        #     'Init Concept': '#B0B0B0',
        #     f'{top5_indices_cos[0]}': '#D3D3D3',
        #     f'{top5_indices_cos[1]}': '#A9A9A9',
        #     f'{top5_indices_cos[2]}': '#808080',
        #     f'{top5_indices_cos[3]}': '#C0C0C0',
        #     f'{top5_indices_cos[4]}': '#696969',
        #     'Concept Trace': 'red',  # 이 부분은 진행됨에 따라 색상을 변경하도록 추가 작업이 필요
        #     'Target Concept': '#2180A6'
        # }
        # markers = {
        #     'Concept Trace': '*'
        # }


        labels = ['ImageNet Features'] * init_features.shape[0] 
        # labels += ['Top5 Features'] * top5_features.shape[0]
        labels += ['Concept Trace'] * intermediate_features.shape[0]
        labels += ['Target Concept'] * cifar_mean_features.shape[0]


        colors = {
            'ImageNet Features': '#B0B0B0',
            # 'Top5 Features': '#D3D3D3',
            'Concept Trace': 'red',  # 이 부분은 진행됨에 따라 색상을 변경하도록 추가 작업이 필요
            'Target Concept': '#2180A6'
        }
        markers = {
            'Concept Trace': '*'
        }

        
        # unique_labels = np.unique(labels)
        # # breakpoint()
        # for label in unique_labels:
        #     mask = np.array(labels) == label
        #     current_colors = np.array(colors)[mask]
        #     if label == 'Top 5 Similar concepts':
        #     # 점에 테두리 추가
        #         plt.scatter(embedded[mask, 0], embedded[mask, 1], label=label, c=current_colors, edgecolors='black')
        #     elif label == 'Concept Trace':
        #         # 별 모양으로 scatter plot 표시
        #         plt.scatter(embedded[mask, 0], embedded[mask, 1], label=label, c=current_colors, marker='*',s=70)
            
        #         # 이동 경로 표시
        #         x_vals = embedded[mask, 0]
        #         y_vals = embedded[mask, 1]
        #         plt.plot(x_vals, y_vals, color='orange', linestyle='--', linewidth=1)

        #         # 화살표 선 추가
        #         start_x, end_x = x_vals[0], x_vals[-1]
        #         start_y, end_y = y_vals[0], y_vals[-1]
        #         plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, shape='full', lw=1.5, length_includes_head=True, head_width=0.3, color='orangered')


        #     else:
        #         plt.scatter(embedded[mask, 0], embedded[mask, 1], label=label, c=current_colors, alpha=0.5)
        # Concept Trace를 연결할 화살표 추가
        # Concept Trace의 시작점과 끝점만 화살표로 연결
        trace_idxs = [i for i, l in enumerate(labels) if l == 'Concept Trace']
        # if len(trace_idxs) > 1:
        #     plt.quiver(tsne_embeddings[trace_idxs[0], 0], tsne_embeddings[trace_idxs[0], 1], 
        #             tsne_embeddings[trace_idxs[-1], 0] - tsne_embeddings[trace_idxs[0], 0], 
        #             tsne_embeddings[trace_idxs[-1], 1] - tsne_embeddings[trace_idxs[0], 1], 
        #             angles='xy', scale_units='xy', scale=0.7, color='red')


        # Concept Trace의 scatter plot을 점진적으로 색상이 변하도록 설정
        start_color = np.array([1, 0.8, 0.8])  # 흰색에 가까운 빨간색 (RGB)
        end_color = np.array([0.8, 0, 0])  # 조금 더 짙은 빨간색 (RGB)
        color_diff = end_color - start_color
        num_trace = len(trace_idxs)
        # trace_colors = [start_color + i * color_diff / (num_trace - 1) for i in range(num_trace)]

        if num_trace > 1:
            trace_colors = [start_color + (i * color_diff) / (num_trace - 1) for i in range(num_trace)]
        else:
            trace_colors = [start_color]  # If there's only one point, no gradient is needed
        
        
        for label, color in colors.items():
            if label == 'Target Concept':  # Concept Trace는 이미 처리됨
                idxs = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(tsne_embeddings[idxs, 0], tsne_embeddings[idxs, 1], c=color, label=label, s=80)
            elif label != 'Concept Trace':  # Concept Trace는 이미 처리됨
                idxs = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(tsne_embeddings[idxs, 0], tsne_embeddings[idxs, 1], c=color, s=50, alpha=0.5)

        for idx, color in zip(trace_idxs, trace_colors):
            print(f"Index:{idx}, Color:{color}")
            plt.scatter(tsne_embeddings[idx, 0], tsne_embeddings[idx, 1], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'],s=80)
        # Concept Trace의 모든 x와 y 값 가져오기
        x_vals = [tsne_embeddings[i, 0] for i in trace_idxs]
        y_vals = [tsne_embeddings[i, 1] for i in trace_idxs]

        plt.plot(x_vals, y_vals, color='orange', linestyle='--', linewidth=1)
            # # 이동 경로 표시
            # x_vals = tsne_embeddings[idx, 0]
            # y_vals = tsne_embeddings[idx, 1]
            

        # 화살표 선 추가
        start_x, end_x = x_vals[0], x_vals[-1]
        start_y, end_y = y_vals[0], y_vals[-1]
        plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, shape='full', lw=1.5, length_includes_head=True, head_width=0.7, color='red')

        

        plt.legend()
        plt.rc('axes', labelsize=15)   # x,y축 label 폰트 크기
        # plt.rc('figure', titlesize=30) # figure title 폰트 크기
        plt.title('t-SNE of Embeddings',fontsize=20)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')



        # plt.xlim([-20,20])
        # plt.ylim([-20,20])
        plt.show()

        # 이미지 저장
        plt.savefig(f'{class_dir}/{it}_{target}_TSNE_imagenet1000_{seed_z}.png') 

        print("Image generation imagenet 1000 classes finished!")

        plt.clf()


        # UMAP 설정
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
        umap_embeddings = reducer.fit_transform(all_embeddings)

        plt.figure(figsize=(10, 10))

        # ...[기존 코드 레이블 및 색상 설정 부분은 그대로 유지]...

        for label, color in colors.items():
            if label == 'Target Concept':  # Concept Trace는 이미 처리됨
                idxs = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(umap_embeddings[idxs, 0], umap_embeddings[idxs, 1], c=color, label=label, s=80)
            elif label != 'Concept Trace':  # Concept Trace는 이미 처리됨
                idxs = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(umap_embeddings[idxs, 0], umap_embeddings[idxs, 1], c=color, s=50, alpha=0.5)

        for idx, color in zip(trace_idxs, trace_colors):
            plt.scatter(umap_embeddings[idx, 0], umap_embeddings[idx, 1], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'],s=80)
        # for label, color in colors.items():
        #     # ...[기존 scatter plot 그리기 코드는 그대로 유지]...

        #     idxs = [i for i, l in enumerate(labels) if l == label]
        #     plt.scatter(umap_embeddings[idxs, 0], umap_embeddings[idxs, 1], c=color, label=label, s=50)

        # ...[화살표 선 추가 및 기타 설정 부분은 그대로 유지]...

         # Concept Trace의 모든 x와 y 값 가져오기
        x_vals = [umap_embeddings[i, 0] for i in trace_idxs]
        y_vals = [umap_embeddings[i, 1] for i in trace_idxs]

        plt.plot(x_vals, y_vals, color='orange', linestyle='--', linewidth=1)
            # # 이동 경로 표시
            # x_vals = tsne_embeddings[idx, 0]
            # y_vals = tsne_embeddings[idx, 1]
            

        # 화살표 선 추가
        start_x, end_x = x_vals[0], x_vals[-1]
        start_y, end_y = y_vals[0], y_vals[-1]
        plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, shape='full', lw=1.5, length_includes_head=True, head_width=0.5, color='red')

        plt.title('UMAP of Embeddings',fontsize=20)
        plt.xlabel('UMAP Component 1',fontsize=15)
        plt.ylabel('UMAP Component 2',fontsize=15)
        plt.show()

        # 이미지 저장
        plt.savefig(f'{class_dir}/{it}_{target}_UMAP_imagenet1000_{seed_z}.png') 

        print("Image generation imagenet 1000 classes finished!")




        print("3D tsne for imagenet 1000 classes!")
        tsne2 = TSNE(n_components=3, n_jobs=4)

        tsne_embeddings_3d = tsne2.fit_transform(all_embeddings)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')


        # # 색상 지정
        # # Convert 'grey' to RGBA tuple
        # grey_rgba = matplotlib.colors.to_rgba('grey')
        # colors = [grey_rgba] * imagenet_features.shape[0] 


        # top5_colors = ['darkolivegreen', 'lawngreen', 'green', 'springgreen', 'mediumaquamarine']

        # # Convert string colors to RGBA tuples
        # top5_rgba = [matplotlib.colors.to_rgba(color) for color in top5_colors]
        # # colors += top5_rgba
        # # cstart_value = 0.3
        # colors += [plt.cm.Reds(i/len(intermediate_features)) for i in range(len(intermediate_features))]
        # blue_rgba = matplotlib.colors.to_rgba('blue')
        # colors += [blue_rgba] * cifar_features.shape[0] 
        # skyblue_rgba = matplotlib.colors.to_rgba('skyblue')

        # # colors += [skyblue_rgba] * cifar_features_100.shape[0] 

        # # colors = ['grey'] * imagenet_features.shape[0] + top5_colors + intermediate_reds + cifar_blues

        # # 라벨 지정
        # # labels = ['Imagenet 995'] * imagenet_features.shape[0] + ['Imagenet 5'] * top5_features.shape[0] + ['CIFAR10 Class 1'] * intermediate_features.shape[0] + ['Custom Embedding'] * cifar_features.shape[0]

        # labels = ['Class concept space'] * imagenet_features.shape[0] 
        # # labels += ['Top 5 Similar concepts'] * top5_features.shape[0]
        # labels += ['Concept Trace'] * intermediate_features.shape[0]
        # labels += ['Target Concept'] * cifar_features.shape[0]
        # # labels += [f'Cifar100 {target}'] * cifar_features_100.shape[0]
        
        # unique_labels = np.unique(labels)
        # # breakpoint()
        # for label in unique_labels:
        #     mask = np.array(labels) == label
        #     current_colors = np.array(colors)[mask]
            
        #     if label == 'Top 5 Similar concepts':
        #         ax.scatter(embedded[mask, 0], embedded[mask, 1], embedded[mask, 2], label=label, c=current_colors, edgecolors='black')
        #     elif label == 'Concept Trace':
        #         ax.scatter(embedded[mask, 0], embedded[mask, 1], embedded[mask, 2], label=label, c=current_colors, marker='*',s=70)
                
        #         # 이동 경로 표시
        #         x_vals = embedded[mask, 0]
        #         y_vals = embedded[mask, 1]
        #         z_vals = embedded[mask, 2]
        #         ax.plot(x_vals, y_vals, z_vals, color='orange', linestyle='--', linewidth=1)

        #         # 화살표 선 추가
        #         start_x, start_y, start_z = x_vals[0], y_vals[0], z_vals[0]
        #         end_x, end_y, end_z = x_vals[-1], y_vals[-1], z_vals[-1]

        #         arrow_prop_dict = dict(mutation_scale=15, lw=1.5, arrowstyle="-|>", color="orangered",shrinkA=0, shrinkB=0)
        #         arrow = Arrow3D([start_x, end_x], [start_y, end_y], [start_z, end_z], **arrow_prop_dict)
        #         ax.add_artist(arrow)

        #     else:
        #         ax.scatter(embedded[mask, 0], embedded[mask, 1], embedded[mask, 2], label=label, c=current_colors)
        # Concept Trace를 연결할 화살표 추가
        # Concept Trace를 연결할 화살표 추가
        # # 3D 화살표는 quiver를 사용하기보다 직선으로 표현
        # trace_idxs = [i for i, l in enumerate(labels) if l == 'Concept Trace']

        # Concept Trace의 시작점과 끝점만 화살표로 연결
        trace_idxs = [i for i, l in enumerate(labels) if l == 'Concept Trace']
        # if len(trace_idxs) > 1:
        #     ax.quiver(tsne_embeddings_3d[trace_idxs[0], 0], tsne_embeddings_3d[trace_idxs[0], 1], tsne_embeddings_3d[trace_idxs[0], 2],
        #             tsne_embeddings_3d[trace_idxs[-1], 0] - tsne_embeddings_3d[trace_idxs[0], 0], 
        #             tsne_embeddings_3d[trace_idxs[-1], 1] - tsne_embeddings_3d[trace_idxs[0], 1],
        #             tsne_embeddings_3d[trace_idxs[-1], 2] - tsne_embeddings_3d[trace_idxs[0], 2],
        #             length=1, color='red')

        # Concept Trace의 scatter plot을 점진적으로 색상이 변하도록 설정
        start_color = np.array([1, 0.8, 0.8])  # 흰색에 가까운 빨간색 (RGB)
        end_color = np.array([0.8, 0, 0])  # 조금 더 짙은 빨간색 (RGB)
        color_diff = end_color - start_color
        num_trace = len(trace_idxs)
        trace_colors = [start_color + i * color_diff / (num_trace - 1) for i in range(num_trace)]



        for label, color in colors.items():
            if label == 'Target Concept':  # Concept Trace는 이미 처리됨
                idxs = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(tsne_embeddings_3d[idxs, 0], tsne_embeddings_3d[idxs, 1], tsne_embeddings_3d[idxs, 2], c=color, label=label, s=80)
            elif label != 'Concept Trace':  # Concept Trace는 이미 처리됨
                idxs = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(tsne_embeddings_3d[idxs, 0], tsne_embeddings_3d[idxs, 1], tsne_embeddings_3d[idxs, 2], c=color, s=50)
        for idx, color in zip(trace_idxs, trace_colors):
            ax.scatter(tsne_embeddings_3d[idx, 0], tsne_embeddings_3d[idx, 1], tsne_embeddings_3d[idx, 2], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'], edgecolors='black', s=85)


             # 이동 경로 표시
            # x_vals = tsne_embeddings_3d[idx, 0]
            # y_vals = tsne_embeddings_3d[idx, 1]
            # z_vals = tsne_embeddings_3d[idx, 2]
            # ax.plot(x_vals, y_vals, z_vals, color='orange', linestyle='--', linewidth=1)

            # 화살표 선 추가
            # start_x, start_y, start_z = x_vals[0], y_vals[0], z_vals[0]
            # end_x, end_y, end_z = x_vals[-1], y_vals[-1], z_vals[-1]

            # arrow_prop_dict = dict(mutation_scale=15, lw=1.5, arrowstyle="-|>", color="orangered",shrinkA=0, shrinkB=0)
            # arrow = Arrow3D([start_x, end_x], [start_y, end_y], [start_z, end_z], **arrow_prop_dict)
            # ax.add_artist(arrow)
        x_vals = [tsne_embeddings_3d[i, 0] for i in trace_idxs]
        y_vals = [tsne_embeddings_3d[i, 1] for i in trace_idxs]
        z_vals = [tsne_embeddings_3d[i, 2] for i in trace_idxs]

        ax.plot(x_vals, y_vals, z_vals, color='orange', linestyle='--', linewidth=1)

            # # 이동 경로 표시
            # x_vals = tsne_embeddings[idx, 0]
            # y_vals = tsne_embeddings[idx, 1]
            
        # 화살표 선 추가
        start_x, start_y, start_z = x_vals[0], y_vals[0], z_vals[0]
        end_x, end_y, end_z = x_vals[-1], y_vals[-1], z_vals[-1]

        arrow_prop_dict = dict(mutation_scale=15, lw=1.5, arrowstyle="-|>", color='red',shrinkA=0, shrinkB=0)
        arrow = Arrow3D([start_x, end_x], [start_y, end_y], [start_z, end_z], **arrow_prop_dict)
        ax.add_artist(arrow)
        # # 화살표 선 추가
        # start_x, end_x = x_vals[0], x_vals[-1]
        # start_y, end_y = y_vals[0], y_vals[-1]
        # plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, shape='full', lw=1.5, length_includes_head=True, head_width=0.3, color='orangered')

    
        # for i in range(1, len(trace_idxs)):
        #     ax.plot(tsne_embeddings_3d[trace_idxs[i-1:i+1], 0], tsne_embeddings_3d[trace_idxs[i-1:i+1], 1], tsne_embeddings_3d[trace_idxs[i-1:i+1], 2], color='red')
        # 나머지 데이터 표시
        

        # 나머지 데이터 표시
        # for label, color in colors.items():
        #     idxs = [i for i, l in enumerate(labels) if l == label]
        #     if label in markers:
        #         ax.scatter(tsne_embeddings_3d[idxs, 0], tsne_embeddings_3d[idxs, 1], tsne_embeddings_3d[idxs, 2], c=color, label=label, marker=markers[label], edgecolors='black', s=70)
        #     else:
        #         ax.scatter(tsne_embeddings_3d[idxs, 0], tsne_embeddings_3d[idxs, 1], tsne_embeddings_3d[idxs, 2], c=color, label=label, s=50)




        ax.set_title('t-SNE of Embeddings', fontsize=20)
        ax.set_xlabel('t-SNE Component 1', fontsize=15)
        ax.set_ylabel('t-SNE Component 2', fontsize=15)
        ax.set_zlabel('t-SNE Component 3', fontsize=15)
        ax.legend()

        
        # 이미지 저장 전에 호출
        # arrow._proj_changed()

        # 이미지 저장
        plt.savefig(f'{class_dir}/{it}_{target}_TSNE_imagenet1000_3D_{seed_z}.png') 

        plt.show()
        print("3D Image generation imagenet 1000 classes finished!")





        print("3D UMAP for imagenet 1000 classes!")
        umap_model = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=3)
        umap_embeddings_3d = umap_model.fit_transform(all_embeddings)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Concept Trace's scatter plot to gradually change color
        start_color = np.array([1, 0.8, 0.8])  # A red close to white (RGB)
        end_color = np.array([0.8, 0, 0])  # A slightly darker red (RGB)
        color_diff = end_color - start_color
        num_trace = len(trace_idxs)
        trace_colors = [start_color + i * color_diff / (num_trace - 1) for i in range(num_trace)]

        for label, color in colors.items():
            if label == 'Target Concept':  # Concept Trace is already handled
                idxs = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(umap_embeddings_3d[idxs, 0], umap_embeddings_3d[idxs, 1], umap_embeddings_3d[idxs, 2], c=color, label=label, s=70)
            elif label != 'Concept Trace':  # Concept Trace is already handled
                idxs = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(umap_embeddings_3d[idxs, 0], umap_embeddings_3d[idxs, 1], umap_embeddings_3d[idxs, 2], c=color, s=50)

        for idx, color in zip(trace_idxs, trace_colors):
            ax.scatter(umap_embeddings_3d[idx, 0], umap_embeddings_3d[idx, 1], umap_embeddings_3d[idx, 2], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'], edgecolors='black', s=85)

        x_vals = [umap_embeddings_3d[i, 0] for i in trace_idxs]
        y_vals = [umap_embeddings_3d[i, 1] for i in trace_idxs]
        z_vals = [umap_embeddings_3d[i, 2] for i in trace_idxs]

        ax.plot(x_vals, y_vals, z_vals, color='orange', linestyle='--', linewidth=1)

        start_x, start_y, start_z = x_vals[0], y_vals[0], z_vals[0]
        end_x, end_y, end_z = x_vals[-1], y_vals[-1], z_vals[-1]

        arrow_prop_dict = dict(mutation_scale=15, lw=1.5, arrowstyle="-|>", color='red',shrinkA=0, shrinkB=0)
        arrow = Arrow3D([start_x, end_x], [start_y, end_y], [start_z, end_z], **arrow_prop_dict)
        ax.add_artist(arrow)

        ax.set_title('UMAP of Embeddings', fontsize=20)
        ax.set_xlabel('UMAP Component 1', fontsize=15)
        ax.set_ylabel('UMAP Component 2', fontsize=15)
        ax.set_zlabel('UMAP Component 3', fontsize=15)
        ax.legend()

        plt.savefig(f'{class_dir}/{it}_{target}_UMAP_imagenet1000_3D_{seed_z}.png') 

        plt.show()
        print("3D Image generation using UMAP for imagenet 1000 classes finished!")









        # all_embeddings_low = np.vstack([init_features, low5_100_features, intermediate_features, cifar_features])


        # all_embeddings_low = np.vstack([init_features, low5_100_features, intermediate_features, cifar_features])

        # tsne3 = TSNE(n_jobs=4)

        # tsne_embeddings = tsne3.fit_transform(all_embeddings_low)

        # plt.figure(figsize=(10, 10))



        # labels = ['Init Concept'] * init_features.shape[0] 
        # labels += [f'{low5_indices_cos[0]}'] * (int(low5_100_features.shape[0]/5))
        # labels += [f'{low5_indices_cos[1]}'] * (int(low5_100_features.shape[0]/5))
        # labels += [f'{low5_indices_cos[2]}'] * (int(low5_100_features.shape[0]/5))
        # labels += [f'{low5_indices_cos[3]}'] * (int(low5_100_features.shape[0]/5))
        # labels += [f'{low5_indices_cos[4]}'] * (int(low5_100_features.shape[0]/5))
        # labels += ['Concept Trace'] * intermediate_features.shape[0]
        # labels += ['Target Concept'] * cifar_features.shape[0]
        # # labels += [f'Cifar100 {target}'] * cifar_features_100.shape[0]


        # colors = {
        #     'Init Concept': '#B0B0B0',
        #     f'{low5_indices_cos[0]}': '#D3D3D3',
        #     f'{low5_indices_cos[1]}': '#A9A9A9',
        #     f'{low5_indices_cos[2]}': '#808080',
        #     f'{low5_indices_cos[3]}': '#C0C0C0',
        #     f'{low5_indices_cos[4]}': '#696969',
        #     'Concept Trace': 'red',  # 이 부분은 진행됨에 따라 색상을 변경하도록 추가 작업이 필요
        #     'Target Concept': '#2180A6'
        # }
        # markers = {
        #     'Concept Trace': '*'
        # }

        # trace_idxs = [i for i, l in enumerate(labels) if l == 'Concept Trace']
       


        # # Concept Trace의 scatter plot을 점진적으로 색상이 변하도록 설정
        # start_color = np.array([1, 0.8, 0.8])  # 흰색에 가까운 빨간색 (RGB)
        # end_color = np.array([0.8, 0, 0])  # 조금 더 짙은 빨간색 (RGB)
        # color_diff = end_color - start_color
        # num_trace = len(trace_idxs)
        # # trace_colors = [start_color + i * color_diff / (num_trace - 1) for i in range(num_trace)]

        # if num_trace > 1:
        #     trace_colors = [start_color + (i * color_diff) / (num_trace - 1) for i in range(num_trace)]
        # else:
        #     trace_colors = [start_color]  # If there's only one point, no gradient is needed
        
        
        # for label, color in colors.items():
        #     if label == 'Target Concept':  # Concept Trace는 이미 처리됨
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         plt.scatter(tsne_embeddings[idxs, 0], tsne_embeddings[idxs, 1], c=color, label=label, s=50)
        #     elif label != 'Concept Trace':  # Concept Trace는 이미 처리됨
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         plt.scatter(tsne_embeddings[idxs, 0], tsne_embeddings[idxs, 1], c=color, s=50, alpha=0.5)

        # for idx, color in zip(trace_idxs, trace_colors):
        #     plt.scatter(tsne_embeddings[idx, 0], tsne_embeddings[idx, 1], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'],s=70)
        # # Concept Trace의 모든 x와 y 값 가져오기
        # x_vals = [tsne_embeddings[i, 0] for i in trace_idxs]
        # y_vals = [tsne_embeddings[i, 1] for i in trace_idxs]

        # plt.plot(x_vals, y_vals, color='orange', linestyle='--', linewidth=1)
        #     # # 이동 경로 표시
        #     # x_vals = tsne_embeddings[idx, 0]
        #     # y_vals = tsne_embeddings[idx, 1]
            

        # # 화살표 선 추가
        # start_x, end_x = x_vals[0], x_vals[-1]
        # start_y, end_y = y_vals[0], y_vals[-1]
        # plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, shape='full', lw=1.5, length_includes_head=True, head_width=0.7, color='#FFB343')

        

        # plt.legend()
        # plt.rc('axes', labelsize=15)   # x,y축 label 폰트 크기
        # # plt.rc('figure', titlesize=30) # figure title 폰트 크기
        # plt.title('t-SNE of Embeddings',fontsize=20)
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')



        # # plt.xlim([-20,20])
        # # plt.ylim([-20,20])
        # plt.show()

        # # 이미지 저장
        # plt.savefig(f'{class_dir}/{it}_{target}_TSNE_imagenet1000_Low_{seed_z}.png') 

        # print("Image generation imagenet 1000 classes - Low finished!")

        # plt.clf()


        # # UMAP 설정
        # reducer3 = umap.UMAP(n_neighbors=15, min_dist=0.1)
        # umap_embeddings = reducer3.fit_transform(all_embeddings_low)

        # plt.figure(figsize=(10, 10))

        # # ...[기존 코드 레이블 및 색상 설정 부분은 그대로 유지]...

        # for label, color in colors.items():
        #     if label == 'Target Concept':  # Concept Trace는 이미 처리됨
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         plt.scatter(umap_embeddings[idxs, 0], umap_embeddings[idxs, 1], c=color, label=label, s=50)
        #     elif label != 'Concept Trace':  # Concept Trace는 이미 처리됨
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         plt.scatter(umap_embeddings[idxs, 0], umap_embeddings[idxs, 1], c=color, s=50, alpha=0.5)

        # for idx, color in zip(trace_idxs, trace_colors):
        #     plt.scatter(umap_embeddings[idx, 0], umap_embeddings[idx, 1], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'],s=70)
        

        #  # Concept Trace의 모든 x와 y 값 가져오기
        # x_vals = [umap_embeddings[i, 0] for i in trace_idxs]
        # y_vals = [umap_embeddings[i, 1] for i in trace_idxs]

        # plt.plot(x_vals, y_vals, color='orange', linestyle='--', linewidth=1)
        #     # # 이동 경로 표시
        #     # x_vals = tsne_embeddings[idx, 0]
        #     # y_vals = tsne_embeddings[idx, 1]
            

        # # 화살표 선 추가
        # start_x, end_x = x_vals[0], x_vals[-1]
        # start_y, end_y = y_vals[0], y_vals[-1]
        # plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, shape='full', lw=1.5, length_includes_head=True, head_width=0.5, color='#FFB343')

        # plt.title('UMAP of Embeddings',fontsize=20)
        # plt.xlabel('UMAP Component 1',fontsize=15)
        # plt.ylabel('UMAP Component 2',fontsize=15)
        # plt.show()

        # # 이미지 저장
        # plt.savefig(f'{class_dir}/{it}_{target}_UMAP_imagenet1000_low_{seed_z}.png') 

        # print("Image generation imagenet 1000 classes Low finished!")




        # print("3D tsne Low for imagenet 1000 classes!")
        # tsne4 = TSNE(n_components=3, n_jobs=4)

        # tsne_embeddings_3d = tsne4.fit_transform(all_embeddings_low)

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # # Concept Trace의 시작점과 끝점만 화살표로 연결
        # trace_idxs = [i for i, l in enumerate(labels) if l == 'Concept Trace']
        
        # # Concept Trace의 scatter plot을 점진적으로 색상이 변하도록 설정
        # start_color = np.array([1, 0.8, 0.8])  # 흰색에 가까운 빨간색 (RGB)
        # end_color = np.array([0.8, 0, 0])  # 조금 더 짙은 빨간색 (RGB)
        # color_diff = end_color - start_color
        # num_trace = len(trace_idxs)
        # trace_colors = [start_color + i * color_diff / (num_trace - 1) for i in range(num_trace)]


        # for label, color in colors.items():
        #     if label == 'Target Concept':  # Concept Trace는 이미 처리됨
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         ax.scatter(tsne_embeddings_3d[idxs, 0], tsne_embeddings_3d[idxs, 1], tsne_embeddings_3d[idxs, 2], c=color, label=label, s=50)
        #     elif label != 'Concept Trace':  # Concept Trace는 이미 처리됨
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         ax.scatter(tsne_embeddings_3d[idxs, 0], tsne_embeddings_3d[idxs, 1], tsne_embeddings_3d[idxs, 2], c=color, s=50)
        # for idx, color in zip(trace_idxs, trace_colors):
        #     ax.scatter(tsne_embeddings_3d[idx, 0], tsne_embeddings_3d[idx, 1], tsne_embeddings_3d[idx, 2], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'], edgecolors='black', s=85)


        # x_vals = [tsne_embeddings_3d[i, 0] for i in trace_idxs]
        # y_vals = [tsne_embeddings_3d[i, 1] for i in trace_idxs]
        # z_vals = [tsne_embeddings_3d[i, 2] for i in trace_idxs]

        # ax.plot(x_vals, y_vals, z_vals, color='orange', linestyle='--', linewidth=1)

        #     # # 이동 경로 표시
        #     # x_vals = tsne_embeddings[idx, 0]
        #     # y_vals = tsne_embeddings[idx, 1]
            
        # # 화살표 선 추가
        # start_x, start_y, start_z = x_vals[0], y_vals[0], z_vals[0]
        # end_x, end_y, end_z = x_vals[-1], y_vals[-1], z_vals[-1]

        # arrow_prop_dict = dict(mutation_scale=15, lw=1.5, arrowstyle="-|>", color='#FFB343',shrinkA=0, shrinkB=0)
        # arrow = Arrow3D([start_x, end_x], [start_y, end_y], [start_z, end_z], **arrow_prop_dict)
        # ax.add_artist(arrow)
        # # # 화살표 선 추가
       



        # ax.set_title('t-SNE of Embeddings', fontsize=20)
        # ax.set_xlabel('t-SNE Component 1', fontsize=15)
        # ax.set_ylabel('t-SNE Component 2', fontsize=15)
        # ax.set_zlabel('t-SNE Component 3', fontsize=15)
        # ax.legend()

        


        # # 이미지 저장
        # plt.savefig(f'{class_dir}/{it}_{target}_TSNE_imagenet1000_3D_low_{seed_z}.png') 

        # plt.show()
        # print("3D Image generation imagenet 1000 classes - low finished!")





        # print("3D UMAP for imagenet 1000 classes!")
        # umap_model_2 = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=3)
        # umap_embeddings_3d = umap_model_2.fit_transform(all_embeddings_low)

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # # Concept Trace's scatter plot to gradually change color
        # start_color = np.array([1, 0.8, 0.8])  # A red close to white (RGB)
        # end_color = np.array([0.8, 0, 0])  # A slightly darker red (RGB)
        # color_diff = end_color - start_color
        # num_trace = len(trace_idxs)
        # trace_colors = [start_color + i * color_diff / (num_trace - 1) for i in range(num_trace)]

        # for label, color in colors.items():
        #     if label == 'Target Concept':  # Concept Trace is already handled
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         ax.scatter(umap_embeddings_3d[idxs, 0], umap_embeddings_3d[idxs, 1], umap_embeddings_3d[idxs, 2], c=color, label=label, s=50)
        #     elif label != 'Concept Trace':  # Concept Trace is already handled
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         ax.scatter(umap_embeddings_3d[idxs, 0], umap_embeddings_3d[idxs, 1], umap_embeddings_3d[idxs, 2], c=color, s=50)

        # for idx, color in zip(trace_idxs, trace_colors):
        #     ax.scatter(umap_embeddings_3d[idx, 0], umap_embeddings_3d[idx, 1], umap_embeddings_3d[idx, 2], c=[color], label='Concept Trace' if idx == trace_idxs[0] else "", marker=markers['Concept Trace'], edgecolors='black', s=85)

        # x_vals = [umap_embeddings_3d[i, 0] for i in trace_idxs]
        # y_vals = [umap_embeddings_3d[i, 1] for i in trace_idxs]
        # z_vals = [umap_embeddings_3d[i, 2] for i in trace_idxs]

        # ax.plot(x_vals, y_vals, z_vals, color='orange', linestyle='--', linewidth=1)

        # start_x, start_y, start_z = x_vals[0], y_vals[0], z_vals[0]
        # end_x, end_y, end_z = x_vals[-1], y_vals[-1], z_vals[-1]

        # arrow_prop_dict = dict(mutation_scale=15, lw=1.5, arrowstyle="-|>", color='#FFB343',shrinkA=0, shrinkB=0)
        # arrow = Arrow3D([start_x, end_x], [start_y, end_y], [start_z, end_z], **arrow_prop_dict)
        # ax.add_artist(arrow)

        # ax.set_title('UMAP of Embeddings', fontsize=20)
        # ax.set_xlabel('UMAP Component 1', fontsize=15)
        # ax.set_ylabel('UMAP Component 2', fontsize=15)
        # ax.set_zlabel('UMAP Component 3', fontsize=15)
        # ax.legend()

        # plt.savefig(f'{class_dir}/{it}_{target}_UMAP_imagenet1000_3D_low_{seed_z}.png') 

        # plt.show()
        # print("3D Image generation using UMAP for imagenet 1000 classes - low finished!")






    '''
    import matplotlib.pyplot as plt

    #x = ['a', 'b', 'c', 'd']
    #y = [18.5, 13.3, 12.5, 14.2]
    bar = plt.bar(sort_index, avg_target_prob[sort_index].cpu().detach(), color = 'pink')
    plt.ylim(0, 1)
    plt.title('Sorting by Softmax probability')
    plt.xlabel('class index')
    plt.ylabel('Softmax probability')
    # 숫자 넣는 부분
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 12)

    plt.show()
    plt.savefig('./{}_softmax_criterion_fig.png'.format(target))
    '''
    # np.savetxt(os.path.join(f"{final_dir}/{target}",'_class_index_sortedby_softmax.csv'), sort_index, fmt='%.4f')
    # np.savetxt(os.path.join(f"{final_dir}/{target}",'_sort_softmax.csv'), avg_target_prob[sort_index].cpu().detach(), fmt='%.3f')
    # np.savetxt(os.path.join(f"{final_dir}/{target}",'_cosine_similarity_sortedby_softmax.csv'), cos_list, fmt='%.3f')

def get_diversity_loss(
    half_z_num, zs, dloss_function, pred_probs, net, resized_images_tensor):
    pairs = list(itertools.combinations(range(len(zs)), 2))
    random.shuffle(pairs)

    first_idxs = []
    second_idxs = []
    for pair in pairs[:half_z_num]:
        first_idxs.append(pair[0])
        second_idxs.append(pair[1])

    denom = F.pairwise_distance(zs[first_idxs, :], zs[second_idxs, :])
    denom = torch.sum(denom)

    if dloss_function == "softmax":
        num = torch.sum(
            F.pairwise_distance(pred_probs[first_idxs, :], pred_probs[second_idxs, :])
        )

    elif dloss_function == "features":
        _,_,_,_,_,features_out = net(resized_images_tensor)
        num = torch.sum(
            F.pairwise_distance(
                features_out[first_idxs, :].view(half_z_num, -1),
                features_out[second_idxs, :].view(half_z_num, -1),
            )
        )

    else:
        num = torch.sum(
            F.pairwise_distance(
                resized_images_tensor[first_idxs, :].view(half_z_num, -1),
                resized_images_tensor[second_idxs, :].view(half_z_num, -1),
            )
        )
    return num / denom

def run_biggan_am(
    init_embeddings,
    device,
    lr,
    dr,
    state_z,
    n_iters,
    z_num,
    dim_z,
    steps_per_z,
    min_clamp,
    max_clamp,
    G,
    feature_extractor,
    model,
    net,
    criterion,
    labels,
    dloss_function,
    half_z_num,
    alpha,
    target_class,
    intermediate_dir,
    use_noise_layer,
    total_class,
    img_size,
    class_dir,
    model_name,
    threshold_prob,
    threshold_ce,
    seed_z
):
    optim_embedding = init_embeddings.to(device)
    print("optim embedding : ",optim_embedding.shape)
   
    optim_comps = {
        "optim_embedding": optim_embedding,
        "use_noise_layer": use_noise_layer,
    }
    optim_embedding.requires_grad_()
  
    if use_noise_layer:
        noise_layer = nn.Linear(dim_z, dim_z).to(device)
        print("noise layer shape :",dim_z)
        noise_layer.train()
        optim_params += [params for params in noise_layer.parameters()]
        optim_comps["noise_layer"] = noise_layer

    T = torch.tensor(1.0)
    # for T experiment
    T.requires_grad_(True)

    total_loss = []
    total_T = []
    total_prob =[]
    logit_magnitude = []
    total_ce_loss = []
    best_acc=0.0
    best_err=100.0

    # For detailed embedding training
    threshold_ce = threshold_ce
    threshold_prob = threshold_prob

    #warm up stage - Total variance loss
    for epoch in range(n_iters):
        global total_epoch
        global over_thres_num
        print(target_class,'-th class image synthesis stage start!')
        labels = torch.LongTensor([target_class] * z_num).to(device)
        optim_comps = {
        "class_embedding":optim_embedding,
        "use_noise_layer": use_noise_layer,
        "T": T,
        "threshold_over": 0,
        "best_threshold" : 0.0,
        "best_epoch":0
        }


        
        
        optim_params = [optim_embedding]
        if use_noise_layer:
            noise_layer = nn.Linear(dim_z, dim_z).to(device)
            noise_layer.train()
            optim_params += [params for params in noise_layer.parameters()]
       
        optimizer = optim.Adam(optim_params+[T], lr=lr, weight_decay=dr)
        # For T experiment
        # optimizer = optim.Adam(optim_params, lr=lr, weight_decay=dr)
        prev_zs = None
        best_epoch = 0
        best_threshold = 0.0
        # flag = 0



        # For t-SNE decomposition -> one2one????
        zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)
        # zs = None
        
        for z_step in range(steps_per_z):
            optimizer.zero_grad()
            final_T = torch.clamp(T,min=0.001)

            # if flag == 1:
            # # if optimizer.param_groups[0]["lr"] == lr * 0.5:
            #     zs = prev_zs
            # else:
            # zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)

            # Image Generation : o2m approach
            # zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)
            # prev_zs = zs
            # store the random variable z
            # if optimizer.param_groups[0]["lr"] == lr * 0.1:
            #     prev_zs = zs
            print(f'learning rate: {optimizer.param_groups[0]["lr"]}')
            if use_noise_layer:
                noise_layer.train()
                z_hats = noise_layer(zs)
            else:
                z_hats = zs

            clamp_embedding = torch.clamp(optim_embedding,min_clamp, max_clamp)
            embedding_norm = clamp_embedding.norm(2, dim=0, keepdim=True)
            repeat_clamped_embedding = clamp_embedding.repeat(z_num, 1).to(device)
            gan_images_tensor = G(z_hats, repeat_clamped_embedding)
            flip = random.random() > 0.5
            if flip:
                gan_images_tensor = torch.flip(gan_images_tensor, dims = (3,))
            
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size # ViT CIFAR10 224 Flower 224, CelebA 128
            )
            small_noise = torch.randn_like(resized_images_tensor) * 0.005
            resized_images_tensor.add_(small_noise).clamp_(min=-1.0, max=1.0)

            if 'vit_inat' in model_name:
                pred_logits = net(resized_images_tensor)
            elif 'vit_cifar100' in model_name:
                model = model.to(device)
                pred_logits = model(resized_images_tensor)

            elif 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
                model = model.to(device)
                outputs = model(resized_images_tensor)
                pred_logits = outputs.logits

            elif 'vgg16' in model_name:
                # model = model.to(device)
                # outputs = model(resized_images_tensor)
                # pred_logits = outputs.logits
                
                pred_logits = net(resized_images_tensor)

            elif 'cct' in model_name or 'vgg' in model_name:
                pred_logits = net(resized_images_tensor)
            elif 'resnet34' in model_name:
                pred_logits,f1,f2,f3,f4,f5 = net(resized_images_tensor)
            else:
                pred_logits = net(resized_images_tensor)

            norm = torch.norm(pred_logits,p=2,dim=-1,keepdim=True) + 1e-7
            logit_norm = torch.div(pred_logits,norm)/final_T.cuda()
           
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()

            loss_ce = criterion(logit_norm, labels)
            origin_ce = criterion(pred_logits,labels)
            # loss = criterion(pred_logits,labels)
            diff1 = resized_images_tensor[:,:,:,:-1] - resized_images_tensor[:,:,:,1:]
            diff2 = resized_images_tensor[:,:,:-1,:] - resized_images_tensor[:,:,1:,:]
            diff3 = resized_images_tensor[:,:,1:,:-1] - resized_images_tensor[:,:,:-1,1:]
            diff4 = resized_images_tensor[:,:,:-1,:-1] - resized_images_tensor[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            # for reproducing FID
            
            # loss = loss_ce + 1.3e-4*loss_var
            # Original
            loss = loss_ce+  1.3e-3*loss_var

            if dloss_function:
                diversity_loss = get_diversity_loss(
                    half_z_num,
                    zs,
                    dloss_function,
                    pred_probs,
                    net,
                    resized_images_tensor,
                )
                loss += -alpha * diversity_loss

            loss.backward()
            optimizer.step()

            log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
            log_line += f"Average Target Probability:{avg_target_prob:.4f}\t"
            log_line += f"Temperature Probability: {T.item():.3f}\t"
            log_line += f"CE loss: {loss_ce.item():.4f}\t"
            log_line += f"Origin loss:{origin_ce.item():.4f}\t"
            log_line += f"total loss: {loss.item():.4f}\t"
            log_line += f"logit mag: {torch.sum(torch.norm(logit_norm,p=2)).item()/z_num:.4f}\t"

            print(log_line)
            total_loss.append(loss.item())
            total_ce_loss.append(loss_ce.item())
            total_T.append(T.item())
            total_prob.append(avg_target_prob)
            logit_magnitude.append(torch.sum(norm).item()/z_num)
    
            if intermediate_dir:
                if avg_target_prob > best_acc:
                    best_acc = avg_target_prob
                    # Editted here!! - by jihwan
                    if origin_ce.item() < best_err:
                        best_err = origin_ce.item()
                    global_step_id = epoch * steps_per_z + z_step
                    img_f = f"{global_step_id:0=7d}_{seed_z}.jpg"
                    output_image_path = f"{intermediate_dir}/{img_f}"
                    best_epoch = z_step
                    best_threshold = best_acc
                    if 0:
                        bar = plt.bar([i for i in range(pred_probs.shape[1])], pred_probs.mean(dim=0).cpu().detach(), color = 'pink')
                        plt.ylim(0, 1)
                        plt.title('The softmax outputs for checking overconfidence issue')
                        plt.xlabel('class index')
                        plt.ylabel('Softmax probability')
                        for rect in bar:
                            height = rect.get_height()
                            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 12)
                        plt.show()
                        plt.savefig('./'+output_image_path+'vanilaCE_softmax_criterion_fig.png')
                        plt.clf()
                    save_image(
                    gan_images_tensor, output_image_path, normalize=True, nrow=10
                    )
                    # np.save(f"{class_dir}/{target_class}_embedding_{best_epoch}_{seed_z}.npy", clamp_embedding.detach().cpu().numpy())
            torch.cuda.empty_cache()
            # Editted threshold part
            if best_acc >= threshold_prob:
            # if best_err <= threshold_ce:
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr * 0.5 
                # prev_zs = zs
                # if best_err <= threshold_ce:
                total_epoch += z_step
                over_thres_num += 1
                flag = 1
                # change threshold_over value as 2
                # optim_comps["threshold_over"] = True
                optim_comps["threshold_over"] = 2
                break   
            elif z_step == steps_per_z - 1:
                total_epoch += z_step
                optim_comps["threshold_over"] = 1
                optim_comps["best_epoch"] = best_epoch
                optim_comps["best_theshold"] = best_threshold
                # optim_comps["class_embedding"] = torch.from_numpy(np.load(f"{class_dir}/{target_class}_embedding_{best_epoch}_{seed_z}.npy")).to(device)
                break
                # else:
                #     continue
            #elif best_err <=0.9:
            #    break

            # For decomposition
            if z_step % 10 == 0:
                np.save(f"{class_dir}/{target_class}_embedding_{z_step // 10}_{seed_z}.npy", clamp_embedding.detach().cpu().numpy())

        np.save(f"{class_dir}/{target_class}_embedding_{seed_z}.npy", clamp_embedding.detach().cpu().numpy())
        np.save(f"{class_dir}/{target_class}_z_{seed_z}.npy",zs.detach().cpu().numpy())

    file_path = f"{intermediate_dir}/"
    np.savetxt(os.path.join(file_path,f'total_loss_{seed_z}.csv'), total_loss, fmt='%.4f')
    np.savetxt(os.path.join(file_path,f'CE_loss_{seed_z}.csv'), total_ce_loss, fmt='%.4f')
    np.savetxt(os.path.join(file_path,f'temperature_{seed_z}.csv'), total_T, fmt='%.3f')
    np.savetxt(os.path.join(file_path,f'gt_probability_{seed_z}.csv'), total_prob, fmt='%.3f')

    return optim_comps


def save_final_samples(
    optim_comps,
    min_clamp,
    max_clamp,
    device,
    state_z,
    num_final,
    dim_z,
    G,
    repeat_original_embedding,
    class_dir,
    target_class,
    seed_z
):
    optim_embedding = optim_comps["class_embedding"]
    optim_embedding_clamped = torch.clamp(optim_embedding, min_clamp, max_clamp)
    repeat_optim_embedding = optim_embedding_clamped.repeat(4, 1).to(device)

    if optim_comps["use_noise_layer"]:
        optim_comps["noise_layer"].eval()

    optim_imgs = []
    original_imgs = []

    torch.set_rng_state(state_z)

    for show_id in range(num_final):
        zs = torch.randn((num_final, dim_z), device=device, requires_grad=False)
        if optim_comps["use_noise_layer"]:
            with torch.no_grad():
                z_hats = optim_comps["noise_layer"](zs)
        else:
            z_hats = zs

        with torch.no_grad():
            optim_imgs.append(G(z_hats, repeat_optim_embedding))
            original_imgs.append(G(z_hats, repeat_original_embedding))

    final_image_path = f"{class_dir}/{target_class}_synthetic_{seed_z}.jpg"
    optim_imgs = torch.cat(optim_imgs, dim=0)
    save_image(optim_imgs, final_image_path, normalize=True, nrow=4)
    np.save(
        f"{class_dir}/{target_class}_embedding_{seed_z}.npy",
        optim_embedding.detach().cpu().numpy()
    )
    if optim_comps["use_noise_layer"]:
        torch.save(
            optim_comps["noise_layer"].state_dict(),
            f"{class_dir}/{target_class}_noise_layer_{seed_z}.pth",
        )

    original_image_path = f"{class_dir}/{target_class}_original_{seed_z}.jpg"
    original_imgs = torch.cat(original_imgs, dim=0)
    save_image(original_imgs, original_image_path, normalize=True, nrow=4)


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

def denorm_cifar(image_tensor,dataset):
    # CIFAR10의 채널별 평균 및 표준편차
    cifar10_means = torch.tensor([0.4914, 0.4822, 0.4465])
    cifar10_stds = torch.tensor([0.2023, 0.1994, 0.2010])

    # 이미지 텐서의 채널별 평균 및 표준편차 계산
    tensor_means = image_tensor.mean(dim=[1,2])
    tensor_stds = image_tensor.std(dim=[1,2])
    # 이미지 텐서 정규화
    normalized_tensor = (image_tensor - tensor_means[:, None, None]) / tensor_stds[:,None, None]
    
    # CIFAR10의 평균 및 분산을 적용하여 값을 조정

    device = normalized_tensor.device
    cifar10_stds = cifar10_stds.to(device)
    cifar10_means = cifar10_means.to(device)
    adjusted_tensor = normalized_tensor * cifar10_stds[:, None, None] + cifar10_means[:, None, None]

    # adjusted_tensor = normalized_tensor * cifar10_stds[:, None, None] + cifar10_means[:, None, None]

    return adjusted_tensor



def save_instance_image(optim_comps,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,img_len,total_class,global_epoch,net,model,model_name,threshold_img,threshold_prob,device,seed_z):
    final_dir_fake = 'Fake_'+final_dir+'/'+str(target_class)+'/'
    img_len_per_class = img_len/total_class
    num_generations = int(img_len_per_class/batch_size) #flower 250 cifar100 500 
    # 한번에 batch개 만큼 생성하며, num_generation 만큼 순회해서 이미지 생성해야 함
    # for i in range(num_generations):
    i = 0
    # j = 0
    num_generations_i =num_generations
    num_generations_j = num_generations
    success_zs = []
    fail_zs = []
    criterion = nn.CrossEntropyLoss()
    while i < num_generations_i:
        zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
        if optim_comps["use_noise_layer"]:
            noise_layer = nn.Linear(dim_z,dim_z)
            noise_layer.load_state_dict(torch.load(f"{class_dir}/{target_class}_noise_layer.pth"))

            noise_layer = noise_layer.cuda()
            zs = noise_layer(zs)

        with torch.no_grad():
            # batch_size = 1
            repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
            
            # print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
            gan_images_tensor = G(zs, repeat_class_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size #Flower 224, CelebA 128
                )
        targets = torch.LongTensor([target_class] * batch_size)
        
        images = resized_images_tensor.data.clone()

        if 'cifar' in dataset:
            images = nn.functional.interpolate(
                images, size=32 #Flower 224, CelebA 128 
            )
        else:
             images = nn.functional.interpolate(
                images, size=img_size
            )
        
        # load pretrained classifier
        resized_images_tensor_check = nn.functional.interpolate(
                gan_images_tensor, size=img_size # ViT CIFAR10 224 Flower 224, CelebA 128
            )
        small_noise = torch.randn_like(resized_images_tensor_check) * 0.005
        resized_images_tensor_check.add_(small_noise).clamp_(min=-1.0, max=1.0)

        correct = 0
        total = 0 
        pred_probs = 0.0
        pred_logits = None
        with torch.no_grad():

            if 'vit_inat' in model_name:
                pred_logits = net(resized_images_tensor_check)
            elif 'vit_cifar100' in model_name:
                model = model.to(device)
                pred_logits = model(resized_images_tensor_check)

            elif 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
                model = model.to(device)
                outputs = model(resized_images_tensor_check)
                pred_logits = outputs.logits

            elif 'vgg16' in model_name:
                # model = model.to(device)
                # outputs = model(resized_images_tensor)
                # pred_logits = outputs.logits
                
                pred_logits = net(resized_images_tensor_check)

            elif 'cct' in model_name or 'vgg' in model_name:
                pred_logits = net(resized_images_tensor_check)
            elif 'resnet34' in model_name:
                pred_logits,f1,f2,f3,f4,f5 = net(resized_images_tensor_check)
            else:
                pred_logits = net(resized_images_tensor_check)

            # _, predicted = torch.max(pred_logits.data, 1)

            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            labels = targets.to(device)
            pred_ce = criterion(pred_logits, labels).item()
            # breakpoint()
            target_prob = pred_probs[:, target_class].item()
            # total += targets.size(0)
            # correct += (predicted == targets.cuda()).sum().item()

            pred_probs = target_prob

        if pred_probs >= threshold_prob and i <num_generations and pred_ce <= threshold_img  :
            numpy_zs = zs.cpu().numpy()
            success_zs.append(numpy_zs)
            for id in range(images.shape[0]):
                class_id = str(targets[id].item()).zfill(2)
                # print(images[id])
                # breakpoint()
                if 'cifar' in dataset:
                    image = images[id].reshape(3,32,32)
                else:
                    image = images[id].reshape(3,img_size,img_size)

                # Storing Options
                # A. w/o denorm -> cifar10 statistics
                final_dir_no_denorm = 'Fake_'+final_dir+'_thres_both' +'_no_denorm'+'/'+str(target_class)+'/'
                if not os.path.exists(final_dir_no_denorm):
                    os.makedirs(final_dir_no_denorm)

                vutils.save_image(image,os.path.join(final_dir_no_denorm,'{}_ge_{}_output_{}_{}'.format(global_epoch,i,id,seed_z))+'.png',normalize=True,scale_each=True,nrow=1)

                # B. w/ denorm normalize, scale_each = False -> 2개
                image_denorm = denormalize(image,dataset)
                # image_np = images[id].data.cpu().numpy()
                # pil_images = torch.from_numpy(image_np)

                # final_dir_norm_scale_false = 'Fake_'+final_dir+'_norm_scale_false'+'/'+str(target_class)+'/'

                # if not os.path.exists(final_dir_norm_scale_false):
                #     os.makedirs(final_dir_norm_scale_false)

                # vutils.save_image(image_denorm,os.path.join(final_dir_norm_scale_false,'{}_ge_{}_output_{}_{}'.format(global_epoch,i,id,seed_z))+'.png',normalize=False,scale_each=False,nrow=1)
                # C. w/ denorm normalize, scale_each = True

                final_dir_fake = 'Fake_'+final_dir+'_thres_both'+'/'+str(target_class)+'/'
                if not os.path.exists(final_dir_fake):
                    os.makedirs(final_dir_fake)
                # denorm 한거 안한거 check
                # denorm 버전 일단 생성해보고 norm 버전도 한번 생성해봐야 할 듯?
                # 이부분 함 체크해보고...
                # vutils.save_image(image,os.path.join(final_dir,'{}_ge_{}_output_{}'.format(global_epoch,i,id))+'.png',normalize=False,scale_each=False,nrow=1)
                # print(f'{i}th image generated!')
                vutils.save_image(image_denorm,os.path.join(final_dir_fake,'{}_ge_{}_output_{}_{}'.format(global_epoch,i,id,seed_z))+'.png',normalize=True,scale_each=True,nrow=1)

                # image_denorm_cifar10 = denorm_cifar(image, dataset)

                # # D. w/ denorm like cifar10 norm scale_each = True
                # final_dir_denorm_cifar_t= 'Fake_'+final_dir+'_denorm_cifar_t'+'/'+str(target_class)+'/'

                # if not os.path.exists(final_dir_denorm_cifar_t):
                #     os.makedirs(final_dir_denorm_cifar_t)

                # vutils.save_image(image_denorm_cifar10,os.path.join(final_dir_denorm_cifar_t,'{}_ge_{}_output_{}'.format(global_epoch,i,id))+'.png',normalize=True,scale_each=True,nrow=1)
                # # E. w/ denorm like cifar10 norm scale_each = False
                # final_dir_denorm_cifar_f= 'Fake_'+final_dir+'_denorm_cifar_f'+'/'+str(target_class)+'/'

                # if not os.path.exists(final_dir_denorm_cifar_f):
                #     os.makedirs(final_dir_denorm_cifar_f)

                # vutils.save_image(image_denorm_cifar10,os.path.join(final_dir_denorm_cifar_f,'{}_ge_{}_output_{}'.format(global_epoch,i,id))+'.png',normalize=False,scale_each=False,nrow=1)
            
            i += 1
        # if pred_ce <= threshold_img and j <num_generations:
        #     numpy_zs = zs.cpu().numpy()
        #     success_zs.append(numpy_zs)
        #     for id in range(images.shape[0]):
        #         class_id = str(targets[id].item()).zfill(2)
        #         # print(images[id])
        #         # breakpoint()
        #         if 'cifar' in dataset:
        #             image = images[id].reshape(3,32,32)
        #         else:
        #             image = images[id].reshape(3,img_size,img_size)

        #         # Storing Options
        #         # A. w/o denorm -> cifar10 statistics
        #         final_dir_no_denorm = 'Fake_'+final_dir+'_thres_ce' +'_no_denorm'+'/'+str(target_class)+'/'
        #         if not os.path.exists(final_dir_no_denorm):
        #             os.makedirs(final_dir_no_denorm)

        #         vutils.save_image(image,os.path.join(final_dir_no_denorm,'{}_ge_{}_output_{}_{}'.format(global_epoch,j,id,seed_z))+'.png',normalize=True,scale_each=True,nrow=1)

        #         # B. w/ denorm normalize, scale_each = False -> 2개
        #         image_denorm = denormalize(image,dataset)
        #         # image_np = images[id].data.cpu().numpy()
        #         # pil_images = torch.from_numpy(image_np)

        #         # final_dir_norm_scale_false = 'Fake_'+final_dir+'_norm_scale_false'+'/'+str(target_class)+'/'

        #         # if not os.path.exists(final_dir_norm_scale_false):
        #         #     os.makedirs(final_dir_norm_scale_false)

        #         # vutils.save_image(image_denorm,os.path.join(final_dir_norm_scale_false,'{}_ge_{}_output_{}_{}'.format(global_epoch,i,id,seed_z))+'.png',normalize=False,scale_each=False,nrow=1)
        #         # C. w/ denorm normalize, scale_each = True
        #         final_dir_fake = 'Fake_'+final_dir+'_thres_ce'+'/'+str(target_class)+'/'
        #         if not os.path.exists(final_dir_fake):
        #             os.makedirs(final_dir_fake)
        #         # denorm 한거 안한거 check
        #         # denorm 버전 일단 생성해보고 norm 버전도 한번 생성해봐야 할 듯?
        #         # 이부분 함 체크해보고...
        #         # vutils.save_image(image,os.path.join(final_dir,'{}_ge_{}_output_{}'.format(global_epoch,i,id))+'.png',normalize=False,scale_each=False,nrow=1)
        #         # print(f'{i}th image generated!')
        #         vutils.save_image(image_denorm,os.path.join(final_dir_fake,'{}_ge_{}_output_{}_{}'.format(global_epoch,j,id,seed_z))+'.png',normalize=True,scale_each=True,nrow=1)
            
        #     j += 1
        if (pred_ce > threshold_img and pred_probs < threshold_prob):
            numpy_zs = zs.cpu().numpy()
            fail_zs.append(numpy_zs)
    # fail z experiment
    # fail_zs.shape : N * bs * 128

    fail_probs = []
    for i in range(len(fail_zs)):

        zs = torch.from_numpy(fail_zs[i]).cuda()
        with torch.no_grad():
            # batch_size = 1
            repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
            
            # print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
            gan_images_tensor = G(zs, repeat_class_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size #Flower 224, CelebA 128
                )
        targets = torch.LongTensor([target_class] * batch_size)
        
        images = resized_images_tensor.data.clone()

        if 'cifar' in dataset:
            images = nn.functional.interpolate(
                images, size=32 #Flower 224, CelebA 128 
            )
        else:
             images = nn.functional.interpolate(
                images, size=img_size
            )
        
        # load pretrained classifier
        resized_images_tensor_check = nn.functional.interpolate(
                gan_images_tensor, size=img_size # ViT CIFAR10 224 Flower 224, CelebA 128
            )
        small_noise = torch.randn_like(resized_images_tensor_check) * 0.005
        resized_images_tensor_check.add_(small_noise).clamp_(min=-1.0, max=1.0)

        correct = 0
        total = 0 
        pred_probs = 0.0
        pred_logits = None
        with torch.no_grad():

            if 'vit_inat' in model_name:
                pred_logits = net(resized_images_tensor_check)
            elif 'vit_cifar100' in model_name:
                model = model.to(device)
                pred_logits = model(resized_images_tensor_check)

            elif 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
                model = model.to(device)
                outputs = model(resized_images_tensor_check)
                pred_logits = outputs.logits

            elif 'vgg16' in model_name:
                # model = model.to(device)
                # outputs = model(resized_images_tensor)
                # pred_logits = outputs.logits
                
                pred_logits = net(resized_images_tensor_check)

            elif 'cct' in model_name or 'vgg' in model_name:
                pred_logits = net(resized_images_tensor_check)
            elif 'resnet34' in model_name:
                pred_logits,f1,f2,f3,f4,f5 = net(resized_images_tensor_check)
            else:
                pred_logits = net(resized_images_tensor_check)

            # _, predicted = torch.max(pred_logits.data, 1)

            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            labels = targets.to(device)
            pred_ce = criterion(pred_logits, labels).item()
            # breakpoint()
            target_prob = pred_probs[:, target_class].item()
            # total += targets.size(0)
            # correct += (predicted == targets.cuda()).sum().item()

            pred_probs = target_prob

            fail_probs.append(pred_probs)
        
    plt.hist(fail_probs, bins=20, range=(0,1), edgecolor= "black", alpha=0.7)
    plt.title(f"Histogram of Class {target_class} fail zs ")
    plt.xlabel('Probability')
    plt.ylabel('Count')

    fail_probs.sort(reverse=True)

    plt.savefig(f'{class_dir}/{target_class}_fail_zs_histogram_{seed_z}_max_{fail_probs[0]}.png')

    plt.clf()




    print(f"{target_class}th class {img_len_per_class} image generation done..")
    print(f"# of success zs:{len(success_zs)}")
    print(f"# of fail zs:{len(fail_zs)}")
    np.save(f'{class_dir}/success_zs_{seed_z}.npy',success_zs)
    np.save(f'{class_dir}/fail_zs_{seed_z}.npy',fail_zs)



def main():
    opts = yaml.safe_load(open("opts.yaml"))
    seed_z = opts["seed_z"]
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

    init_method = opts["init_method"]
    print(f"Initialization method: {init_method}")
    if init_method == "target":
        noise_std = opts["noise_std"]
        print(f"The noise std is: {noise_std}")
    else:
        noise_std = None

    # Load the models.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    print("Loading the BigGAN generator model...")
    resolution = opts["resolution"]
    config = get_config(resolution)
    start_time = time.time()
    G = BigGAN.Generator(**config)
    if resolution == 128:
        biggan_weights = "./generator_pretrained_weights/138k/G_ema.pth"
    else:
        biggan_weights = "./generator_pretrained_weights/biggan_256_weights.pth"

    G.load_state_dict(torch.load(f"{biggan_weights}"), strict=False)
    G = nn.DataParallel(G).to(device)
    G.eval()

    model_name = opts["model"]

    if 'vit_inat' in model_name:
        feature_extractor=None
        model=None
        net = nn.DataParallel(load_net(model_name)).to(device)
    elif 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
        feature_extractor,model = load_net(model_name)
        net = None

    else:
        feature_extractor=None
        model=None
        net = nn.DataParallel(load_net(model_name)).to(device)
        net.eval()

    z_num = opts["z_num"]
    dloss_function = opts["dloss_function"]
    half_z_num = z_num // 2
    print(f"BigGAN initialization time: {time.time() - start_time}")

    # Set up optimization.
    init_num = opts["init_num"]
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    target_class = opts["target_class"]
    batch_size = opts["generation_batch_size"]

    final_dir = opts["final_dir"] # store all class images for evaluating the IS and FID scores
    if final_dir:
        print(f"Saving final samples in {final_dir}.")
        os.makedirs(final_dir, exist_ok=True)

    original_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
    original_embeddings = torch.from_numpy(original_embeddings)
    
    criterion = nn.CrossEntropyLoss()
    labels = torch.LongTensor([target_class] * z_num).to(device)
    state_z = torch.get_rng_state()


    global total_epoch
    total_epoch = 0
    global over_thres_num
    over_thres_num = 0
    seed_list =[]



    # for target_class in range(int(opts["total_class"])):
    # for target_class in range(11,100):
    # for target_class in [56]
    for target_class in [1,7]:
        epoch = 0
        # for epoch in range(10):
        while epoch < 10:
            seed_z = change_seed()
            print(f'seed:{seed_z}')
            seed_list.append(seed_z)
            init_embeddings = get_initial_embeddings(
            resolution,
            init_method,
            model_name,
            init_num,
            min_clamp,
            max_clamp,
            dim_z,
            G,
            net,
            feature_extractor,
            model,
            target_class,
            noise_std,
            opts["img_size"]
            )
            intermediate_dir = final_dir+str(seed_z)+'/'+str(target_class)+'/intermediate' # store intermediate images
            class_dir = final_dir+str(seed_z)+'/'+str(target_class) # store final embedding and image of each class (target class)
            # class_dir = final_dir+'/'+str(target_class)
            if class_dir:
                print(f"Saving class-wise samples in {class_dir}.")
                os.makedirs(class_dir, exist_ok=True)

            if intermediate_dir:
                print(f"In class_dir, Saving intermediate samples in {intermediate_dir}.")
                os.makedirs(intermediate_dir, exist_ok=True)

            original_embedding_clamped = torch.clamp(
            original_embeddings[target_class].unsqueeze(0), min_clamp, max_clamp
            )
            num_final = 4
            repeat_original_embedding = original_embedding_clamped.repeat(num_final, 1).to(device)

            #image synthesis start!
            optim_comps = run_biggan_am(
                init_embeddings,
                device,
                opts["lr"],
                opts["dr"],
                state_z,
                opts["n_iters"],
                z_num,
                dim_z,
                opts["steps_per_z"],
                min_clamp,
                max_clamp,
                G,
                feature_extractor,
                model,
                net,
                criterion,
                labels,
                dloss_function,
                half_z_num,
                opts["alpha"],
                target_class,
                intermediate_dir,
                opts["use_noise_layer"],
                opts["total_class"],
                opts["img_size"],
                class_dir,
                model_name,
                opts["threshold_prob"],
                opts["threshold_ce"],
                seed_z
            )

            # Get signals 

            if optim_comps["threshold_over"] > 0:
                epoch += 1
            else:
                continue

            if class_dir:
                save_final_samples(
                    optim_comps,
                    min_clamp,
                    max_clamp,
                    device,
                    state_z,
                    num_final,
                    dim_z,
                    G,
                    repeat_original_embedding,
                    class_dir,
                    target_class,
                    seed_z
                )

            # if final_dir:
            #     save_instance_image(optim_comps,
            #     G,
            #     opts["img_size"],
            #     batch_size,
            #     target_class,
            #     z_num,
            #     dim_z,
            #     class_dir,
            #     final_dir,
            #     opts["dataset"],
            #     opts["img_len"],
            #     opts["total_class"],
            #     epoch,
            #     net,
            #     model,
            #     model_name,
            #     opts["threshold_img"],
            #     opts["threshold_prob"],
            #     device,
            #     seed_z
            # )



            # For 
            if 1:
                decompose_latent(optim_comps,target_class,G,net,z_num,dim_z,final_dir,device,model_name,class_dir,seed_z)

        np.savetxt(os.path.join(class_dir,f'{target_class}_seed_list.csv'), seed_list, fmt='%.4f')       
    #     print(f"total epoch:{total_epoch}, over thres class num: {over_thres_num}")
    # print(f"average epoch per class {total_epoch // over_thres_num}") 
        

if __name__ == "__main__":
    main()
