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
import torchvision
from transformers import pipeline
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
from skimage.util import random_noise
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def change_seed():
    seed_z = np.random.randint(1000000)
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

def get_initial_embeddings(
    resolution,
    init_method,
    model_name,
    init_num,
    min_clamp,
    max_clamp,
    dim_z,
    G,
    feature_extractor,
    model,
    net,
    target_class,
    noise_std,
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
        for i in range(1000):
            class_embedding = class_embeddings_clamped[i]
            repeat_class_embedding = class_embedding.repeat(num_samples, 1)
            final_z = torch.randn((num_samples, dim_z), requires_grad=False)

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
                    pred_logits,_,_,_,_,_ = net(resized_images_tensor)
                    pred_logits = net(resized_images_tensor)
            
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

        init_embeddings = (
            class_embeddings[target_class].unsqueeze(0).repeat(init_num, 1)
        )
        init_embeddings += torch.randn((init_num, embedfding_dim)) * noise_std

    return init_embeddings

def decompose_latent(optim_comps,target,G,net,z_num,dim_z,final_dir,device,model_name):
    repeat_optim_embedding = optim_comps["class_embedding"].repeat(4, 1).to(device)
    class_embeddings = np.load(f"biggan_embeddings_256.npy")
    class_embeddings = torch.from_numpy(class_embeddings).to(device)

    if optim_comps["use_noise_layer"]:
        optim_comps["noise_layer"].eval()

    optim_imgs = []
    original_imgs = []
    avg_list=[]

    zs = torch.randn((z_num, dim_z), device=device, requires_grad=False)
    embedding = optim_comps["class_embedding"]
    repeat_embedding = embedding.repeat(z_num,1).to(device)
    gan_embedding_images = G(zs, repeat_embedding)
    vutils.save_image(gan_embedding_images,os.path.join(final_dir,'{}class_optimal_embedding'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)
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

    sort_index = np.argsort(avg_target_prob.cpu().detach())
    print(sort_index)
    cos = nn.CosineSimilarity(dim=0)

    optim_embedding = torch.zeros_like(embedding).to(device)
    cossim_list = []
    for i in range(1000):
        cos_out = cos(embedding,class_embeddings[i])
        #print("cosine similarity: ",cos_out.item())
        cossim_list.append(cos_out.item())
    cos_sort_index = np.argsort(cossim_list)
    #cossim_list = cossim_list.sort()
    print(cossim_list)
    print("cosine similarity sort: ",cos_sort_index)
    
    #cosine similarity map 기준으로 embedding vector 더했을때
    for i in range(1,1000):
        if cossim_list[cos_sort_index[-i]] <= 0:
            break
        class_embedding = class_embeddings[cos_sort_index[-i]] # top 10 embedding
        cos_out = cos(embedding,class_embedding)
        print("sort index : ",cos_sort_index[-i],"softmax probability: ",avg_target_prob[cos_sort_index[-i]].item(),"cosine similarity: ",cos_out.item())
        optim_embedding = optim_embedding + (avg_target_prob[cos_sort_index[-i]]*class_embedding)
    if optim_comps["use_noise_layer"]:
         with torch.no_grad():
            z_hats = optim_comps["noise_layer"](zs)
    else:
        z_hats = zs
    repeat_optimal_embedding = optim_embedding.repeat(z_num, 1).to(device)
    with torch.no_grad():
        gan_images = G(z_hats, repeat_optimal_embedding)
        output = cos(embedding, optim_embedding)
        print("cosine similarity: ",output.item())

    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    #cossim_list = cossim_list.sort()
    vutils.save_image(gan_images,os.path.join(final_dir,'{}_cosine_similarity_latent_decompose'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)
    np.save(f"{final_dir}/{target}_cosine_similarity_decompose_embedding.npy", optim_embedding.detach().cpu().numpy())
    np.savetxt(os.path.join(f"{final_dir}/{target}",'_class_index_sortedby_cosinesim.csv'), cos_sort_index, fmt='%.4f')
    np.savetxt(os.path.join(f"{final_dir}/{target}",'_sort_softmax_sortedby_cosinesim.csv'), avg_target_prob[cos_sort_index].cpu().detach(), fmt='%.3f')
    np.savetxt(os.path.join(f"{final_dir}/{target}",'_cosine_similarity_sortedby_cosinesim.csv'),cossim_list, fmt='%.3f')

    #softmax값 기준으로 embedding vector 더했을때    
    cos_list = []
    for i in range(1,1000):
        class_embedding = class_embeddings[sort_index[-i]] # top 10 embedding
        cos_out = cos(embedding,class_embedding)
        cos_list.append(cos_out.item())
        print("sort index : ",sort_index[-i],"softmax probability: ",avg_target_prob[sort_index[-i]].item(),"cosine similarity: ",cos_out.item())
        optim_embedding = optim_embedding + (avg_target_prob[sort_index[-i]]*class_embedding)
    if optim_comps["use_noise_layer"]:
         with torch.no_grad():
            z_hats = optim_comps["noise_layer"](zs)
    else:
        z_hats = zs
    repeat_optimal_embedding = optim_embedding.repeat(z_num, 1).to(device)
    with torch.no_grad():
        gan_images = G(z_hats, repeat_optimal_embedding)
        output = cos(embedding, optim_embedding)
        print("cosine similarity: ",output.item())

    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    vutils.save_image(gan_images,os.path.join(final_dir,'{}_latent_decompose'.format(target))+'.png',normalize=True,scale_each=True,nrow=10)
    np.save(f"{final_dir}/{target}_decompose_embedding.npy", optim_embedding.detach().cpu().numpy())
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
    np.savetxt(os.path.join(f"{final_dir}/{target}",'_class_index_sortedby_softmax.csv'), sort_index, fmt='%.4f')
    np.savetxt(os.path.join(f"{final_dir}/{target}",'_sort_softmax.csv'), avg_target_prob[sort_index].cpu().detach(), fmt='%.3f')
    np.savetxt(os.path.join(f"{final_dir}/{target}",'_cosine_similarity_sortedby_softmax.csv'), cos_list, fmt='%.3f')

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
    threshold
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
    T.requires_grad_(True)

    total_loss = []
    total_T = []
    total_prob =[]
    logit_magnitude = []
    best_acc=0.0
    best_err=100.0
    threshold = threshold

    #warm up stage - Total variance loss
    for epoch in range(n_iters):
        print(target_class,'-th class image synthesis stage start!')
        labels = torch.LongTensor([target_class] * z_num).to(device)
        optim_comps = {
        "class_embedding":optim_embedding,
        "use_noise_layer": use_noise_layer,
        "T": T
        }
        optim_params = [optim_embedding]
        if use_noise_layer:
            noise_layer = nn.Linear(dim_z, dim_z).to(device)
            noise_layer.train()
            optim_params += [params for params in noise_layer.parameters()]

        #optimizer = optim.SGD(optim_params+[T], lr=lr, momentum=0.9)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        optimizer = optim.Adam(optim_params+[T], lr=lr, weight_decay=dr)

        for z_step in range(steps_per_z):
            optimizer.zero_grad()
            final_T = torch.clamp(T,min=0.001)
            
            zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)
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


            if 'vit_cifar100' in model_name:
                model = model.to(device)
                pred_logits = model(resized_images_tensor)

            elif 'vit' in model_name or 'beit' in model_name:
                model = model.to(device)
                outputs = model(resized_images_tensor)
                pred_logits = outputs.logits

            elif 'cct' in model_name or 'vgg' in model_name:
                pred_logits = net(resized_images_tensor)

            elif 'resnet34' in model_name:
                pred_logits,f1,f2,f3,f4,f5 = net(resized_images_tensor)
            else:
                pred_logits = net(resized_images_tensor)

            norm = torch.norm(pred_logits,p=2,dim=-1,keepdim=True) + 1e-7
            logit_norm = torch.div(pred_logits,norm)#/final_T.cuda()
           
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()

            #loss = criterion(logit_norm, labels)
            loss = criterion(pred_logits,labels)

            diff1 = resized_images_tensor[:,:,:,:-1] - resized_images_tensor[:,:,:,1:]
            diff2 = resized_images_tensor[:,:,:-1,:] - resized_images_tensor[:,:,1:,:]
            diff3 = resized_images_tensor[:,:,1:,:-1] - resized_images_tensor[:,:,:-1,1:]
            diff4 = resized_images_tensor[:,:,:-1,:-1] - resized_images_tensor[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss +=  1.3e-3*loss_var

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
            #scheduler.step()

            log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
            log_line += f"Average Target Probability:{avg_target_prob:.4f}\t"
            log_line += f"Temperature Probability: {T.item():.3f}\t"
            log_line += f"CE loss: {loss.item():.4f}\t"
            log_line += f"logit mag: {torch.sum(torch.norm(logit_norm,p=2)).item()/z_num:.4f}\t"

            print(log_line)
            total_loss.append(loss.item())
            total_T.append(T.item())
            total_prob.append(avg_target_prob)
            logit_magnitude.append(torch.sum(norm).item()/z_num)
    
            if intermediate_dir:
                if avg_target_prob > best_acc:
                    best_acc = avg_target_prob
                    best_err = loss.item()
                    global_step_id = epoch * steps_per_z + z_step
                    img_f = f"{global_step_id:0=7d}.jpg"
                    output_image_path = f"{intermediate_dir}/{img_f}"
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
            torch.cuda.empty_cache()
            if best_acc >= threshold:
                break
            #elif best_err <=0.9:
            #    break

        np.save(f"{class_dir}/{target_class}_embedding.npy", clamp_embedding.detach().cpu().numpy())

    file_path = f"{intermediate_dir}/"
    np.savetxt(os.path.join(file_path,'CE_loss.csv'), total_loss, fmt='%.4f')
    np.savetxt(os.path.join(file_path,'temperature.csv'), total_T, fmt='%.3f')
    np.savetxt(os.path.join(file_path,'gt_probability.csv'), total_prob, fmt='%.3f')

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
    target_class
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

    final_image_path = f"{class_dir}/{target_class}_synthetic.jpg"
    optim_imgs = torch.cat(optim_imgs, dim=0)
    save_image(optim_imgs, final_image_path, normalize=True, nrow=4)
    np.save(
        f"{class_dir}/{target_class}_embedding.npy",
        optim_embedding.detach().cpu().numpy()
    )
    if optim_comps["use_noise_layer"]:
        torch.save(
            optim_comps["noise_layer"].state_dict(),
            f"{class_dir}/{target_class}_noise_layer.pth",
        )

    original_image_path = f"{class_dir}/{target_class}_original.jpg"
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


def save_instance_image(optim_comps,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,img_len,total_class,global_epoch):
    final_dir = 'Fake_'+final_dir+'/'+str(target_class)+'/'
    img_len_per_class = img_len/total_class
    num_generations = int(img_len_per_class/(batch_size*5)) #flower 250 cifar100 500 
    print("num generation : ",num_generations)
    for i in range(num_generations):
        zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
        if optim_comps["use_noise_layer"]:
            noise_layer = nn.Linear(dim_z,dim_z)
            noise_layer.load_state_dict(torch.load(f"{class_dir}/{target_class}_noise_layer.pth"))

            noise_layer = noise_layer.cuda()
            zs = noise_layer(zs)

        with torch.no_grad():
            repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
            print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
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
        for id in range(images.shape[0]):
            class_id = str(targets[id].item()).zfill(2)
            if 'cifar' in dataset:
                image = images[id].reshape(3,32,32)
            else:
                image = images[id].reshape(3,img_size,img_size)
            image = denormalize(image,dataset)
            image_np = images[id].data.cpu().numpy()
            pil_images = torch.from_numpy(image_np)

            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            vutils.save_image(image,os.path.join(final_dir,'{}_ge_{}_output_{}'.format(global_epoch,i,id))+'.png',normalize=True,scale_each=True,nrow=1)

def test(dataset,net,device):
    print('==> Preparing data..')
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='../data/CIFAR10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=256, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='../data/CIFAR10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=2)

    elif datset =='cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='../data/CIFAR100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='../data/CIFAR100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    net.cuda().eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    best_acc=0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            resized_images_tensor = nn.functional.interpolate(
                inputs, size=224
            )
            outputs = net(inputs)
            #print("outputs shape : ",outputs.shape)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        #print(len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print("test accuracy :  ",100.*correct/total, correct, total)

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
        biggan_weights = "pretrained_weights/138k/G_ema.pth"
    else:
        biggan_weights = "pretrained_weights/biggan_256_weights.pth"

    G.load_state_dict(torch.load(f"{biggan_weights}"), strict=False)
    G = nn.DataParallel(G).to(device)
    G.eval()

    model_name = opts["model"]
    if 'vit' in model_name or 'swin' in model_name:
        feature_extractor,model = load_net(model_name)
        net = None

    else:
        feature_extractor=None
        model=None
        net = nn.DataParallel(load_net(model_name)).to(device)
        net.eval()

    if 0:
        test(opts["dataset"],net,device)

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

    for target_class in range(int(opts["total_class"])):
        for epoch in range(1):
            change_seed()
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
            )
            intermediate_dir = final_dir+'/'+str(target_class)+'/intermediate' # store intermediate images
            class_dir = final_dir+'/'+str(target_class) # store final embedding and image of each class (target class)

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
                opts["threshold"]
            )

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
                    target_class
                )

            if final_dir:
                save_instance_image(optim_comps,
                G,
                opts["img_size"],
                batch_size,
                target_class,
                z_num,
                dim_z,
                class_dir,
                final_dir,
                opts["dataset"],
                opts["img_len"],
                opts["total_class"],
                epoch
            )
            if 0:
                decompose_latent(optim_comps,target_class,G,net,z_num,dim_z,final_dir,device,model_name)
        

if __name__ == "__main__":
    main()
