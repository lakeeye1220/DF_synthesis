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
from torchvision import datasets, transforms
import glob
import os
import argparse

def get_diversity_loss(half_z_num, zs, dloss_function, pred_probs, alexnet_conv5, resized_images_tensor):

    pairs = list(itertools.combinations(range(len(zs)), 2))
    random.shuffle(pairs)

    first_idxs = []
    second_idxs = []
    for pair in pairs[:half_z_num]:
        first_idxs.append(pair[0])
        second_idxs.append(pair[1])

    denom = F.pairwise_distance(zs[first_idxs, :], zs[second_idxs, :])

    if dloss_function == "softmax":

        num = torch.sum(
            F.pairwise_distance(pred_probs[first_idxs, :], pred_probs[second_idxs, :])
        )

    elif dloss_function == "features":

        features_out = alexnet_conv5(resized_images_tensor)
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

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        tot_correct = []
        print(tot_correct)
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            tot_correct.append(correct_k)
    return tot_correct

def test(net,model, arch, final_dir,itp_policy):
    batch_size = 128
    criterion_CE = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(final_dir, transform=transforms.Compose([
                            transforms.Pad(4, padding_mode='reflect'),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            # transforms.Resize(224),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
                batch_size=batch_size, shuffle=True)

    # net.eval()

    if 'resnet' in arch or 'mlp_mixer' in arch:
        net.eval()
    else:
        model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            if 'resnet34' in arch:
                images = nn.functional.interpolate(
                    images, size=32,
                    mode=itp_policy
                )
                outputs,_,_,_,_,_ = net(images)
            elif 'vit_cifar100'== arch:
                images = nn.functional.interpolate(
                    images, size=224,
                    mode=itp_policy
                )
                # model.eval()
                outputs = model(images)
            elif 'vit' in arch or 'deit' in arch:
                images = nn.functional.interpolate(
                    images, size=224,
                    mode=itp_policy
                )
                # encoding = image_processor()
                outputs = model(images)
                outputs = outputs.logits
            else:
                images = nn.functional.interpolate(
                    images, size=224,
                    mode=itp_policy
                )
                outputs = net(images)
            # outputs,_,_,_,_,_ = net(images)
            print("label : ",labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            print("predicted : ",predicted)
            total += labels.size(0)
            correct += (predicted.cuda() == labels.cuda()).sum().item()
    print(f'Accuracy of the network on the {len(train_loader.dataset)} test images: {100 * correct // total} %')
    return float(correct / total)


def change_seed():
    #Function for change seed
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
    target_class
):
    # For initalize the class embedding, we use this function.

    class_embeddings = np.load(f"biggan_embeddings_{resolution}.npy") #load ImageNet 1k class embdding 
    class_embeddings = torch.from_numpy(class_embeddings)
    print("class embedding shape: ",class_embeddings.shape)
    embedding_dim = class_embeddings.shape[-1]

    if init_method == "mean": 
        mean_class_embedding = torch.mean(class_embeddings, dim=0)
        std_class_embedding = torch.std(class_embeddings, dim=0)
        print("mean class embedding : ",mean_class_embedding.shape)
        init_embeddings = torch.normal(mean=mean_class_embedding, std=std_class_embedding) #axis-wise sampling 
        print("init embedding : ",init_embeddings.shape)

    
    elif init_method == "random": #utlize the randomly chosen embedding as initial class embedding
        index_list = random.sample(range(1000), init_num)
        print(f"The {init_num} random classes: {index_list}")
        init_embeddings = class_embeddings[index_list]

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

    elif init_method == "target":

        init_embeddings = (
            class_embeddings[target_class].unsqueeze(0).repeat(init_num, 1)
        )
        init_embeddings += torch.randn((init_num, embedfding_dim)) * noise_std

    return init_embeddings


def run_concept_inversion(
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
    half_z_num,
    target_class,
    intermediate_dir,
    total_class,
    img_size,
    class_dir,
    model_name,
    threshold,
    global_iter,
    itp_policy
):
    optim_embedding = init_embeddings.to(device)
    print("optim embedding : ",optim_embedding.shape)
    optim_embedding.requires_grad_()
    T = torch.tensor(1.0) #class-wise adaptive temperature scaling
    T.requires_grad_(True)

    total_loss = []
    total_T = []
    total_prob =[]
    logit_magnitude = []
    best_acc=0.0
    best_err=100.0
    threshold = threshold
    iter_flag =0
    # epoch =0
    for epoch in range(n_iters):
    # while (iter_flag != 1):
        change_seed()
        # epoch += 1
        print(target_class,'-th class image synthesis stage start!')
        labels = torch.LongTensor([target_class] * z_num).to(device)
        optim_comps = { # we update the class embedding and temeperature parameter by optimizing the objective function
        "class_embedding":optim_embedding,
        "T": T
        }
        optim_params = [optim_embedding]

        optimizer = optim.Adam(optim_params+[T], lr=lr, weight_decay=dr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        for z_step in range(steps_per_z):
            optimizer.zero_grad()
            final_T = torch.clamp(T,min=0.001) # For preventing the temperature from converging negative value, we clip the temperature with 0.001 
            
            zs = torch.randn((z_num, dim_z), requires_grad=False).to(device) #noise
            
            clamp_embedding = torch.clamp(optim_embedding,min_clamp, max_clamp)
            embedding_norm = clamp_embedding.norm(2, dim=0, keepdim=True)
            repeat_clamped_embedding = clamp_embedding.repeat(z_num, 1).to(device)
            gan_images_tensor = G(zs, repeat_clamped_embedding) # image synthesis via BigGAN generato
            flip = random.random() > 0.5
            if flip:
                gan_images_tensor = torch.flip(gan_images_tensor, dims = (3,))
            
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size,
                mode=itp_policy
            )
            small_noise = torch.randn_like(resized_images_tensor) * 0.005
            resized_images_tensor.add_(small_noise).clamp_(min=-1.0, max=1.0)

            # output logit according to the different architecture
            if 'vit_cifar100' in model_name:
                model = model.to(device)
                pred_logits = model(resized_images_tensor)

            elif 'vit' in model_name or 'beit' in model_name or 'deit' in model_name:
                model = model.to(device)
                outputs = model(resized_images_tensor)
                pred_logits = outputs.logits

            elif 'cct' in model_name or 'vgg' in model_name:
                pred_logits = net(resized_images_tensor)

            elif 'resnet34' in model_name:
                pred_logits,f1,f2,f3,f4,f5 = net(resized_images_tensor)
            else:
                pred_logits = net(resized_images_tensor)

            norm = torch.norm(pred_logits,p=2,dim=-1,keepdim=True) + 1e-7 #logit normalization
            logit_norm = torch.div(pred_logits,norm)/final_T.cuda() # Apply the Adaptive temeperature scaling with normalized logits
           
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()

            loss = criterion(logit_norm, labels)
            # loss = criterion(pred_logits,labels)

            dloss_function ='image'
            alpha = 0.1
            diversity_loss = get_diversity_loss(
                    half_z_num,
                    zs,
                    dloss_function,
                    pred_probs,
                    None,
                    #alexnet_conv5,
                    resized_images_tensor,
                )
            #rint(diversity_loss)
            #loss += -alpha * diversity_loss.mean()

            diff1 = resized_images_tensor[:,:,:,:-1] - resized_images_tensor[:,:,:,1:]
            diff2 = resized_images_tensor[:,:,:-1,:] - resized_images_tensor[:,:,1:,:]
            diff3 = resized_images_tensor[:,:,1:,:-1] - resized_images_tensor[:,:,:-1,1:]
            diff4 = resized_images_tensor[:,:,:-1,:-1] - resized_images_tensor[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss +=  1.4e-3*loss_var #total variance regualizer
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
            log_line += f"Average Target Probability:{avg_target_prob:.4f}\t"
            log_line += f"Temperature Probability: {T.item():.3f}\t"
            log_line += f"Total loss: {loss.item():.4f}\t"
            log_line += f"logit mag: {torch.sum(torch.norm(logit_norm,p=2)).item()/z_num:.4f}\t"

            print(log_line)
            if z_step % 50:
                print(f'lr = {optimizer.param_groups[0]["lr"]}')
            total_loss.append(loss.item())
            total_T.append(T.item())
            total_prob.append(avg_target_prob)
            logit_magnitude.append(torch.sum(norm).item()/z_num)
    
            if intermediate_dir: # store the synthesized images on intermedaite process
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
            if best_acc >= threshold: # If the target probablity is over than threshold, stop the image generation 
                iter_flag = 1
                break
            # else:
            #     change_seed()
                

        np.save(f"{class_dir}/{target_class}_embedding_{global_iter}.npy", clamp_embedding.detach().cpu().numpy())

    file_path = f"{intermediate_dir}/"
    np.savetxt(os.path.join(file_path,'CE_loss.csv'), total_loss, fmt='%.4f')
    np.savetxt(os.path.join(file_path,'temperature.csv'), total_T, fmt='%.3f')
    np.savetxt(os.path.join(file_path,f'gt_probability_{global_iter}.csv'), total_prob, fmt='%.3f')

    return optim_comps,best_acc


def save_final_samples( #save completed images in local directory
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


    optim_imgs = []
    original_imgs = []

    torch.set_rng_state(state_z)

    for show_id in range(num_final):
        z_hats = torch.randn((num_final, dim_z), device=device, requires_grad=False)

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

    original_image_path = f"{class_dir}/{target_class}_original.jpg"
    original_imgs = torch.cat(original_imgs, dim=0)
    save_image(original_imgs, original_image_path, normalize=True, nrow=4)


def denormalize(image_tensor, dataset):
    channel_num = 0
    if dataset == 'cifar':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        channel_num = 3

    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        channel_num = 3

    for c in range(channel_num):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c]*s+m, 0, 1)

    return image_tensor


def save_instance_image(optim_comps,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,img_len,total_class,global_epoch):
    #save images by same class
    final_dir = 'Fake_'+final_dir+'/'+str(target_class)+'/'
    img_len_per_class = img_len/total_class
    num_generations = int(img_len_per_class/(batch_size*5)) #flower 250 cifar100 500 
    print("num generation : ",num_generations)
    for i in range(num_generations):
        zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
        with torch.no_grad():
            repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
            print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
            gan_images_tensor = G(zs, repeat_class_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size
                )
        targets = torch.LongTensor([target_class] * batch_size)
        images = resized_images_tensor.data.clone()

        if 'cifar' in dataset:
            images = nn.functional.interpolate(
                images, size=32
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

def infinite_save(flag,avg_prob,optim_comps,net,model,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,model_name,img_len,total_class,global_epoch,itp_policy):
    criterion = nn.CrossEntropyLoss()
    if target_class < 10:
        final_dir = 'Fake_'+final_dir+'/00'+str(target_class)+'/'
        final_dir_denorm = 'Fake_32size'+final_dir+'/'
    else:
        final_dir = 'Fake_'+final_dir+'/0'+str(target_class)+'/' 
        final_dir_denorm = 'Fake_32size'+final_dir+'/'
    if flag ==1 : #지환님, threshold를 못넘는 이미지는 삭제하고 다시 만듭니다! 
        # import glob
        # import os

        # 파일 이름이 '0'으로 시작하는 모든 파일 가져오기
        files_to_delete = glob.glob(os.path.join(final_dir, str(global_epoch)+'*'))
        # 파일 삭제
        for file_path in files_to_delete:
            os.remove(file_path)

    img_len_per_class = img_len/total_class
    image_count = 0
    batch_size = 1
    num_generations = int(img_len_per_class/(batch_size*10)) #지환님, 여기에 bs에 10을 곱한 이유는 multiseed 10이기 때문이에요! 5seed를 원하시면 5로 바꿔주세요
    
    #print("num generation : ",num_generations)
    #batch_size = 1
    save_criterion = 0.0
    if avg_prob < 0.5:
        save_criterion = 0.0
    else:
        save_criterion = 0.9
    best_err = 0.001
    repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
    print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
    image_count = 0
    while(image_count<num_generations):
        zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
        with torch.no_grad():
            gan_images_tensor = G(zs, repeat_class_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size,
                mode=itp_policy
                )
        targets = torch.LongTensor([target_class] * batch_size)
        image = resized_images_tensor.data.clone()
        
        if 'vit_cifar100'==model_name:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            model.eval()
            output = model(image)

        elif 'vit_cifar' in model_name or 'deit' in model_name or 'vit-L-CIFAR100' == model_name or 'deit' in model_name or 'vit' in model_name:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            model.eval()
            outputs = model(image)
            output = outputs.logits

        
            #output = outputs.logits

        elif 'vgg' in model_name:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            net.eval()
            output = net(image)
            #output = outputs.logits


        elif 'resnet34' in model_name:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            net.eval()
            output,_,_,_,_,_ = net(image)
        else:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            net.eval()
            output = net(image)


        _,pred = output.max(1)
        gt_prob =  nn.functional.softmax(output, dim=1)
        ce_loss = criterion(output,torch.LongTensor([target_class]).cuda())

        image_32 = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
                )

        if gt_prob[:,target_class].item()> save_criterion:
            print(image_count,"_image saved!!")
            image_count +=1
            image_denorm = denormalize(image,dataset)
            image_np = image.data.cpu().numpy()
            pil_images = torch.from_numpy(image_np)

            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            if not os.path.exists(final_dir_denorm):
                os.makedirs(final_dir_denorm)

            vutils.save_image(image_32,os.path.join(final_dir,'{}epoch_{}_gt_prob_output_{}'.format(global_epoch,gt_prob[:,target_class].item(),image_count))+'.png',normalize=True,scale_each=True,nrow=1)
            #vutils.save_image(image_32,os.path.join(final_dir_denorm,'{}epoch_{}_gt_prob_output_{}'.format(global_epoch+100,gt_prob[:,target_class].item(),image_count))+'.png',normalize=True,scale_each=True,nrow=1)
    #return final_dir

def save_zo2m_image(optim_comps,avg_prob,net,model,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,model_name,img_len,total_class,global_epoch,itp_policy):
    #save images by same class
    flag = 0
    infinite_save(flag,avg_prob,optim_comps,net,model,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,model_name,img_len,total_class,global_epoch,itp_policy)
    # train_acc= test(net,model,model_name,'Fake_'+final_dir,itp_policy)
    # print("train_acc :",train_acc) #지환님, 한클래스를 만들고 test를 합니다. 
    # if avg_prob <= 0.5:
    #     return
    # else:
    #     while(train_acc <= 0.85): #지환님, threshold를 넘지 못하면, 이 while문을 무한으로 반복하게 됩니다.
    #         flag = 1
    #         infinite_save(flag,avg_prob,optim_comps,net,model,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,model_name,img_len,total_class,global_epoch,itp_policy)
    #         train_acc= test(net,model,model_name,'Fake_'+final_dir,itp_policy)

def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Inversion')
    parser.add_argument('--itp_policy', type=str, default='nearest',
                        help='interpolation_policy') 
    args = parser.parse_args()

    # Set value for interpolation policy
    itp_policy = args.itp_policy

    opts = yaml.safe_load(open("opts_ci_2.yaml"))
    seed_z = opts["seed_z"]
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

    init_method = opts["init_method"]
    print(f"Initialization method: {init_method}")
    
    # Load the models.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    print("Loading the BigGAN generator model...")
    resolution = opts["resolution"]
    config = get_config(resolution)
    start_time = time.time()
    G = BigGAN.Generator(**config)
    # we can choose the generator option 
    if resolution == 128:
        biggan_weights = "./generator_pretrained_weights/138k/G_ema.pth"
    else:
        biggan_weights = "./generator_pretrained_weights/biggan_256_weights.pth"

    G.load_state_dict(torch.load(f"{biggan_weights}"), strict=False)
    G = nn.DataParallel(G).to(device)
    G.eval()

    model_name = opts["model"]
    if 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
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
    half_z_num = z_num // 2
    print(f"BigGAN initialization time: {time.time() - start_time}")

    # Set up optimization.
    init_num = opts["init_num"]
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    target_class = opts["target_class"]
    batch_size = opts["generation_batch_size"]

    final_dir = opts["final_dir"] + args.itp_policy # store all class images for evaluating the IS and FID scores
    if final_dir:
        print(f"Saving final samples in {final_dir}.")
        os.makedirs(final_dir, exist_ok=True)

    original_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
    original_embeddings = torch.from_numpy(original_embeddings)
    
    criterion = nn.CrossEntropyLoss()
    labels = torch.LongTensor([target_class] * z_num).to(device)
    state_z = torch.get_rng_state()
    
    # for target_class in range(int(opts["total_class"])):
    # for target_class in range(17,100):
    # for target_class in range(75,81):
    for target_class in [1,7]:
        # 1seed
        for epoch in range(10): # if you want to generate multi seed images per images, change this number, 지환님, multiseed를 몇개 만들건지 괄호 안에 써주세요! 10seed면 10을 써줌
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
            optim_comps,avg_prob = run_concept_inversion(
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
                half_z_num,
                target_class,
                intermediate_dir,
                opts["total_class"],
                opts["img_size"],
                class_dir,
                model_name,
                opts["threshold"],
                epoch,
                itp_policy
            )

            #if class_dir:
            if 0:
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
            #if 1:
                #save_instance_image(optim_comps,
                save_zo2m_image(optim_comps,
                avg_prob,
                net,
                model,
                G,
                opts["img_size"],
                batch_size,
                target_class,
                z_num,
                dim_z,
                class_dir,
                final_dir,
                opts["dataset"],
                opts["model"],
                opts["img_len"],
                opts["total_class"],
                epoch,
                itp_policy
            )
            
if __name__ == "__main__":
    main()
