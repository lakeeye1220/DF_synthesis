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
import argparse

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
            class_embeddings[817].unsqueeze(0).repeat(init_num, 1)
        )
        init_embeddings += torch.randn((init_num, embedding_dim)) * 0.1

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
    global_epoch,
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
    best_z_iter = 0
    best_threshold = 0.0

    for epoch in range(n_iters):
        print(target_class,'-th class image synthesis stage start!')
        labels = torch.LongTensor([target_class] * z_num).to(device)
        optim_comps = { # we update the class embedding and temeperature parameter by optimizing the objective function
        "class_embedding":optim_embedding,
        "T": T,
        "best_threshold":best_threshold
        }
        
        optim_params = [optim_embedding]

        optimizer = optim.Adam(optim_params+[T], lr=lr, weight_decay=dr)

        for z_step in range(steps_per_z):
            optimizer.zero_grad()
            final_T = torch.clamp(T,min=0.001) # For preventing the temperature from converging negative value, we clip the temperature with 0.001 
            
            z_hats = torch.randn((z_num, dim_z), requires_grad=False).to(device) #noise
            
            clamp_embedding = torch.clamp(optim_embedding,min_clamp, max_clamp)
            embedding_norm = clamp_embedding.norm(2, dim=0, keepdim=True)
            repeat_clamped_embedding = clamp_embedding.repeat(z_num, 1).to(device)
            gan_images_tensor = G(z_hats, repeat_clamped_embedding) # image synthesis via BigGAN generato

            # breakpoint()
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
            elif 'vit-mae-cub' in model_name:
                model = model.to(device)
                # encoding = image_processor()
                outputs = model(resized_images_tensor)
                pred_logits = outputs.logits

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
            # loss = criterion(pred_logits, labels)

            diff1 = resized_images_tensor[:,:,:,:-1] - resized_images_tensor[:,:,:,1:]
            diff2 = resized_images_tensor[:,:,:-1,:] - resized_images_tensor[:,:,1:,:]
            diff3 = resized_images_tensor[:,:,1:,:-1] - resized_images_tensor[:,:,:-1,1:]
            diff4 = resized_images_tensor[:,:,:-1,:-1] - resized_images_tensor[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss +=  1.3e-3*loss_var #total variance regualizer
            # loss +=  1.3e-4*loss_var #total variance regualizer

            loss.backward()
            optimizer.step()

            log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
            log_line += f"Average Target Probability:{avg_target_prob:.4f}\t"
            log_line += f"Temperature Probability: {T.item():.3f}\t"
            log_line += f"Total loss: {loss.item():.4f}\t"
            log_line += f"logit mag: {torch.sum(torch.norm(logit_norm,p=2)).item()/z_num:.4f}\t"

            print(log_line)
            total_loss.append(loss.item())
            total_T.append(T.item())
            total_prob.append(avg_target_prob)
            logit_magnitude.append(torch.sum(norm).item()/z_num)
    
            if intermediate_dir: # store the synthesized images on intermedaite process
                if avg_target_prob > best_acc:
                    # store intermediate information
                    best_threshold = avg_target_prob
                    best_z_iter = z_step
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
                    np.save(f"{class_dir}/{target_class}_embedding_{best_z_iter}_{global_epoch}.npy", clamp_embedding.detach().cpu().numpy())
            torch.cuda.empty_cache()
            if best_acc >= threshold: # If the target probablity is over than threshold, stop the image generation 
                break
            elif z_step == steps_per_z-1:
                print("Optimized Class Embedding Changed as best classs embedding!")
                optim_comps["class_embedding"] = torch.from_numpy(np.load(f"{class_dir}/{target_class}_embedding_{best_z_iter}_{global_epoch}.npy")).to(device)

        np.save(f"{class_dir}/{target_class}_embedding.npy", clamp_embedding.detach().cpu().numpy())

    file_path = f"{intermediate_dir}/"
    np.savetxt(os.path.join(file_path,'CE_loss.csv'), total_loss, fmt='%.4f')
    np.savetxt(os.path.join(file_path,'temperature.csv'), total_T, fmt='%.3f')
    np.savetxt(os.path.join(file_path,'gt_probability.csv'), total_prob, fmt='%.3f')

    return optim_comps


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

def save_zo2m_image(optim_comps,net,model,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset,img_len,total_class,global_epoch,threshold,model_name,itp_policy):
    #save images by same class
    final_dir_fake = 'Fake_'+final_dir+'/'+str(target_class)+'/'
    final_dir_denorm = 'Fake_denorm'+final_dir+'/'+str(target_class)+'/'
    img_len_per_class = img_len/total_class
    image_count = 0
    num_generations = int(img_len_per_class/(batch_size*5)) #flower 250 cifar100 500 
    #print("num generation : ",num_generations)
    batch_size = 1
    save_criterion = 0.98
    if optim_comps["best_threshold"] <= threshold:
        # save_criterion
        save_criterion = optim_comps["best_threshold"] * 1.1
    repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
    print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
    if 'resnet' in model_name or 'mlp_mixer' in model_name:
        net.eval()
    else:
        model.eval()
    fail_zs = []
    success_zs = []
    criterion = nn.CrossEntropyLoss()
    while(image_count<img_len_per_class):
        zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
        with torch.no_grad():
            gan_images_tensor = G(zs, repeat_class_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size,
                mode=itp_policy
                )
        targets = torch.LongTensor([target_class] * batch_size)
        image = resized_images_tensor.data.clone()

        if 'resnet34' in model_name:
            image = nn.functional.interpolate(
                image, size=32,
                mode=itp_policy
            )
            output,_,_,_,_,_ = net(image)
        elif 'vit_cifar100' in model_name:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            # encoding = image_processor()
            output = model(image)
        elif 'vit' in model_name or 'deit' in model_name:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            # encoding = image_processor()
            output = model(image)
            output = output.logits
        else:
            image = nn.functional.interpolate(
                image, size=img_size,
                mode=itp_policy
            )
            output = net(image)
            # output = output.logits
        gt_prob =  nn.functional.softmax(output, dim=1)

        if gt_prob[:,target_class].item()> save_criterion:
            # print(image_count,"_image saved!!")
            image_count +=1
            if 'cifar' in model_name:
                image = nn.functional.interpolate(
                    image, size=32,
                mode=itp_policy
                )
            image_denorm = denormalize(image,dataset)
            image_np = image.data.cpu().numpy()
            pil_images = torch.from_numpy(image_np)

            if not os.path.exists(final_dir_fake):
                os.makedirs(final_dir_fake)
            if not os.path.exists(final_dir_denorm):
                os.makedirs(final_dir_denorm)
            vutils.save_image(image,os.path.join(final_dir_fake,'{}_gt_prob_output_{}_{}'.format(gt_prob[:,target_class].item(),image_count,global_epoch))+'.png',normalize=True,scale_each=True,nrow=1)
            vutils.save_image(image_denorm,os.path.join(final_dir_denorm,'{}_gt_prob_output_{}_{}'.format(gt_prob[:,target_class].item(),image_count,global_epoch))+'.png',normalize=True,scale_each=True,nrow=1)
            numpy_zs = zs.cpu().numpy()
            success_zs.append(numpy_zs)
        else:
            numpy_zs = zs.cpu().numpy()
            fail_zs.append(numpy_zs)
    print(f"{target_class}th class {image_count} images saved")
    print(f"sucess zs: {len(success_zs)}")
    print(f"fail zs: {len(fail_zs)}")

    np.save(f'{class_dir}/fail_zs_{global_epoch}.npy',fail_zs)
    np.save(f'{class_dir}/success_zs_{global_epoch}.npy',success_zs)

    # fail_probs = []
    # for i in range(len(fail_zs)):

    #     zs = torch.from_numpy(fail_zs[i]).cuda()
    #     with torch.no_grad():
    #         # batch_size = 1
    #         repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
            
    #         # print("repeat class_embedding : ", repeat_class_embedding.shape," batch size : ",batch_size)
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
    #          images = nn.functional.interpolate(
    #             images, size=img_size
    #         )
        
    #     # load pretrained classifier
    #     resized_images_tensor_check = nn.functional.interpolate(
    #             gan_images_tensor, size=img_size # ViT CIFAR10 224 Flower 224, CelebA 128
    #         )
    #     small_noise = torch.randn_like(resized_images_tensor_check) * 0.005
    #     resized_images_tensor_check.add_(small_noise).clamp_(min=-1.0, max=1.0)

    #     correct = 0
    #     total = 0 
    #     pred_probs = 0.0
    #     pred_logits = None
    #     with torch.no_grad():

    #         if 'vit_inat' in model_name:
    #             pred_logits = net(resized_images_tensor_check)
    #         elif 'vit_cifar100' in model_name:
    #             model = model.to(device)
    #             pred_logits = model(resized_images_tensor_check)

    #         elif 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
    #             model = model.to(device)
    #             outputs = model(resized_images_tensor_check)
    #             pred_logits = outputs.logits

    #         elif 'vgg16' in model_name:
    #             # model = model.to(device)
    #             # outputs = model(resized_images_tensor)
    #             # pred_logits = outputs.logits
                
    #             pred_logits = net(resized_images_tensor_check)

    #         elif 'cct' in model_name or 'vgg' in model_name:
    #             pred_logits = net(resized_images_tensor_check)
    #         elif 'resnet34' in model_name:
    #             pred_logits,f1,f2,f3,f4,f5 = net(resized_images_tensor_check)
    #         else:
    #             pred_logits = net(resized_images_tensor_check)

    #         # _, predicted = torch.max(pred_logits.data, 1)

    #         pred_probs = nn.functional.softmax(pred_logits, dim=1)
    #         labels = targets.cuda()
    #         pred_ce = criterion(pred_logits, labels).item()
    #         # breakpoint()
    #         target_prob = pred_probs[:, target_class].item()
    #         # total += targets.size(0)
    #         # correct += (predicted == targets.cuda()).sum().item()

    #         pred_probs = target_prob

    #         fail_probs.append(pred_probs)
        
    # plt.hist(fail_probs, bins=20, range=(0,1), edgecolor= "black", alpha=0.7)
    # plt.title(f"Histogram of Class {target_class} fail zs ")
    # plt.xlabel('Probability')
    # plt.ylabel('Count')

    # fail_probs.sort(reverse=True)

    # plt.savefig(f'{class_dir}/{target_class}_fail_zs_histogram_{global_epoch}_max_{fail_probs[0]}.png')

    # plt.clf()




    # print(f"{target_class}th class {img_len_per_class} image generation done..")
    # print(f"# of success zs:{len(success_zs)}")
    # print(f"# of fail zs:{len(fail_zs)}")
    # np.save(f'{class_dir}/success_zs_{seed_z}.npy',success_zs)
    # np.save(f'{class_dir}/fail_zs_{seed_z}.npy',fail_zs)

def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Inversion')
    parser.add_argument('--itp_policy', type=str, default='nearest',
                        help='interpolation_policy')

    args = parser.parse_args()

    # Set value for interpolation policy
    itp_policy = args.itp_policy



    opts = yaml.safe_load(open("opts_ci.yaml"))
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
    if 'vit-mae-cub' in model_name:
        image_processor, model = load_net(model_name)
        net = None
        feature_extractor = None

    elif 'vit' in model_name or 'swin' in model_name or 'deit' in model_name:
        feature_extractor,model = load_net(model_name)
        net = None
        image_processor = None

    else:
        feature_extractor=None
        model=None
        image_processor = None
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

    # final_dir = opts["final_dir"] # store all class images for evaluating the IS and FID scores
    final_dir = opts["final_dir"] + args.itp_policy
    if final_dir:
        print(f"Saving final samples in {final_dir}.")
        os.makedirs(final_dir, exist_ok=True)

    original_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
    original_embeddings = torch.from_numpy(original_embeddings)
    
    criterion = nn.CrossEntropyLoss()
    labels = torch.LongTensor([target_class] * z_num).to(device)
    state_z = torch.get_rng_state()

    for target_class in range(int(opts["total_class"])):
    # for target_class in range(155,200):
    # for target_class in [8,9]:
        for epoch in range(5): # if you want to generate multi seed images per images, change this number
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
            optim_comps = run_concept_inversion(
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
            # if 1:
            #     save_final_samples(
            #         optim_comps,
            #         min_clamp,
            #         max_clamp,
            #         device,
            #         state_z,
            #         num_final,
            #         dim_z,
            #         G,
            #         repeat_original_embedding,
            #         class_dir,
            #         target_class
            #     )

            if final_dir:
                #save_instance_image(optim_comps,
                save_zo2m_image(optim_comps,
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
                opts["img_len"],
                opts["total_class"],
                epoch,
                opts["threshold"],
                model_name,
                itp_policy
            )
            
if __name__ == "__main__":
    main()
