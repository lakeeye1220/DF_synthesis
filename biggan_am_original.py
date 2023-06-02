# 2023/06/02 시점 fix 된 code 그대로입니다. 출력 빈도 등만 수정했습니다.

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
from tqdm import tqdm

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
        #init_embeddings = mean_class_embedding.repeat(init_num, 1)
        #init_embeddings += torch.randn((init_num, embedding_dim)) * 0.1

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
        init_embeddings += torch.randn((init_num, embedding_dim)) * noise_std

    return init_embeddings


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
    model_name
):
    
    embedding_layer = nn.Embedding(total_class,128) # num_embeddings, embedding_dim
    optim_embedding = embedding_layer(torch.LongTensor([target_class])).detach()
    mse_criterion = nn.MSELoss()
    #mean_optimal_embedding = init_embeddings.detach()
    #optim_embedding = init_embeddings.to(device)
    print("optim embedding : ",optim_embedding.shape)
   
    optim_comps = {
        "optim_embedding": optim_embedding,
        "use_noise_layer": use_noise_layer,
    }
    optim_embedding.requires_grad_()
    optim_params = [optim_embedding]
  
    if use_noise_layer:
        noise_layer = nn.Linear(dim_z, dim_z).to(device)
        print("noise layer shape :",dim_z)
        noise_layer.train()
        optim_params += [params for params in noise_layer.parameters()]
        optim_comps["noise_layer"] = noise_layer

    #T = torch.empty((1,))
    #torch.nn.init.normal(T,3.0,1)
    T = torch.tensor(1.0)
    T.requires_grad_(True)

    optimizer = optim.Adam(optim_params, lr=lr, weight_decay=dr)
    torch.set_rng_state(state_z)
    total_loss = []
    total_T = []
    total_prob =[]
    logit_magnitude = []

    #warm up stage - Total variance loss
    for epoch in range(n_iters):
        for z_step in range(50):
            zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)
            optimizer.zero_grad()
            if use_noise_layer:
                z_hats = noise_layer(zs)
            else:
                z_hats = zs
          
            clamped_embedding = torch.clamp(optim_embedding ,min_clamp, max_clamp)
            repeat_clamped_embedding = clamped_embedding.repeat(z_num, 1).to(device)
            inputs_jit = G(z_hats, repeat_clamped_embedding)
           
            diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
            diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
            diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
            diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = 6.0e-3*loss_var

            loss.backward()
            optimizer.step()
            

            if z_step % 15 == 0:
                log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
                log_line += f"TV loss: {loss.item():.4f}\t"
          
                print(log_line)

            total_loss.append(loss.item())
    
            if intermediate_dir:
                if z_step %50 ==0:
                    global_step_id = epoch * steps_per_z + z_step
                    img_f = f"{global_step_id:0=7d}_initial.jpg"
                    output_image_path = f"{intermediate_dir}/{img_f}"
                    save_image(
                    inputs_jit, output_image_path, normalize=True, nrow=10
                )

        #save class embedding trained by TV loss
        np.save(f"{class_dir}/{target_class}_TV_embedding.npy", clamped_embedding.detach().cpu().numpy())
        class_embedding = np.load(f"{class_dir}/{target_class}_TV_embedding.npy")
        class_embedding = torch.from_numpy(class_embedding)    
        class_embedding.requires_grad_()

        # call the class embedding optimized by total variation, and predict the pseudo label
        '''
        clamped_embedding = torch.clamp(class_embedding, min_clamp, max_clamp)
        zs = torch.randn((1, dim_z), requires_grad=False).to(device)
        if use_noise_layer:
            z_hats = noise_layer(zs)
        else:
            z_hats = zs
        gan_images_tensor = G(z_hats, clamped_embedding)
        resized_images_tensor = nn.functional.interpolate(
                    gan_images_tensor, size=img_size #Flower 224, CelebA 128
                )
        '''
        print(target_class,'-th class image synthesis stage start!')
        labels = torch.LongTensor([target_class] * z_num).to(device)

        optim_comps2 = {
        "class_embedding": class_embedding,
        "use_noise_layer": use_noise_layer,
        "T": T
        }
        optim_params2 = [class_embedding]
        #optim_params2 = [optim_embedding]
        if use_noise_layer:
            noise_layer = nn.Linear(dim_z, dim_z).to(device)
            noise_layer.train()
            optim_params2 += [params for params in noise_layer.parameters()]
       
        optimizer2 = optim.Adam(optim_params2+[T], lr=lr, weight_decay=dr)

        for z_step in tqdm(range(steps_per_z), desc="main train loop"):
            optimizer2.zero_grad()
            final_T = torch.clamp(T,min=0.001)
            
            zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)
            if use_noise_layer:
                noise_layer.train()
                z_hats = noise_layer(zs)
            else:
                z_hats = zs

            clamp_embedding = torch.clamp(class_embedding ,min_clamp, max_clamp)
            #clamp_embedding = torch.clamp(optim_embedding ,min_clamp, max_clamp)
            embedding_norm = clamp_embedding.norm(2, dim=0, keepdim=True)
            repeat_clamped_embedding = clamp_embedding.repeat(z_num, 1).to(device)
            gan_images_tensor = G(z_hats, repeat_clamped_embedding)
            flip = random.random() > 0.5
            if flip:
                gan_images_tensor = torch.flip(gan_images_tensor, dims = (3,))
            
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size # ViT CIFAR10 224 Flower 224, CelebA 128
            )
            if 'vit' in model_name:
                model = model.to(device)
                outputs = model(resized_images_tensor)
                pred_logits = outputs.logits
            else:
                pred_logits,_,_,_,_,_ = net(resized_images_tensor)
                #pred_logits = net(resized_images_tensor)

            norm = torch.norm(pred_logits,p=2,dim=-1,keepdim=True) + 1e-7
            logit_norm = torch.div(pred_logits,norm)/final_T.cuda()
           
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()

            loss = criterion(logit_norm, labels)

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
            optimizer2.step()

            if z_step %50 == 0:
                log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
                log_line += f"Average Target Probability:{avg_target_prob:.4f}\t"
                log_line += f"Temperature Probability: {T.item():.3f}\t"
                log_line += f"CE loss: {loss.item():.4f}\t"
                log_line += f"logit mag: {torch.sum(torch.norm(logit_norm,p=2)).item()/z_num:.4f}\t"
                log_line += f"embedding mag: {torch.sum(embedding_norm).item():.4f}"

                print(log_line)
            
            total_loss.append(loss.item())
            total_T.append(T.item())
            total_prob.append(avg_target_prob)
            logit_magnitude.append(torch.sum(norm).item()/z_num)
    
            if intermediate_dir:
                if z_step %50 ==0:
                    global_step_id = epoch * steps_per_z + z_step
                    img_f = f"{global_step_id:0=7d}.jpg"
                    output_image_path = f"{intermediate_dir}/{img_f}"
                    save_image(
                    gan_images_tensor, output_image_path, normalize=True, nrow=10
                )
            torch.cuda.empty_cache()

        np.save(f"{class_dir}/{target_class}_embedding.npy", clamped_embedding.detach().cpu().numpy())
        #class_embedding = np.load(f"{class_dir}/{target_class}_embedding.npy")
        #class_embedding = torch.from_numpy(class_embedding)    

    file_path = f"{intermediate_dir}/"
    np.savetxt(os.path.join(file_path,'CE_loss.csv'), total_loss, fmt='%.4f')
    np.savetxt(os.path.join(file_path,'temperature.csv'), total_T, fmt='%.3f')
    np.savetxt(os.path.join(file_path,'gt_probability.csv'), total_prob, fmt='%.3f')

    return optim_comps2


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

import time
timestr = time.strftime("%Y%m%d_%H%M%S_")
def save_instance_image(optim_comps,G,img_size,batch_size,target_class,z_num,dim_z,class_dir,final_dir,dataset):
    final_dir = 'Fake_'+timestr+final_dir+'/'+str(target_class)+'/'
    num_generations = int(5000/batch_size)+1
    for i in range(num_generations):
        zs = torch.randn((batch_size,dim_z),requires_grad=False).cuda()
        if optim_comps["use_noise_layer"]:
            noise_layer = nn.Linear(dim_z,dim_z)
            noise_layer.load_state_dict(torch.load(f"{class_dir}/{target_class}_noise_layer.pth"))

            noise_layer = noise_layer.cuda()
            zs = noise_layer(zs)

        with torch.no_grad():
            repeat_class_embedding = optim_comps["class_embedding"].repeat(batch_size,1).cuda() 
            gan_images_tensor = G(zs, repeat_class_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=img_size #Flower 224, CelebA 128
                )
        targets = torch.LongTensor([target_class] * batch_size)
        images = resized_images_tensor.data.clone()

        if images.shape[2] > 32:
            images = nn.functional.interpolate(
                images, size=32 #Flower 224, CelebA 128 
            )
        for id in range(images.shape[0]):
            class_id = str(targets[id].item()).zfill(2)
            image = images[id].reshape(3,32,32) #cifar라서 32 32 이다 여기도 파라미터로 바꿔야할듯
            image = denormalize(image,dataset)
            image_np = images[id].data.cpu().numpy()
            pil_images = torch.from_numpy(image_np)

            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            vutils.save_image(image,os.path.join(final_dir,'{}_output_{}'.format(i,id))+'.png',normalize=True,scale_each=True,nrow=1)

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
    # Set up cudnn.benchmark for free speed.
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
    if 'vit' in model_name:
        feature_extractor,model = load_net(model_name)
        net = None
        #feature_extractor = nn.DataParallel(fe)
        #model = nn.DataParallel(m)
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
    final_dir = opts["final_dir"] # store all class images for evaluating the IS and FID scores
    if final_dir:
        print(f"Saving final samples in {final_dir}.")
        os.makedirs(final_dir, exist_ok=True)

    original_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
    original_embeddings = torch.from_numpy(original_embeddings)
    
    criterion = nn.CrossEntropyLoss()
    labels = torch.LongTensor([target_class] * z_num).to(device)
    state_z = torch.get_rng_state()

    for target_class in tqdm(range(int(opts["total_class"])), desc="saving class-wise imgs"):
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
            model_name
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
            opts["dataset"]
        )
        

if __name__ == "__main__":
    main()