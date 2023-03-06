import BigGAN
import itertools
import numpy as np
import random
import time
import torch.nn.functional as F
import yaml

from torch import optim
from torchvision.utils import save_image
from utils import *
from resnet import ResNet34
from WGAN import LSUNGenerator
from collections import OrderedDict
from model_resnet import Generator


class NaturalInversionFeatureHook():
    def __init__(self, module, rs):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.rs = rs

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type())  - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_initial_embeddings(
    resolution,
    init_method,
    init_num,
    min_clamp,
    max_clamp,
    dim_z,
    G,
    net,
    target_class,
    noise_std,
):
    #class_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
    #class_embeddings = torch.from_numpy(class_embeddings)
    #print("class embedding shape :",class_embeddings.shape)
    class_embeddings = torch.randn((1000,128), requires_grad=False)
    embedding_dim = class_embeddings.shape[-1]
    #print("embedding dim : ",embedding_dim.shape)

    if init_method == "mean":
        mean_class_embedding = torch.mean(class_embeddings, dim=0)
        init_embeddings = mean_class_embedding.repeat(init_num, 1)
        init_embeddings += torch.randn((init_num, embedding_dim)) * 0.1

    elif init_method == "top":

        class_embeddings_clamped = torch.clamp(class_embeddings, min_clamp, max_clamp)

        num_samples = 10
        avg_list = []
        for i in range(1000):
            class_embedding = class_embeddings_clamped[i]
            repeat_class_embedding = class_embedding.repeat(num_samples, 1)
            final_z = torch.randn((num_samples, dim_z), requires_grad=False)

            with torch.no_grad():
                #print("geneartor : ",G)
                gan_images_tensor = G(final_z, repeat_class_embedding)
                resized_images_tensor = nn.functional.interpolate(
                    gan_images_tensor, size=32
                )
                pred_logits,_,_,_,_,_ = net(resized_images_tensor)

            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()

            #labels = torch.LongTensor([0,1,2,3,4,5,6,7,8,9] * 2).to('cpu')
            #avg_target_prob = pred_probs[:, labels].mean().item()

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
    half_z_num, zs, dloss_function, pred_probs, alexnet_conv5, resized_images_tensor
):
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
        #alexnet = 그냥 net이다
        _,_,_,_,_,features_out = alexnet_conv5(resized_images_tensor)
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
    init_embedding,
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
    net,
    criterion,
    labels,
    dloss_function,
    half_z_num,
    alexnet_conv5,
    alpha,
    target_class,
    init_embedding_idx,
    intermediate_dir,
    use_noise_layer,
):
    labels_target = torch.LongTensor([0])
    optim_embedding =F.one_hot(labels_target,num_classes=10)
    optim_embedding = optim_embedding.type(torch.FloatTensor)
    #optim_embedding = init_embedding.unsqueeze(0).to(device)
    print("optim embedding : ",optim_embedding.shape)
    optim_embedding.requires_grad_()
    print("optim embedding : ",optim_embedding)
    optim_comps = {
        "optim_embedding": optim_embedding,
        "use_noise_layer": use_noise_layer,
    }
    optim_params = [optim_embedding]
    if use_noise_layer:
        noise_layer = nn.Linear(dim_z, dim_z).to(device)
        #noise_layer = nn.Sequential(
        #    nn.Linear(dim_z,dim_z),
        #    nn.Linear(dim_z,dim_z)
        #).to(device)
        #instance_norm = nn.InstanceNorm2d(3)
        #print("noise layer shape :",dim_z)
        #noise_layer.train()
        #optimizer_nl = optim.Adam(noise_layer.parameters(), lr=lr, weight_decay=dr)
    
        noise_layer.train()
        optim_params += [params for params in noise_layer.parameters()]
        #optim_params +=[params for params in instance_norm.parameters()]
        optim_comps["noise_layer"] = noise_layer
    
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(NaturalInversionFeatureHook(module, 0))

    optimizer = optim.Adam(optim_params, lr=lr, weight_decay=dr)

    torch.set_rng_state(state_z)
    for epoch in range(n_iters):
        zs = torch.randn((z_num, dim_z), requires_grad=False).to(device) # 랜덤하게 noise가 들어가는중
        print("zs shape :",zs.shape) #120
        for z_step in range(steps_per_z):
            optimizer.zero_grad()
            optim_embedding.requires_grad=True
            if use_noise_layer:
                z_hats = noise_layer(zs)
            else:
                z_hats = zs

            clamped_embedding = torch.clamp(optim_embedding, min_clamp, max_clamp)
            #clamped_embedding = F.one_hot(labels_target,num_classes=128)
            repeat_clamped_embedding = clamped_embedding.repeat(z_num, 1).to(device)
            #print("repeat clamped_embedding shape :",repeat_clamped_embedding.shape)
            #gan_images_tensor = G(z_hats, repeat_clamped_embedding)
            gan_images_tensor = G(z_hats,repeat_clamped_embedding)
            #WGAN
            #gan_images_tensor = G(z_hats,)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=32
            )
            pred_logits,_,_,_,_,_ = net(resized_images_tensor)
            loss = criterion(pred_logits, labels)
            #print('CE loss : ',loss.item())
            pred_probs = nn.functional.softmax(pred_logits, dim=1)
    
            if dloss_function:
                diversity_loss = get_diversity_loss(
                    half_z_num,
                    zs,
                    dloss_function,
                    pred_probs,
                    #alexnet_conv5,
                    net,
                    resized_images_tensor,
                )
                #print("diversity loss : ",diversity_loss)
                loss += -alpha * diversity_loss

            #optim_embedding.requires_grad=False
            #loss_distr = sum([mod.r_feature.to(device) for idx, mod in enumerate(loss_r_feature_layers)])
            #loss = loss + 0.1*loss_distr # best for noise before BN

            loss.backward()
            optimizer.step()


            avg_target_prob = pred_probs[:, target_class].mean().item()
            #avg_target_prob = pred_probs[:, np.array(labels_target.cpu())].mean().item()
            log_line = f"Embedding: {init_embedding_idx}\t"
            log_line += f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
            log_line += f"Average Target Probability:{avg_target_prob:.4f}"
            print(log_line)

            if intermediate_dir:
                if epoch % 10 ==0:
                    global_step_id = epoch * steps_per_z + z_step
                    img_f = f"{init_embedding_idx}_{global_step_id:0=7d}.jpg"
                    output_image_path = f"{intermediate_dir}/{img_f}"
                    save_image(
                    gan_images_tensor, output_image_path, normalize=True, nrow=10
                )

            torch.cuda.empty_cache()

    return optim_comps


def save_final_samples(
    optim_comps,
    min_clamp,
    max_clamp,
    device,
    model,
    state_z,
    num_final,
    dim_z,
    G,
    repeat_original_embedding,
    final_dir,
    init_embedding_idx,
):
    optim_embedding = optim_comps["optim_embedding"]
    optim_embedding_clamped = torch.clamp(optim_embedding, min_clamp, max_clamp)
    repeat_optim_embedding = optim_embedding_clamped.repeat(4, 1).to(device)

    if optim_comps["use_noise_layer"]:
        optim_comps["noise_layer"].eval()

    optim_imgs = []
    if model not in {"mit_alexnet", "mit_resnet18"}:
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
            #WGAN
            #optim_imgs.append(G(z_hats))
            if model not in {"mit_alexnet", "mit_resnet18"}:
                original_imgs.append(G(z_hats, repeat_original_embedding))
                #original_imgs.append(G(z_hats))

    final_image_path = f"{final_dir}/{init_embedding_idx}.jpg"
    optim_imgs = torch.cat(optim_imgs, dim=0)
    save_image(optim_imgs, final_image_path, normalize=True, nrow=4)
    np.save(
        f"{final_dir}/{init_embedding_idx}.npy",
        optim_embedding_clamped.detach().cpu().numpy(),
    )
    if optim_comps["use_noise_layer"]:
        torch.save(
            optim_comps["noise_layer"].state_dict(),
            f"{final_dir}/{init_embedding_idx}_noise_layer.pth",
        )

    if model not in {"mit_alexnet", "mit_resnet18"}:
        original_image_path = f"{final_dir}/{init_embedding_idx}_original.jpg"
        original_imgs = torch.cat(original_imgs, dim=0)
        save_image(original_imgs, original_image_path, normalize=True, nrow=4)


def main():
    opts = yaml.safe_load(open("opts.yaml"))
    # Set random seed.
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
    #BigGAN
    #G = BigGAN.Generator(**{**config,'skip_init':True,'no_optim':True})
    #G = Generator(code_dim=128, n_class=10,chn=64).to(device)
    G = Generator(code_dim=120, n_class=10, chn=64)
    print(G)
    #WGAN
    #G = LSUNGenerator(100)s
    #print(G)
    if resolution == 128:
        #original code
        #biggan_weigyhts = "pretrained_weights/138k/G_ema.pth"
        #biggan_weights = "pretrained_weights/G_ema_ep_82.pth"
        biggan_weights = "pretrained_weights/biggan-pytorch-901208_G.pth"
    else:
        #biggan_weights = "pretrained_weights/biggan_256_weights.pth"
        biggan_weights = "pretrained_weights/lsun-generator.pt"

    #G.load_state_dict(torch.load(f"{biggan_weights}"))
    G.load_state_dict(torch.load(biggan_weights))
    #print(torch.load("pretrained_weights/biggan-ptorch-901208_G.pth"))
    
    loaded_state_dict = torch.load(biggan_weights)
    '''
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        #print("name : ",n)
        name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
        
        if "_u" in n:
            name2 = n.replace("weight_u","u0")
            print(name2)
            new_state_dict[name2]=v
        if "_v" in n:
            name2 = n.replace("weight_bar","weight")
            new_state_dict[name2]=v
        #name = n.replace("weight_u.","weight")
        #name = n.replace("weight_v.","weight")
        
        new_state_dict[name] = v
    
    
    G.load_state_dict(new_state_dict)
    '''
    G = G.to(device)
    #G = nn.DataParallel(G).to(device)
    G.eval()

    model = opts["model"]
    net = nn.DataParallel(load_net(model)).to(device)
    alexnet_conv5 = model
    #model = ResNet34()
    net.eval()

    z_num = opts["z_num"]
    dloss_function = opts["dloss_function"]
    if dloss_function:
        half_z_num = z_num // 2
        print(f"Using diversity loss: {dloss_function}")
        if dloss_function == "features":
            if model != "alexnet":
                alexnet_conv5 = nn.DataParallel(load_net("alexnet_conv5")).to(device)
                alexnet_conv5.eval()

            else:
                alexnet_conv5 = model

    else:
        half_z_num = alexnet_conv5 = None

    print(f"BigGAN initialization time: {time.time() - start_time}")

    # Set up optimization.
    init_num = opts["init_num"]
    dim_z = dim_z_dict[resolution]
    #dim_z = 100
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    target_class = opts["target_class"]
    init_embeddings = get_initial_embeddings(
        resolution,
        init_method,
        init_num,
        min_clamp,
        max_clamp,
        dim_z,
        G,
        net,
        target_class,
        noise_std,
    )

    criterion = nn.CrossEntropyLoss()
    labels = torch.LongTensor([target_class] * z_num).to(device)
    #labels = torch.LongTensor([0,1,2,3,4,5,6,7,8,9] * 2).to(device)
    state_z = torch.get_rng_state()

    intermediate_dir = opts["intermediate_dir"]
    if intermediate_dir:
        print(f"Saving intermediate samples in {intermediate_dir}.")
        os.makedirs(intermediate_dir, exist_ok=True)

    final_dir = opts["final_dir"]
    if final_dir:
        print(f"Saving final samples in {final_dir}.")
        os.makedirs(final_dir, exist_ok=True)
        if model not in {"mit_alexnet", "mit_resnet18"}:
            original_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
            original_embeddings = torch.from_numpy(original_embeddings)
            #print("shape : ",original_embeddings[np.array(labels.cpu())].unsqueeze(0).shape)
            #print("shape : ",original_embeddings[target_class].unsqueeze(0).shape)
            original_embedding_clamped = torch.clamp(
                original_embeddings[target_class].unsqueeze(0), min_clamp, max_clamp
                #original_embeddings[np.array(labels.cpu())].unsqueeze(0), min_clamp, max_clamp
            )
            #repeat_original_embedding = original_embedding_clamped
            num_final = 4
            repeat_original_embedding = original_embedding_clamped.repeat(
                num_final, 1
            ).to(device)

        else:
            num_final = None
            repeat_original_embedding = None

    for (init_embedding_idx, init_embedding) in enumerate(init_embeddings):
        init_embedding_idx = str(init_embedding_idx).zfill(2)
        optim_comps = run_biggan_am(
            init_embedding,
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
            net,
            criterion,
            labels,
            dloss_function,
            half_z_num,
            alexnet_conv5,
            opts["alpha"],
            target_class,
            init_embedding_idx,
            intermediate_dir,
            opts["use_noise_layer"],
        )
        if final_dir:
            save_final_samples(
                optim_comps,
                min_clamp,
                max_clamp,
                device,
                model,
                state_z,
                num_final,
                dim_z,
                G,
                repeat_original_embedding,
                final_dir,
                init_embedding_idx,
            )


if __name__ == "__main__":
    main()
