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
from torch.autograd import Variable
from resnet import ResNet34
import numpy as np
import math

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
    alpha,
    target_class,
    intermediate_dir,
    use_noise_layer,
    total_class,
    writer
):
    
    embedding_layer = nn.Embedding(total_class,128) # num_embeddings, embedding_dim
    optim_embedding = embedding_layer(torch.LongTensor([target_class])).detach()
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

    T = torch.empty((1,))
    torch.nn.init.normal(T,5.0,1)
    T.requires_grad_(True)

    optimizer = optim.Adam(optim_params+[T], lr=lr, weight_decay=dr)
    torch.set_rng_state(state_z)
    #labels_target = torch.LongTensor([0,1,2,3,4,5,6,7,8,9] * 2).to(device)
    total_loss = []
    total_T = []
    total_prob =[]
    for epoch in range(n_iters):
        #zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)s

        # for saving best img(when CE loss is lowest)
        best_loss = 0
        best_img = None

        for z_step in range(steps_per_z):
            zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)
            optimizer.zero_grad()
            if use_noise_layer:
                z_hats = noise_layer(zs)
            else:
                z_hats = zs

            clamped_embedding = torch.clamp(optim_embedding, min_clamp, max_clamp)
            repeat_clamped_embedding = clamped_embedding.repeat(z_num, 1).to(device)
            gan_images_tensor = G(z_hats, repeat_clamped_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=32 #Flower 224, CelebA 128
            )
            pred_logits,_,_,_,_,_ = net(resized_images_tensor)
            loss = criterion(pred_logits/T.cuda(), labels)
            pred_probs = nn.functional.softmax(pred_logits, dim=1)

            #softmax_o_T = F.softmax(pred_logits, dim = 1).mean(dim = 0)
            softmax_o_T = pred_probs.mean(dim=0)
            loss_entropy = (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(z_num)).sum())
            loss +=loss_entropy

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
            
            
            avg_target_prob = pred_probs[:, target_class].mean().item()
            #avg_target_prob = pred_probs[:, np.array(labels_target.cpu())].mean().item()
            if z_step %50 == 0:
                log_line = f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
                log_line += f"Average Target Probability:{avg_target_prob:.4f}\t"
                log_line += f"Temperature Probability: {T.item():.3f}\t"
                log_line += f"CE loss: {loss.item():.4f}\t"
                log_line += f"entropy loss: {loss_entropy.item():.4f}"
                print(log_line)
            total_loss.append(loss.item())
            total_T.append(T.item())
            total_prob.append(avg_target_prob)

            # tensorboard logging
            writer.add_scalar(f"CE loss", loss.item(), z_step)
            writer.add_scalar(f"avg target prob", avg_target_prob, z_step)
            writer.add_scalar(f"T", T.item(), z_step)
            writer.add_scalar(f"entropy loss", loss_entropy.item(), z_step)
            
            if intermediate_dir:
                if z_step %50 ==0:
                    global_step_id = epoch * steps_per_z + z_step
                    img_f = f"{global_step_id:0=7d}.jpg"
                    output_image_path = f"{intermediate_dir}/{img_f}"
                    save_image(
                    gan_images_tensor, output_image_path, normalize=True, nrow=10
                )
                    
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_img = gan_images_tensor

            torch.cuda.empty_cache()
        
        if best_img is not None:
            best_path = f"{intermediate_dir}/best.jpg"
            save_image(best_img, best_path, normalize=True, nrow=10)

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
    model,
    state_z,
    num_final,
    dim_z,
    G,
    repeat_original_embedding,
    final_dir,
    prefix_idx
):
    optim_embedding = optim_comps["optim_embedding"]
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

    final_image_path = f"{final_dir}/{prefix_idx}.jpg"
    optim_imgs = torch.cat(optim_imgs, dim=0)
    save_image(optim_imgs, final_image_path, normalize=True, nrow=4)
    np.save(
        f"{final_dir}/{prefix_idx}.npy",
        repeat_optim_embedding.detach().cpu().numpy(),
    )
    if optim_comps["use_noise_layer"]:
        torch.save(
            optim_comps["noise_layer"].state_dict(),
            f"{final_dir}/{prefix_idx}_noise_layer.pth",
        )

    original_image_path = f"{final_dir}/{prefix_idx}_original.jpg"
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
    G = BigGAN.Generator(**config)
    if resolution == 128:
        biggan_weights = "pretrained_weights/138k/G_ema.pth"
    else:
        biggan_weights = "pretrained_weights/biggan_256_weights.pth"

    G.load_state_dict(torch.load(f"{biggan_weights}"), strict=False)
    G = nn.DataParallel(G).to(device)
    G.eval()

    model = opts["model"]
    net = nn.DataParallel(load_net(model)).to(device)
    net.eval()

    z_num = opts["z_num"]
    dloss_function = opts["dloss_function"]
    #if dloss_function:
    half_z_num = z_num // 2

    print(f"BigGAN initialization time: {time.time() - start_time}")

    # Set up optimization.
    #init_num = opts["init_num"]
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    target_class = opts["target_class"]
    
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
        #if model not in {"mit_alexnet", "mit_resnet18"}:
        original_embeddings = np.load(f"biggan_embeddings_{resolution}.npy")
        original_embeddings = torch.from_numpy(original_embeddings)
        original_embedding_clamped = torch.clamp(
            original_embeddings[target_class].unsqueeze(0), min_clamp, max_clamp
            #original_embeddings[np.array(labels.cpu())].unsqueeze(0), min_clamp, max_clamp
        )
        #repeat_original_embedding = original_embedding_clamped
        num_final = 4
        repeat_original_embedding = original_embedding_clamped.repeat(num_final, 1).to(device)
    
    # tensorboard writer
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(f'{final_dir}/result')

    for i in range(int(opts["n_iters"])):
        optim_comps = run_biggan_am(
            #init_embedding,
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
            opts["alpha"],
            target_class,
            intermediate_dir,
            opts["use_noise_layer"],
            opts["total_class"],
            writer=writer
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
                i
            )

if __name__ == "__main__":
    main()
