import torch
import torchvision
import torchvision.transforms as transforms

import vgg
# import torchvision.models as mls
import torch.nn as nn
import torch

# from MLP_Mixer_Pytorch
# from MLP_Mixer_Pytorch.models.modeling import MlpMixer, CONFIGS

from modeling import MlpMixer ,CONFIGS
import configs

from resnet import ResNet34
import numpy as np
#net = vgg16_bn(num_classes=100)


# from mlp_mixer_pytorch import MLPMixer


transform_cifar100 = transforms.Compose(
    [
    transforms.Resize((32, 32)),
     transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761)),
     ])



transform_cifar10 = transforms.Compose(
    [
    # transforms.
    transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])


transform_cifar10_rec = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     transforms.Normalize((0.4957, 0.4836, 0.4371), (0.2383, 0.2357, 0.2392)),
     ])



transform_cifar10_Fake = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.4957, 0.4836, 0.4371), (0.2383, 0.2357, 0.2392)),
    ])

transform_cifar10_Fake2 = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5095, 0.4959, 0.4536), (0.2353, 0.2360, 0.2414) ),
    ])

transform_cifar10_Fake_3 = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.4922, 0.4832, 0.4426), (0.2468, 0.2398, 0.2393) ),
    ])


transform_cifar10_Fake_4 = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5243, 0.5066, 0.4444), (0.2451, 0.2457, 0.2435) ),
    ])

transform_cifar10_Fake_5 = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5299, 0.4902, 0.4374), (0.2382, 0.2424, 0.2453) ),
    ])



transform_no_norm = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    #  transforms.Normalize((0.5299, 0.4902, 0.4374), (0.2382, 0.2424, 0.2453) ),
    ])


# [0.5299, 0.4902, 0.4374][0.2382, 0.2424, 0.2453]
batch_size = 5

# FOR CIFAR100
# ACC: 8.548%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_threshold0.94_iter5_cifar100', transform=transform_cifar100)
# ACC: 9.722%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet_threshold0.94_seed300_cifar100_per500', transform=transform_cifar100)
# ACC: 5.473%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_mlpmixer_thres0.94_cifar100_iter5_TV', transform=transform_cifar100)
# ACC: 6.532%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_thres0.94_cifar100_single_TV', transform=transform_cifar100)

# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet_threshold0.94_cifar100_256000_ver2/Fake_ours_resnet_threshold0.94_cifar100_256000_ver1', transform=transform_cifar100)

# FOR MLP_MIXER_CIFAR100
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_mlpmixer_thres0.94_cifar100_iter5_TV', transform=transform_cifar100)

# ACC : 71.86%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar100_bs10_.94_.005_seed4_1011', transform=transform_cifar100)

# FOR ViT_Large
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_vit_large_patch16_224_cifar100_thres0.94_iter5_T', transform=transform_cifar100)


# FOR CIFAR10
# ACC: 71.196%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_threshold0.98_T_cifar10', transform=transform_cifar10)
# Accurcacy : 71.4199%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_threshold0.94_5iters_cifar10', transform=transform_cifar10)


# Acc: 79.696%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_thres0.3_cifar10_seed4', transform=transform_cifar10)

# ACC: 92% -> 75% -> 77% -> 78%
# 50000 images
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_1011/', transform=transform_cifar10_Fake)

# acc: 75%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_random_same_origin_ce/', transform=transform_cifar10_Fake2)

# acc: 83%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.9/', transform=transform_cifar10_Fake_3)

# acc: Fake_4: 91%; cifar10 stat: 87%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.98/', transform=transform_cifar10)


# acc: Fake_5: 90%, cifar10 stat: 84%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.99/', transform=transform_cifar10)
# NI : 99%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/NI_nodenorm_2k_9557_50k/', transform=transform_cifar10)

# DI: 96.6%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/DI_9557_50k/', transform=transform_cifar10)



# Channel-wise Mean: tensor([0.5102, 0.4779, 0.4185])
# Channel-wise Std: tensor([0.2494, 0.2475, 0.2437])  
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_no_denorm/", transform=transform_cifar10)
transform_cifar10_Fake_a = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.2561, 0.2223, 0.1893), (0.2634, 0.2516, 0.2405) ),
    ])
transform_cifar10_Fake_b = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5317, 0.4994, 0.4436), (0.2487, 0.2500, 0.2515) ),
    ])


# Channel-wise Mean: tensor([0.2561, 0.2223, 0.1893])
# Channel-wise Std: tensor([0.2634, 0.2516, 0.2405]) 
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_norm_scale_false/", transform=transform_no_norm)
# Channel-wise Mean: tensor([0.5317, 0.4994, 0.4436])                                                      
# Channel-wise Std: tensor([0.2487, 0.2500, 0.2515])   
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95/", transform=transform_no_norm)
# Channel-wise Mean: tensor([0.5229, 0.5126, 0.4724]) 
# Channel-wise Std: tensor([0.2318, 0.2285, 0.2304]) 
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_denorm_cifar_t/", transform=transform_cifar10)
# Channel-wise Mean: tensor([0.4920, 0.4826, 0.4470]) 
# Channel-wise Std: tensor([0.2000, 0.1978, 0.1993])  
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_denorm_cifar_f/', transform=transform_cifar10)



transform_cifar10_Fake_c = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5278, 0.4964, 0.4328), (0.2552, 0.2512, 0.2548) ),
    ])
transform_cifar10_Fake_d = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5051, 0.4743, 0.4084), (0.2576, 0.2504, 0.2478) ),
    ])
transform_cifar10_Fake_e = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5285, 0.5018, 0.4396), (0.2564, 0.2521, 0.2559) ),
    ])
transform_cifar10_Fake_f = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5333, 0.5064, 0.4448), (0.2613, 0.2575, 0.2637) ),
    ])
transform_cifar10_Fake_g = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.4922, 0.4835, 0.4483), (0.1979, 0.1943, 0.1940) ),
    ])

# Channel-wise Mean: tensor([0.4978, 0.4769, 0.4289])       
# Channel-wise Std: tensor([0.2408, 0.2336, 0.2282])    
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_no_denorm/", transform=transform_cifar10)
# Channel-wise Mean: tensor([0.2314, 0.2059, 0.1934]) 
# Channel-wise Std: tensor([0.2505, 0.2323, 0.2163])  
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_norm_scale_false/", transform=transform_no_norm)

# Channel-wise Mean: tensor([0.5264, 0.5015, 0.4542])                                                       │learning rate: 0.01
# Channel-wise Std: tensor([0.2392, 0.2364, 0.2374]) 
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001/", transform=transform_cifar10_Fake_d)

# Channel-wise Mean: tensor([0.5218, 0.5121, 0.4744])                                                       │learning rate: 0.01
# Channel-wise Std: tensor([0.2196, 0.2165, 0.2182])    
# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_denorm_cifar_t/", transform=transform_cifar10)

# Channel-wise Mean: tensor([0.4922, 0.4835, 0.4483])                                                       │learning rate: 0.01
# Channel-wise Std: tensor([0.1979, 0.1943, 0.1940])  
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_denorm_cifar_f/', transform=transform_cifar10)







# ACC: 89%???
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_1016_cls9/', transform=transform_cifar10)

# ACC: 78%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_random/', transform=transform_cifar10)

# ACC: 90% -> 92% (cifar 0)
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_random_test/', transform=transform_cifar10)

# ACC: 89%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_ci_lr0.1_wd0.9_test', transform=transform_cifar10)

# ACC: 77%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_ci_lr0.1_wd0.99_clamp0.5_test', transform=transform_cifar10)

# testset = torchvision.datasets.ImageFolder(root="/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_ci_lr0.1_cls9/", transform=transform_cifar10)

# ACC:81.324%

# ACC: 89%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_threshold0.94_seed4_cifar10/norm', transform=transform_cifar10)

# ACC: 0.52%
# testset = torchvision.datasets.ImageFolder(root='/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar100_bs20_.94_.005_seed4', transform=transform_cifar10)




def compute_channel_mean_and_std(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.var(2).sum(0)  # 분산을 바로 계산합니다.
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, torch.sqrt(std)  # 분산에서 표준편차로 변환

data_root_to_get_stat = [
    "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_norm_scale_false/",
    "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95/",
    "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_norm_scale_false/",
    "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001/"
]

transform_abcd = [
    transform_cifar10_Fake_c,
    transform_cifar10_Fake_d,
    transform_cifar10_Fake_e,
    transform_cifar10_Fake_f
    # transform_cifar10_Fake_g
]



transform_cifar10_Fake_h = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5203, 0.4880, 0.4233), (0.2420, 0.2408, 0.2366) ),
    ])

transform_cifar10_Fake_i = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.4974, 0.4655, 0.3978), (0.2416, 0.2365, 0.2241) ),
    ])

transform_cifar10_Fake_j = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5267, 0.5025, 0.4439), (0.2397, 0.2397, 0.2413) ),
    ])

transform_cifar10_Fake_k = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5050, 0.4818, 0.4197), (0.2385, 0.2361, 0.2317) ),
    ])
transform_cifar10_Fake_m = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.5075, 0.5051, 0.5057), (0.1483, 0.1445, 0.1498) ),
    ])

transform_cifar10_Fake_n = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     transforms.Normalize((0.4473, 0.4458, 0.4534), (0.2607, 0.2608, 0.2531) ),
    ])


# for
   # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter5_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter5_thres_both_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter10_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter10_thres_both_no_denorm/"

transform_cifar10_Fake_o = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5185, 0.4945, 0.4353), (0.2469, 0.2461, 0.2478) ),
    ])

transform_cifar10_Fake_p = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4951, 0.4718, 0.4097), (0.2463, 0.2426, 0.2380) ),
    ])
transform_cifar10_Fake_q = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5147, 0.4904, 0.4339), (0.2459, 0.2439, 0.2453) ),
    ])

transform_cifar10_Fake_r = transforms.Compose(
    [
    # transforms.
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4908, 0.4672, 0.4077), (0.2450, 0.2399, 0.2348) ),
    ])
transform_hi = [
    transform_cifar10_Fake_h,
    transform_cifar10_Fake_i
]
transform_jk = [
    transform_cifar10_Fake_j,
    transform_cifar10_Fake_k
]
transform_mn =[
    transform_cifar10_Fake_m,
    transform_cifar10_Fake_n
]
transform_mn =[
    transform_cifar10_Fake_m,
    transform_cifar10_Fake_n
]
transform_opqr=[
    transform_cifar10_Fake_o,
    transform_cifar10_Fake_p,
    transform_cifar10_Fake_q,
    transform_cifar10_Fake_r
]
data_root=[
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_no_denorm/',
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_norm_scale_false/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_denorm_cifar_t/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_denorm_cifar_f/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_norm_scale_false/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_denorm_cifar_t/",
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_denorm_cifar_f/'
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_prob/',
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_prob_no_denorm/',
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_ce_no_denorm/',
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_ce/'
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_ce_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_prob/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_ce_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_prob/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_prob/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_ce_no_denorm/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_ce_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_prob/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_both_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_both_thres_both_no_denorm/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_both_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_both_thres_both_no_denorm/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter5_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter5_thres_both_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter10_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter10_thres_both_no_denorm/"
    # "/home/jihwan/DF_synthesis/DI_9557_50k",
    # "/home/jihwan/DF_synthesis/NI_nodenorm_2k_9557_50k"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_threshold0.94_iter5_cifar100"
    # "/home/jihwan/DF_synthesis/Fake_ours_zo2m_resnet34_cifar10",
    # "/home/jihwan/DF_synthesis/Fake_denormours_zo2m_resnet34_cifar10_5seed/",
    # "/home/jihwan/DF_synthesis/Fake_ours_zo2m_resnet34_cifar10_5seed/"
    # "/home/jihwan/DF_synthesis/Fake_multiseed_prob09997_ours_zo2m_resnet34_cifar10/"
    './Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_1011'
]

# for i in data_root:


#     # testset = torchvision.datasets.CIFAR10(root='/home/jihwan/DF_synthesis/data/CIFAR10', train=True,
#     #                                     download=True, transform=transform_cifar10)
#     testset = torchvision.datasets.ImageFolder(root=i, transform=transform_no_norm)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                             shuffle=False, num_workers=2)

#     print(f'root of imgs: {i}')
#     mean, std = compute_channel_mean_and_std(testloader)
#     print("Channel-wise Mean:", mean)
#     print("Channel-wise Std:", std)


# breakpoint()


# 채널별 평균 및 분산을 계산하는 함수







# print(f'mean of the img: {np.mean(testloader.dataset, axis=(0,1,2))/255}')
# print(f'var of the img: {np.std(testloader.dataset, axis=(0,1,2))/255}')


# For mlp-mixer
# config = CONFIGS['Mixer-B_16']
# net = MlpMixer(config, 224, num_classes=100, patch_size=16, zero_head=True)

# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from PIL import Image
# import requests


# for vit b-16 cifar10
# feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
# net = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')

# for vit b-16 cifar100

# import timm
# import torch
# from torch import nn

# net = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",
# pretrained=False)
# net.head = nn.Linear(net.head.in_features, 100)
# net.load_state_dict(
#     torch.hub.load_state_dict_from_url(
#         "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
#         map_location="cpu",
#         file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
#     )
# )


# for ViT L-16 cifar10
from transformers import ViTImageProcessor,AutoFeatureExtractor, AutoModelForImageClassification

# feature_extractor = AutoFeatureExtractor.from_pretrained("tzhao3/vit-L-CIFAR10")
# net = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-CIFAR10")


# # for ViT L-16 for cifar100
# processor = ViTImageProcessor.from_pretrained("tzhao3/vit-L-CIFAR100")
# net = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-CIFAR100")


# for DeiT B-16 for cifar10

net = AutoModelForImageClassification.from_pretrained("edumunozsala/vit_base-224-in21k-ft-cifar100")
# for DeiT B-16 for cifar100

# net = AutoModelForImageClassification.from_pretrained("tzhao3/DeiT-CIFAR100")

# net = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')


# model.load_from(np.load(args.pretrained_dir))
#   /  model.to(args.device)

# net = vgg.__dict__['vgg16_bn'](num_classes=100)
# breakpoint()  


# for mlp-mixer
# net.load_state_dict(torch.load('/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar100_mlp_mixerb16_8434_checkpoint.bin'),strict=True)
# net.load_state_dict(torch.load('/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar10-mlp_mixerb16_9709_checkpoint.bin'),strict=True)





        # net.load_state_dict(checkpoint)


# ResNet CIFAR10
# net = ResNet34(num_classes=10)
# net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar10_resnet34_9557.pt'),strict=True)

# RestNet CIFAR100
# net = ResNet34(num_classes=100)
# net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar100_resnet34_7802.pth'),strict=True)

net.eval().cuda()

# for i in [4]:
    # test
    # testset = torchvision.datasets.ImageFolder(root=f'/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_1011_{i}/', transform=transform_cifar10)

    # # testset = torchvision.datasets.ImageFolder(root=f'/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_seed_161_init_random_same_cls_4678_{i}/', transform=transform_cifar10)

    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                      shuffle=False, num_workers=2)

    # print(f'num of imgs: {len(testloader.dataset)}')

# processor.cuda()
data_root=[
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_no_denorm/',
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_norm_scale_false/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_denorm_cifar_t/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.95_denorm_cifar_f/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_norm_scale_false/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_thres_img_0.001_denorm_cifar_t/",
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_prob/',
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_prob_no_denorm/',
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_ce_no_denorm/',
    # '/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_ce/',
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_ce_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_thres_prob/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_ce_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_real_thres_prob/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_prob/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.0005_whiletest_thres_ce_no_denorm/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_ce/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_ce_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_prob_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_.001_whiletest_thres_prob/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_both_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_both_thres_both_no_denorm/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_both_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.0005_both_thres_both_no_denorm/"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter5_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter5_thres_both_no_denorm/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter10_thres_both/",
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_cifar10_init_mean_thres_0.001_both_iter10_thres_both_no_denorm/"
    # "/home/jihwan/DF_synthesis/DI_9557_50k",
    # "/home/jihwan/DF_synthesis/NI_nodenorm_2k_9557_50k"
    # "/home/jihwan/DF_synthesis/Fake_ours_resnet34_threshold0.94_iter5_cifar100"
    # "/home/jihwan/DF_synthesis/Fake_ours_zo2m_resnet34_cifar10",
    # "/home/jihwan/DF_synthesis/Fake_denormours_zo2m_resnet34_cifar10_5seed/",
    # "/home/jihwan/DF_synthesis/Fake_ours_zo2m_resnet34_cifar10_5seed/"
    # "/home/jihwan/DF_synthesis/Fake_multiseed_prob09997_ours_zo2m_resnet34_cifar10/",
    './Fake_ours_resnet34_cifar10_bs20_.98_.005_seed4_1011'
]


print('We did on CIFAR10 statistics')


# for index, i in enumerate(data_root):
#     # testset = torchvision.datasets.ImageFolder(root=i, transform=transform_opqr[index])
#     testset = torchvision.datasets.ImageFolder(root=i, transform=transform_cifar10)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                             shuffle=False, num_workers=2)


# testset = torchvision.datasets.CIFAR100(root='/home/jihwan/DF_synthesis/data/CIFAR100', train=True,
#                                        download=True, transform=transform_no_norm)

# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
# mean, std = compute_channel_mean_and_std(testloader)
# print("Channel-wise Mean:", mean)
# print("Channel-wise Std:", std)


# testset = torchvision.datasets.CIFAR100(root='/home/jihwan/DF_synthesis/data/CIFAR100', train=True,
#                                        download=True, transform=transform_cifar100)

# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
    # print(f'num of imgs: {len(testloader.dataset)}')                                         
    # correct = 0
    # total = 0
    # # gt  = torch.LongTensor([i]*batch_size)
    # # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         # 신경망에 이미지를 통과시켜 출력을 계산합니다
    #         # breakpoint()

    #         # images = (images + 1) / 2
    #         # inputs = processor(images= images,return_tensors="pt")
    #         # outputs = net(images.cuda(), labels.cuda())
    #         # outputs = net(images.cuda())
    #         # inputs_tensors = inputs["pixel_values"].cuda()
    #         # outputs = net(inputs_tensors)
    #         outputs,f1,f2,f3,f4,f5 = net(images.cuda())
    #         # pred_logits = outputs.logits
    #         # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
    #         # breakpoint()
    #         _, predicted = torch.max(outputs.data, 1)
    #         # print(labels)
    #         # print(predicted)
    #         # print(total)
    #         # print(correct)
    #         # print(f'predicted: {predicted}, labels: {labels}')
    #         # print(f'gt: {gt}')
    #         # _, predicted = torch.max(pred_logits.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels.cuda()).sum().item()
    #     # correct += (predicted == gt.cuda()).sum().item()
    #     # print(labels)


    # hii = float(correct / total)
    # print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {100.0 * hii} %')

# transform_x = transforms.Compose(
#     [
#     # transforms.
#     transforms.Resize((32,32)),
#     transforms.ToTensor(),
#     #  transforms.Normalize((0.2314, 0.2059, 0.1934), (0.2505, 0.2323, 0.2163) ),
#     ])

testset = torchvision.datasets.CIFAR100(root='/home/jihwan/DF_synthesis/data/CIFAR100', train=False,
                                       download=True, transform=transform_cifar10)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# mean, std = compute_channel_mean_and_std(testloader)
# print("Channel-wise Mean:", mean)
# print("Channel-wise Std:", std)

print(f'num of imgs: {len(testloader.dataset)}')                                         
correct = 0
total = 0
# gt  = torch.LongTensor([i]*batch_size)
# 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # 신경망에 이미지를 통과시켜 출력을 계산합니다
        # breakpoint()

        # images = (images + 1) / 2
        # inputs = processor(images= images,return_tensors="pt")
        # outputs = net(images.cuda(), labels.cuda())
        outputs = net(images.cuda())
        # inputs_tensors = inputs["pixel_values"].cuda()
        # outputs = net(inputs_tensors)
        # outputs,f1,f2,f3,f4,f5 = net(images.cuda())
        pred_logits = outputs.logits
        # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
        # breakpoint()
        # _, predicted = torch.max(outputs.data, 1)
        # print(labels)
        # print(predicted)
        # print(total)
        # print(correct)
        # print(f'predicted: {predicted}, labels: {labels}')
        # print(f'gt: {gt}')
        _, predicted = torch.max(pred_logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        # correct += (predicted == gt.cuda()).sum().item()
        # print(labels)


hii = float(correct / total)
print(f'Accuracy of the network on the {testloader.dataset} test images: {100.0 * hii} %')


