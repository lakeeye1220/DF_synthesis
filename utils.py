import os
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

from robustness import datasets, model_utils

dim_z_dict = {128: 120, 256: 140, 512: 128} #원래는 100대신에 120
attn_dict = {128: "64", 256: "128", 512: "64"}
max_clamp_dict = {128: 0.83, 256: 0.2}
min_clamp_dict = {128: -0.88, 256: -0.2} #256 0.61,-0.59
DATA_PATH_DICT = {
    "CIFAR": "/path/tools/cifar",
    "RestrictedImageNet": "/mnt/raid/qi/ILSVRC2012_img_train/ImageNet",
    "ImageNet": "/path/tools/imagenet",
    "H2Z": "/path/tools/horse2zebra",
    "A2O": "/path/tools/apple2orange",
    "S2W": "/path/tools/summer2winter_yosemite",
}


def get_config(resolution):
    return {
        "G_param": "SN",
        "D_param": "SN",
        "G_ch": 96,
        "D_ch": 96,
        "D_wide": True,
        "G_shared": True,
        "shared_dim": 128,
        "dim_z": dim_z_dict[resolution],
        "hier": True,
        "cross_replica": False,
        "mybn": False,
        "G_activation": nn.ReLU(inplace=True),
        "G_attn": attn_dict[resolution],
        "norm_style": "bn",
        "G_init": "ortho",
        "skip_init": True,
        "no_optim": True,
        "G_fp16": False,
        "G_mixed_precision": False,
        "accumulate_stats": False,
        "num_standing_accumulations": 16,
        "G_eval_mode": True,
        "BN_eps": 1e-04,
        "SN_eps": 1e-04,
        "num_G_SVs": 1,
        "num_G_SV_itrs": 1,
        "resolution": resolution,
        "n_classes": 1000,
    }


def load_mit(model_name):
    model_file = f"{model_name}_places365.pth.tar"
    if not os.access(model_file, os.W_OK):
        weight_url = f"http://places2.csail.mit.edu/models_places365/{model_file}"
        os.system(f"wget {weight_url}")

    model = models.__dict__[model_name](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for (k, v) in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    return model


def load_madrylab_imagenet(arch):
    data = "ImageNet"
    dataset_function = getattr(datasets, data)
    dataset = dataset_function(DATA_PATH_DICT[data])
    model_kwargs = {
        "arch": arch,
        "dataset": dataset,
        "resume_path": f"madrylab_models/{data}.pt",
        "state_dict_path": "model",
    }
    (model, _) = model_utils.make_and_restore_model(**model_kwargs)

    return model


def load_net(model_name):
    print(f"Loading {model_name} classifier...")
    if model_name == "resnet50":
        return models.resnet50(pretrained=True)

    elif model_name == "alexnet":
        return models.alexnet(pretrained=True)

    elif model_name == "alexnet_conv5":
        return models.alexnet(pretrained=True).features

    elif model_name == "inception_v3":
        # Modified the original file in torchvision/models/inception.py!!!
        return models.inception_v3(pretrained=True)

    elif model_name == "mit_alexnet":
        return load_mit("alexnet")

    elif model_name == "mit_resnet18":
        return load_mit("resnet18")
    
    elif model_name =="resnet18_CUB200":
        net = models.resnet18(pretrained=False)
        number_of_features = net.fc.in_features
        #net.linear = nn.Linear(number_of_features,200)
        net.fc =  nn.Linear(number_of_features,200)
        checkpoint = torch.load('./CUB_best_model.pt')
        new_state_dict = OrderedDict()
        for key,value in checkpoint['model'].items():
            if 'linear.' in key:
                new_state_dict[key.replace('linear.','fc.')] = value
            else:
                new_state_dict[key]=value
        print(new_state_dict)
        
        #print(checkpoint)
        net.load_state_dict(new_state_dict)
        return net

    
    elif model_name =="resnet18_CelebA":
        from ResNet_CelebA import resnet18
        model = resnet18(2)
        model.load_state_dict(torch.load('./CelebA/resnet18_CelebA_9812.pt'))
        return model

    elif model_name == "madrylab_resnet50":
        return load_madrylab_imagenet("resnet50")
        
    elif model_name =='resnet34':
        from resnet import ResNet34
        net = ResNet34()
        net.load_state_dict(torch.load('cifar10_resnet34_9557.pt'))
        #net.load_state_dict(torch.load('tiny_resnet34_7356.pt'))
        return net
    
    elif model_name =='resnet34_cifar100':
        from resnet import ResNet34
        net = ResNet34()
        net.load_state_dict(torch.load('cifar100_resnet34_7802.pth'))
        #net.load_state_dict(torch.load('tiny_resnet34_7356.pt'))
        return net
    
    elif model_name =='resnet34_tinyImageNet':
        from resnet import ResNet34
        net = ResNet34(num_classes=200)
        net.load_state_dict(torch.load('tiny_resnet34_7356.pt'),strict=False)
        return net
    
    elif model_name =='resnet50':
        from ResNet_FFHQ import resnet50
        net = resnet50(pretrained=True) #이거 이미지넷이다
        #net.load_state_dict(torch.load('cifar10_resnet34_9557.pt'))
        return net
    
    elif model_name=='vgg19_flower':
        net = models.vgg19(pretrained=False)
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)), # First layer
                          ('relu', nn.ReLU()), # Apply activation function
                          ('fc2', nn.Linear(4096, 102)), # Output layer
                          ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                          ]))
        net.classifier = classifier
        checkpoint = torch.load('./Oxford_Flower102/classifier.pth')
        net.load_state_dict(checkpoint['state_dict'])
        return net
    
    elif model_name=='vgg16_flower':
        net = models.vgg16(pretrained=True)
        input_size = net.classifier[0].in_features
        output_size = 102
        classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_size, input_size // 8)),
                ('relu1', nn.ReLU()),
                ('droupout', nn.Dropout(p=0.20)),

                ('fc2', nn.Linear(input_size // 8, input_size // 32)),
                ('relu2', nn.ReLU()),
                ('droupout', nn.Dropout(p=0.20)),

                ('fc3', nn.Linear(input_size // 32, input_size // 128)),
                ('relu3', nn.ReLU()),
                ('droupout', nn.Dropout(p=0.20)),

                ('fc4', nn.Linear(input_size // 128, output_size))
                #('softmax', nn.LogSoftmax(dim=1))
            ])
        )
        net.classifier = classifier
        checkpoint = torch.load('./Oxford_Flower102/classifier_vgg16_9052.pth')
        net.load_state_dict(checkpoint['state_dict'])
        return net
    
    elif model_name=='resnet_flower':
        network = models.resnet34(pretrained=False)
        number_of_features = network.fc.in_features
        network.fc = nn.Linear(number_of_features, 102)
        print(network)
        network.load_state_dict(torch.load('resnet34-333f7ec4.pth'))

    else:
        raise ValueError(f"{model_name} is not a supported classifier...")
