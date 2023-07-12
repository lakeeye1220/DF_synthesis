import os
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

#from robustness import datasets, model_utils

dim_z_dict = {128: 120, 256: 140, 512: 128} #원래는 100대신에 120
attn_dict = {128: "64", 256: "128", 512: "64"}
max_clamp_dict = {128: 0.83, 256: 0.7}
min_clamp_dict = {128: -0.88, 256: -0.7} #256 0.61,-0.59
DATA_PATH_DICT = {
    "CIFAR": "/path/tools/cifar",
    "RestrictedImageNet": "/mnt/raid/qi/ILSVRC2012_img_train/ImageNet",
    "ImageNet": "/path/tools/imagenet",
    "H2Z": "/path/tools/horse2zebra",
    "A2O": "/path/tools/apple2orange",
    "S2W": "/path/tools/summer2winter_yosemite",
}

MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
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

def viz_weights(model_name,model,num_class,file_path):
    weight_norm=[]
    if 'vit' in model_name:
        weight=model.classifier.weight
        bias=model.calssifier.bias.unsqueeze(-1)
    elif 'cct' in model_name:
        weight=model.module.classifier.fc.weight
        bias=model.module.classifier.fc.bias.unsqueeze(-1)
    else:
        weight=model.linear.weight
        bias=model.linear.bias.unsqueeze(-1)

    weight=torch.cat((weight,bias),dim=1)
    for i in range(num_class):
        weight_norm.append(torch.norm(weight[i],2).item())
    plt.figure()
    classes=np.arange(num_class)
    plt.scatter(classes,weight_norm)
    plt.xlabel('Class Index')
    plt.ylabel('Weight Norm')
    plt.xlim(0,weight.shape[0])
    plt.savefig(os.path.join(file_path,model_name+'_weight_norm.pdf'),bbox_inches='tight')
    np.savetxt(os.path.join(file_path,model_name+'_weight_norm.csv'), weight_norm, delimiter=",", fmt='%.2f')
    plt.close()


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
        #print(new_state_dict)
        
        #print(checkpoint)
        net.load_state_dict(new_state_dict)
        return net

    elif model_name =="resnet18_CelebA":
        from ResNet_CelebA import resnet18
        model = resnet18(2)
        model.load_state_dict(torch.load('./classifier_pretrained_weights/resnet18_CelebA_9812.pt'))
        return model

    elif model_name == "madrylab_resnet50":
        return load_madrylab_imagenet("resnet50")
        
    elif model_name =='resnet34':
        from resnet import ResNet34
        import torch
        net = ResNet34()
        net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar10_resnet34_9557.pt'),strict=False)
        #net.load_state_dict(torch.load('tiny_resnet34_7356.pt'))
        return net
    
    elif model_name =='resnet34_cifar100':
        from resnet import ResNet34
        import torch
        net = ResNet34(num_classes=100)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar100_resnet34_7802.pth'))
        #net.load_state_dict(torch.load('tiny_resnet34_7356.pt'))
        return net
    
    elif model_name =='resnet34_tinyImageNet':
        from resnet import ResNet34
        net = ResNet34(num_classes=200)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/tiny_resnet34_7356.pt'),strict=False)
        return net
    
    elif model_name =='resnet50':
        from ResNet_FFHQ import resnet50
        net = resnet50(pretrained=True) #이거 이미지넷이다
        #net.load_state_dict(torch.load('cifar10_resnet34_9557.pt'))
        return net
    
    elif model_name=='vgg19_flower':
        import torch
        import torch.nn as nn
        net = models.vgg19(pretrained=False)
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)), # First layer
                          ('relu', nn.ReLU()), # Apply activation function
                          ('fc2', nn.Linear(4096, 102)), # Output layer
                          ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                          ]))
        net.classifier = classifier
        checkpoint = torch.load('./classifier_pretrained_weights/flower102_vgg19_classifier.pth')
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
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
        checkpoint = torch.load('./classifier_pretrained_weights/classifier_vgg16_9052.pth')
        net.load_state_dict(checkpoint['state_dict'])
        return net
    
    elif model_name=='resnet_flower':
        network = models.resnet34(pretrained=False)
        number_of_features = network.fc.in_features
        network.fc = nn.Linear(number_of_features, 102)
        print(network)
        network.load_state_dict(torch.load('./classifier_pretrained_weights/resnet34-333f7ec4.pth'))
    
    elif model_name=="vit_cifar":
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        from PIL import Image
        import requests

        #url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
        #image = Image.open(requests.get(url, stream=True).raw)
        feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        
        return feature_extractor, model

    elif model_name=="vit_cifar100": #9316
        import timm
        import torch
        from torch import nn

        model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        pretrained=False)
        model.head = nn.Linear(model.head.in_features, 100)
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
                map_location="cpu",
                file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
            )
        )
        #from transformers import ViTFeatureExtractor, ViTForImageClassification
        #from PIL import Image
        #feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        #model = ViTForImageClassification.from_pretrained('edadaltocg/vit_base_patch16_224_in21k_ft_cifar100')

        return None, model

    elif model_name=="vit_food":
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        from PIL import Image
        import requests

        #url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
        #image = Image.open(requests.get(url, stream=True).raw)
        feature_extractor = ViTFeatureExtractor.from_pretrained('eslamxm/vit-base-food101') #Accuracy: 0.8539
        model = ViTForImageClassification.from_pretrained('eslamxm/vit-base-food101')
        
        return feature_extractor, model

    elif model_name=="vit_flower":
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        from PIL import Image
        import requests
        #url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
        #image = Image.open(requests.get(url, stream=True).raw)
        feature_extractor = ViTFeatureExtractor.from_pretrained('chanelcolgate/vit-base-patch16-224-finetuned-flower')
        model = ViTForImageClassification.from_pretrained('chanelcolgate/vit-base-patch16-224-finetuned-flower')
        
        return feature_extractor, model

    elif model_name=="vit_face":
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        feature_extractor = ViTFeatureExtractor.from_pretrained('jayanta/google-vit-base-patch16-224-face')
        model = ViTForImageClassification.from_pretrained('jayanta/google-vit-base-patch16-224-face')
        
        return feature_extractor, model

    elif model_name=="vit_flowers102":
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        from transformers.models.auto.modeling_auto import AutoModelForImageClassification
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k',num_labels=102,ignore_mismatched_sizes=True,image_size=224)
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',num_labels=102,image_size=224)
        #ckpt = torch.load('./vit-finetune/output/flowers102/version_2/checkpoints/best_step_700_acc_9931.ckpt')["state_dict"]
        #ckpt = torch.load('./vit-finetune/output/flowers102/version_4/checkpoints/best_step_1000_acc_9902.ckpt')["state_dict"]
        #ckpt = torch.load('./vit-finetune/output/flowers102/version_6/checkpoints/best_step_1400_acc_9902.ckpt')["state_dict"]
        #ckpt = torch.load('./vit-finetune/output/flowers102/version_6/checkpoints/last.ckpt')["state_dict"]
        '''
        model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=102,
            ignore_mismatched_sizes=True,
            image_size=224,
        )
        # Remove prefix from key names
        
        new_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("net"):
                k = k.replace("net" + ".", "")
                new_state_dict[k] = v
        '''
        #print(new_state_dict)
        #print(model)
        #print(feature_extractor)
        #feature_extractor.load_state_dict(new_state_dict,strict=True)
        #model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        return feature_extractor, model

    elif model_name=="cct_cifar":
        from Compact_Transformers.src import cct_7_3x1_32
        model = cct_7_3x1_32(pretrained=True, progress=True)
        return model

    elif model_name=="cct_flowers_fromScratch":
        from Compact_Transformers.src import cct_7_7x2_224_sine
        model = cct_7_7x2_224_sine(pretrained=True, progress=True)
        model.classifier
        model.eval()
        return model

    elif model_name=="cct_flowers_finetune":
        from Compact_Transformers.src import cct_14_7x2_384_fl
        model = cct_14_7x2_384_fl(pretrained=True, progress=True)
        model.eval()
        return model
    
    elif model_name=='resnet50_iitpet':
        import timm
        import torch
        from torch import nn
        import torchvision
        #from functools import partial
        #import pickle
        #pickle.load = partial(pickle.load, encoding="latin1")
        #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

        model = timm.create_model('resnet50-oxford-iiit-pet', pretrained=True)
        #model =timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=5)
        #model = models.resnet50(pretrained=False)
        #print(model)
        
        #model.fc = nn.Linear(model.fc.in_features, 37)
       
        #checkpoint = torch.hub.load_state_dict_from_url(
                #"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/",
        #        "https://huggingface.co/nateraw/resnet50-oxford-iiit-pet-v2/blob/main/pytorch_model.bin",
        #        map_location="cpu",
        #        #file_name="resnet50_a1_0-14fe96d1.pth"
        #    )
        #print(checkpoint)
        #model = torch.load(checkpoint, map_location=lambda storage, loc: storage, pickle_module=pickle)
        #model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        '''
        #from modelz import ResnetModel
        #model = ResnetModel.from_pretrained('nateraw/resnet50-oxford-iiit-pet-v2')
        model = timm.create_model('nateraw/resnet50-oxford-iiit-pet-v2')
        model.eval()
        '''
        return model
    
    elif model_name =='resnet_place365':
        import torch
        from torch.autograd import Variable as V
        import torchvision.models as models
        from torchvision import transforms as trn
        from torch.nn import functional as F
        import os
        from PIL import Image

        # th architecture to use
        arch = 'resnet18'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()


        # load the image transformer
        centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        return model
        
    elif model_name == 'resnet50_cartoon':
        import torchvision.models as models
        import torch
        model = models.__dict__['resnet50']()
        checkpoint = torch.load('../training_data/examples/imagenet/checkpoint.pth')
        #print(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        #print(model)
        return model

        #inputs = processor(image, return_tensors="pt")



    elif model_name == 'swin-tiny-patch4-window7-224-finetuned-cifar10':

        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10")
        model = AutoModelForImageClassification.from_pretrained("nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10")

        model.eval()

        return model, feature_extractor
    

    elif model_name =="vit-L-CIFAR10":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("tzhao3/vit-L-CIFAR10")
        model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-CIFAR10")

        model.eval()

        return model, feature_extractor
    

    elif model_name =="vit-L-CIFAR100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("tzhao3/vit-L-CIFAR100")
        model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-CIFAR100")

        model.eval()

        return model, feature_extractor
    

    elif model_name =="swin-base-finetuned-cifar100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")
        model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")

        model.eval()

        return model, feature_extractor
    
    elif model_name =="swin-tiny-finetuned-cifar100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-tiny-finetuned-cifar100")
        model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-tiny-finetuned-cifar100")

        model.eval()

        return model, feature_extractor
    

    elif model_name =="swin-base-finetuned-cifar10":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("alfredcs/swin-cifar10")
        model = AutoModelForImageClassification.from_pretrained("alfredcs/swin-cifar10")

        model.eval()

        return model, feature_extractor
    

    elif model_name =="swin-small-finetuned-cifar100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-small-finetuned-cifar100")
        model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-small-finetuned-cifar100")

        model.eval()

        return model, feature_extractor



       
    else:
        raise ValueError(f"{model_name} is not a supported classifier...")
