import os
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from transformers import (
    ViTForImageClassification,
    pipeline,
    AutoImageProcessor,
    ViTConfig,
    ViTModel,
)

from transformers.modeling_outputs import (
    ImageClassifierOutput,
    BaseModelOutputWithPooling,
)

from PIL import Image
import torch
from torch import nn
from typing import Optional, Union, Tuple


class CustomViTModel(ViTModel):
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output[:, 1:, :].mean(dim=1)

        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomViTForImageClassification(ViTForImageClassification):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = CustomViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        loss = None

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )







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
        
    elif model_name =='resnet34_cifar10':
        from resnet import ResNet34
        import torch
        net = ResNet34()
        net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar10_resnet34_9557.pt'),strict=True)
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
        net.load_state_dict(torch.load('./classifier_pretrained_weights/tiny_resnet34_7356.pt'),strict=True)
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

    elif model_name == 'mlp_mixer_cifar10':
        from modeling import MlpMixer ,CONFIGS
        # import configs
        import torch.nn as nn
        import torch

        config = CONFIGS['Mixer-B_16']
        net = MlpMixer(config, 224, num_classes=10, patch_size=16, zero_head=True)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar10-mlp_mixerb16_9709_checkpoint.bin'))
        net.eval()

        return net
    
    elif model_name == 'mlp_mixer_cifar100':
        from modeling import MlpMixer ,CONFIGS
        # import configs
        import torch.nn as nn
        import torch

        config = CONFIGS['Mixer-B_16']
        net = MlpMixer(config, 224, num_classes=100, patch_size=16, zero_head=True)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar100_mlp_mixerb16_8434_checkpoint.bin'))
        net.eval()

        return net

    elif model_name == 'mlp_mixer_cifar10_L_1k':
        from modeling import MlpMixer ,CONFIGS
        # import configs
        import torch.nn as nn
        import torch

        config = CONFIGS['Mixer-L_16']
        net = MlpMixer(config, 224, num_classes=10, patch_size=16, zero_head=True)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/Mixer-L_16_1k_9698_checkpoint.bin'))
        net.eval()

        return net
    
    elif model_name == 'mlp_mixer_cifar10_L_21k':
        from modeling import MlpMixer ,CONFIGS
        # import configs
        import torch.nn as nn
        import torch

        config = CONFIGS['Mixer-L_16-21k']
        net = MlpMixer(config, 224, num_classes=10, patch_size=16, zero_head=True)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/Mixer-L16_21k_9844_checkpoint.bin'))
        net.eval()

        return net


    elif model_name == 'mlp_mixer_cifar100_L_1k':
        from modeling import MlpMixer ,CONFIGS
        # import configs
        import torch.nn as nn
        import torch

        config = CONFIGS['Mixer-L_16']
        net = MlpMixer(config, 224, num_classes=100, patch_size=16, zero_head=True)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/Mixer_L16_cifar100_1k_8569_checkpoint.bin'))
        net.eval()

        return net
    
    elif model_name == 'mlp_mixer_cifar100_L_21k':
        from modeling import MlpMixer ,CONFIGS
        # import configs
        import torch.nn as nn
        import torch

        config = CONFIGS['Mixer-L_16-21k']
        net = MlpMixer(config, 224, num_classes=100, patch_size=16, zero_head=True)
        net.load_state_dict(torch.load('./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin'))
        net.eval()

        return net




    
    elif model_name =='vgg11_cifar10':
        import vgg

        import torchvision.models as mls
        import torch.nn as nn
        import torch
        net = vgg.__dict__['vgg11_bn'](num_classes=10)
        # net = vgg11(num_classes=10)
        # input_size = net.classifier[0].in_features
        # output_size = 10
        # classifier = nn.Sequential(
        #     OrderedDict([
        #         ('fc1', nn.Linear(input_size, input_size // 8)),
        #         ('relu1', nn.ReLU()),
        #         ('droupout', nn.Dropout(p=0.20)),

        #         ('fc2', nn.Linear(input_size // 8, input_size // 32)),
        #         ('relu2', nn.ReLU()),
        #         ('droupout', nn.Dropout(p=0.20)),

        #         ('fc3', nn.Linear(input_size // 32, input_size // 128)),
        #         ('relu3', nn.ReLU()),
        #         ('droupout', nn.Dropout(p=0.20)),

        #         ('fc4', nn.Linear(input_size // 128, output_size)),
        #         # ('softmax', nn.LogSoftmax(dim=1))
        #     ])
        # )
        # net.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        # net.classifier = classifier
        net.load_state_dict(torch.load('./classifier_pretrained_weights/cifar10_vgg_9244.pt'))
        # checkpoint = torch.load('./classifier_pretrained_weights/cifar10_vgg_9244.pt',strict=False)
        # net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        return net
    
    elif model_name =='vgg16_cifar100':
        # from vgg import vgg16_bn
        import vgg
        # import torchvision.models as mls
        import torch.nn as nn
        import torch
        #net = vgg16_bn(num_classes=100)
        net = vgg.__dict__['vgg16_bn'](num_classes=100)
        
        # net = mls.vgg16(pretrained=False)
        # input_size = net.classifier[0].in_features
        # output_size = 100
        # classifier = nn.Sequential(
        #     OrderedDict([
        #         ('fc1', nn.Linear(input_size, input_size // 8)),
        #         ('relu1', nn.ReLU()),
        #         ('droupout', nn.Dropout(p=0.20)),

        #         ('fc2', nn.Linear(input_size // 8, input_size // 32)),
        #         ('relu2', nn.ReLU()),
        #         ('droupout', nn.Dropout(p=0.20)),

        #         ('fc3', nn.Linear(input_size // 32, input_size // 128)),
        #         ('relu3', nn.ReLU()),
        #         ('droupout', nn.Dropout(p=0.20)),

        #         ('fc4', nn.Linear(input_size // 128, output_size))
        #         #('softmax', nn.LogSoftmax(dim=1))
        #     ])
        # )
        # net.classifier = classifier
        # checkpoint = torch.load('./classifier_pretrained_weights/cifar100_vgg16_7375.pt',strict=False)
        net.load_state_dict(torch.load('/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar100_vgg16_7375.pt'),strict=True)
        # net.load_state_dict(checkpoint)
        net.eval()
        return net
    
    elif model_name=='resnet_flower':
        network = models.resnet34(pretrained=False)
        number_of_features = network.fc.in_features
        network.fc = nn.Linear(number_of_features, 102)
        print(network)
        network.load_state_dict(torch.load('./classifier_pretrained_weights/resnet34-333f7ec4.pth'))
    
    elif model_name=="vit_cifar10":
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        from PIL import Image
        import requests

        #url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
        #image = Image.open(requests.get(url, stream=True).raw)
        feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
        
        return feature_extractor, model

    elif model_name=="vit_cifar10_1k":
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        extractor = AutoFeatureExtractor.from_pretrained("verypro/vit-base-patch16-224-cifar10")
        model = AutoModelForImageClassification.from_pretrained("verypro/vit-base-patch16-224-cifar10")

        return extractor,model

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
        # from transformers import ViTFeatureExtractor, ViTForImageClassification
        # from PIL import Image
        # # feature_extractor = ViTFeatureExtractor.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')
        # model = ViTForImageClassification.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')

        model.eval()

        return None, model

    # elif model_name == "vit_b16_cifar100_1k":


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

    elif model_name=="cct_cifar100":
        from Compact_Transformers.src import cct_7_3x1_32_c100
        model = cct_7_3x1_32_c100(pretrained=True, progress=True)
        return model
    
    elif model_name=="cvt_cifar":
        from Compact_Transformers.src import cct_7_3x1_32
        model = cct_7_3x1_32(pretrained=True, progress=True)
        return model

    elif model_name=="cvt_cifar100":
        from Compact_Transformers.src import cct_7_3x1_32_c100
        model = cct_7_3x1_32_c100(pretrained=True, progress=True)
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

        

        model = timm.create_model("hf_hub:nateraw/resnet50-oxford-iiit-pet-v2", pretrained=True)
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

    elif model_name =='densenet201_caltech':
        
        import torch
        from torch.autograd import Variable as V
        import torchvision.models as models
        from torchvision import transforms as trn
        from torch.nn import functional as F
        import os
        from PIL import Image
        net = models.densenet201()
        net.load_state_dict(torch.load('./classifier_fine_tuned_weight/densenet_caltech256_best_acc.pth'),strict=True)
        #net.load_state_dict(torch.load('tiny_resnet34_7356.pt'))
        return net
    
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



    # elif model_name == 'swin-tiny-patch4-window7-224-finetuned-cifar10':

    #     from transformers import AutoFeatureExtractor, AutoModelForImageClassification

    #     feature_extractor = AutoFeatureExtractor.from_pretrained("nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10")
    #     model = AutoModelForImageClassification.from_pretrained("nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10")

    #     model.eval()

    #     return model, feature_extractor

    elif model_name == "resnet50-oxford-iiit-pet":
        import timm
        
        model = timm.create_model("hf_hub:nateraw/resnet50-oxford-iiit-pet", pretrained=True)

        return model

    elif model_name == "vit_stanford-dogs":
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        extractor = AutoFeatureExtractor.from_pretrained("ep44/Stanford_dogs-google_vit_base_patch16_224")
        model = AutoModelForImageClassification.from_pretrained("ep44/Stanford_dogs-google_vit_base_patch16_224")

        model.eval()

        return extractor, model

    elif model_name == "vit_cars196":

        # Load model directly
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained("therealcyberlord/stanford-car-vit-patch16")
        model = AutoModelForImageClassification.from_pretrained("therealcyberlord/stanford-car-vit-patch16")
        model.eval()

        return processor, model

    elif model_name =="vit-L-CIFAR10":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("tzhao3/vit-L-CIFAR10")
        model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-CIFAR10")

        model.eval()

        return feature_extractor, model
    

    elif model_name =="vit-L-CIFAR100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("tzhao3/vit-L-CIFAR100")
        model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-CIFAR100")

        model.eval()

        return feature_extractor, model
    

    elif model_name =="swin-base-finetuned-cifar100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")
        model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")

        model.eval()

        return feature_extractor, model
    
    elif model_name =="swin-tiny-finetuned-cifar100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-tiny-finetuned-cifar100")
        model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-tiny-finetuned-cifar100")

        model.eval()

        return feature_extractor, model
    

    elif model_name =="swin-base-finetuned-cifar10":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("alfredcs/swin-cifar10")
        model = AutoModelForImageClassification.from_pretrained("alfredcs/swin-cifar10")

        model.eval()

        return feature_extractor, model
    

    elif model_name =="swin-small-finetuned-cifar100":
        
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("MazenAmria/swin-small-finetuned-cifar100")
        model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-small-finetuned-cifar100")

        model.eval()

        return feature_extractor, model
    

    elif model_name == "beit-finetuned-cifar10":


        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("jadohu/BEiT-finetuned")
        model = AutoModelForImageClassification.from_pretrained("jadohu/BEiT-finetuned")

        model.eval()

        return feature_extractor, model
    
    elif model_name == "convnext-tiny-finetuned-cifar10":

        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("ahsanjavid/convnext-tiny-finetuned-cifar10")
        model = AutoModelForImageClassification.from_pretrained("ahsanjavid/convnext-tiny-finetuned-cifar10")

        model.eval()

        return feature_extractor, model
    

    elif model_name == 'vit-eurosat':
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        feature_extractor = AutoFeatureExtractor.from_pretrained("internetoftim/dino-vitb16-eurosat")
        model = AutoModelForImageClassification.from_pretrained("internetoftim/dino-vitb16-eurosat")

        return feature_extractor, model
    
    elif model_name == "vit_inat":
        import timm

        #For inaturalist21 dataset; top1 acc 
        model = timm.create_model("hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_inat21", pretrained=True)

        model.eval()

        return model

    elif model_name == "vit_coyo":
        # referred the link below: 
        # https://saturncloud.io/blog/converting-tensorflow-model-to-pytorch-model/
        import torch

        model = torch.jit.load('/home/jihwan/DF_synthesis/pretrained_coyo/last_checkpoint.data-00000-of-00001')

        for param in model.parameters():
            param.requires_grad = False

            if len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        
        model.eval()

        return model

    elif model_name == "vit_b16_flowers":
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        extractor = AutoFeatureExtractor.from_pretrained("dima806/oxford_flowers_image_detection")
        model = AutoModelForImageClassification.from_pretrained("dima806/oxford_flowers_image_detection")

        model.eval()

        return extractor, model
    elif model_name == "deit-cifar10":
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        extractor = AutoFeatureExtractor.from_pretrained("tzhao3/DeiT-CIFAR10")
        model = AutoModelForImageClassification.from_pretrained("tzhao3/DeiT-CIFAR10")

        model.eval()

        return extractor, model
    

    elif model_name == "deit-cifar100":
        # Load model directly
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        extractor = AutoFeatureExtractor.from_pretrained("tzhao3/DeiT-CIFAR100")
        model = AutoModelForImageClassification.from_pretrained("tzhao3/DeiT-CIFAR100")

        model.eval()

        return extractor, model


    elif model_name == "vit-mae-cub":

        from transformers import AutoImageProcessor
        model = CustomViTForImageClassification.from_pretrained("vesteinn/vit-mae-cub")
        image_processor = AutoImageProcessor.from_pretrained("vesteinn/vit-mae-cub")

        model.eval()

        return image_processor, model







       
    else:
        raise ValueError(f"{model_name} is not a supported classifier...")
