import argparse
import datetime
import logging
import os.path
import sys
import time

# import common
# import cv2
import numpy as np
import torch
import torch.nn as nn
# from datasets import BrownDataset, HPatches
# from desc_eval import DescriptorEvaluator, GenericLearnedDescriptorExtractor
# from modules import DynamicSoftMarginLoss, HardNetLoss, L2Net
import misc
from resnet_cifar3_cifar10 import ResNet34
import torchvision.transforms as transforms
import torchvision.datasets as datasets
logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()


def add_arg(*args, **kwargs):
    kwargs["help"] = "(default: %(default)s)"
    if not kwargs.get("type", bool) == bool:
        kwargs["metavar"] = ""
    parser.add_argument(*args, **kwargs)


add_arg("--cpuonly", action="store_true")
add_arg("--bs", type=int, default=1024)
add_arg("--model_dir", type=str, default=None)
add_arg("--binary", action="store_true")
add_arg("--test_data", nargs="+", type=str, default="CIFAR10")
add_arg("--patch_size", type=int, default=32)

args = parser.parse_args()

# select device
device = "cpu" if args.cpuonly else "cuda"

# set up the model
model = ResNet34(num_classes=10)
# model = ResNet34(num_classes=100)
# assert args.model_dir is not None, "model directory not specified"\
# for cifar10
model.load_state_dict(torch.load("/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar10_resnet34_9557.pt"))
# for cifar100
# model.load_state_dict(torch.load("/home/jihwan/DF_synthesis/classifier_pretrained_weights/cifar100_resnet34_7802.pth"))
model = model.to(device)

# mean_std = torch.load(os.path.join(args.model_dir, "mean_std.pt"))
# tforms = common.get_basic_input_transform(
#     args.patch_size, mean_std["mean"], mean_std["std"]
# )

tforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# desc_extractor = GenericLearnedDescriptorExtractor(
#     patch_size=args.patch_size,
#     model=model,
#     batch_size=args.bs,
#     transform=tforms,
#     device=device,
# )



evaluators = {}
for dset_name in args.test_data:
    # dset_name, seq_name = dset_dot_seq.split(".")
    if dset_name == "CIFAR10":
        # dset = BrownDataset(
        #     root="./data/brown",
        #     name=seq_name,
        #     download=True,
        #     train=False,
        #     transform=tforms,
        #     data_aug=False,
        # )
        dset = datasets.ImageFolder(root='/home/jihwan/DF_synthesis/ours_beit_threshold0.94_seed300_cifar10', transform=tforms)
    elif dset_name == "CIFAR100":
        raise RuntimeError("please use the offical HPatches evaluation scripts")
    else:
        raise ValueError("dataset not recognized")

    logger.info(f"adding evaluator {dset_name}")
    evaluators[dset_name] = DescriptorEvaluator(
        extractor=desc_extractor,
        datasets=dset,
        batch_size=args.bs,
        binarize=args.binary,
    )


def test():
    logger.info("running evaluation...")
    fpr95 = {}
    for dset_dot_seq, evaluator in evaluators.items():
        evaluator.run()
        fpr95[dset_dot_seq] = evaluator.computeFPR95()
    return fpr95


test_result = test()
for dset_dot_seq, fpr95 in test_result.items():
    logger.info(f"FPR95-{dset_dot_seq} = {fpr95 * 100}%")
