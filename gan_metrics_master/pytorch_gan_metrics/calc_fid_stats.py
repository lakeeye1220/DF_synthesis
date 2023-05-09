"""Calculate statistics for FID and save to a file."""

import argparse
import os

from utils import calc_and_save_stats, class_wise_calc_and_save_stats
import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "A handy cli tool to calculate FID statistics.")
    parser.add_argument("--data_path", type=str, default='../../',
                        help='path to image directory (include subfolders)')
    parser.add_argument("--output", type=str, required=True,
                        help="output path")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size")
    parser.add_argument("--img_size", type=int, default=None,
                        help="resize image to this size")
    parser.add_argument('--use_torch', action='store_true',
                        help='using pytorch as the matrix operations backend')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="dataloader workers")

    args = parser.parse_args()

    """calc_and_save_stats(
        args.path,
        args.output,
        args.batch_size,
        args.img_size,
        args.use_torch,
        args.num_workers)"""
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    class_wise_calc_and_save_stats(dataset, args.output, args.batch_size, args.use_torch)
