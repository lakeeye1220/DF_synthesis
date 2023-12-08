#!/bin/bash


# CUDA_VISIBLE_DEVICES=0,1,3 python3 \
#     vit_kd_yj_2.py \
# 	--teacher_id "nateraw/vit-base-patch16-224-cifar10" --student_id "facebook/deit-tiny-patch16-224" --dataset_id Fake_ours_vit_base_patch16_224_cifar10_thres0.98_iter5_T \

# CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc-per-node=3 \
#     vit_kd_yj_2.py \
# 	--teacher_id "verypro/vit-base-patch16-224-cifar10" --student_id "microsoft/beit-base-patch16-224" --dataset_id Fake_ours_zo2m_vit_b16_cifar10_1k_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_1113 \



# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
#     vit_kd_yj_2.py \
# 	--teacher_id "nateraw/vit-base-patch16-224-cifar10" --student_id "facebook/deit-tiny-patch16-224" --dataset_id Fake_ours_zo2m_vit_b16_21k_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_1117 \
#     --output_dir "t_vit_b16_21k_s_deit_t16_1k_cifar10_2" --per_device_train_batch_size 128 --per_device_eval_batch_size 128



# CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc-per-node 3 \
#     vit_kd_yj_2.py \
# 	--teacher_id "nateraw/vit-base-patch16-224-cifar10" --student_id "facebook/deit-tiny-patch16-224" --dataset_id cifar10_vit-B_0 \
#     --output_dir "t_vit_b16_21k_s_deit_t16_1k_cifar10_pii" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \


# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc-per-node 4 \
#     vit_kd_yj_2.py \
# 	--teacher_id "tzhao3/vit-L-CIFAR10" --student_id "google/vit-base-patch16-224" --dataset_id cvpr_cifar10_vit_large_1 \
#     --output_dir "t_vit_l16_1k_s_vit_b16_1k_cifar10_pii_2" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc-per-node 4 \
#     vit_kd_yj_2.py \
# 	--teacher_id "verypro/vit-base-patch16-224-cifar10" --student_id "microsoft/beit-base-patch16-224" --dataset_id vit_base16_cifar10_1k_0 \
#     --output_dir "t_vit_b16_1k_s_beit_b16_1k_cifar10_pii" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \


# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc-per-node 4 \
#     vit_kd_yj_2.py \
# 	--teacher_id "verypro/vit-base-patch16-224-cifar10" --student_id "microsoft/beit-base-patch16-224" --dataset_id vit_base16_cifar10_1k_0 \
#     --output_dir "t_vit_b16_1k_s_beit_b16_1k_cifar10_pii" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \
#     vit_kd_yj_2.py \
# 	--teacher_id "tzhao3/vit-L-CIFAR100" --student_id "google/vit-base-patch16-224-in21k" --dataset_id Fake_ours_vit_large_patch16_224_cifar100_thres0.94_iter5_T \
#     --output_dir "t_vit_l16_21k_s_vit_b16_21k_cifar100" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
#     vit_kd_yj_2.py \
# 	--teacher_id "tzhao3/DeiT-CIFAR100" --student_id "facebook/deit-tiny-patch16-224" --dataset_id Fake_85over_10seed_th094_deit-cifar100nearest \
#     --output_dir "t_deit_b16_21k_s_deit_t16_1k_cifar100_1" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \
#     vit_kd_yj_2.py \
# 	--teacher_id "tzhao3/DeiT-CIFAR100" --student_id "facebook/deit-tiny-patch16-224" --dataset_id cifar100_deit-B_0 \
#     --output_dir "t_deit_b16_1k_s_deit_t16_1k_cifar100_pii_2" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \


# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
#     vit_kd_yj_2.py \
# 	--teacher_id "tzhao3/vit-L-CIFAR100" --student_id "google/vit-base-patch16-224-in21k" --dataset_id Fake_85over_10seed_th094_vit-L-CIFAR100nearest \
#     --output_dir "t_vit_l16_21k_s_vit_b16_21k_cifar100_ours" --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \


# Fake_85over_10seed_th094_vit_cifar100nearest


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \
#     vit_kd_yj_stanford_dogs.py \
# 	--teacher_id "ep44/Stanford_dogs-google_vit_base_patch16_224" --student_id "facebook/deit-tiny-patch16-224" --dataset_id stanford-dogs_1 \
#     --output_dir "t_vit_b16_1k_s_deit_t16_1k_stanforddogs_pii" --per_device_train_batch_size 100 --per_device_eval_batch_size 100 \


CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 \
    vit_kd_yj_stanford_dogs.py \
	--teacher_id "ep44/Stanford_dogs-google_vit_base_patch16_224" --student_id "facebook/deit-tiny-patch16-224" --dataset_id Fake_90over_10seed_th094_vit_stanford-dogs_nearest \
    --stanford_dataset_id "./data/StanfordDogs/Images/" --output_dir "t_vit_b16_1k_s_deit_t16_1k_stanforddogs_ours" --per_device_train_batch_size 100 --per_device_eval_batch_size 100 \