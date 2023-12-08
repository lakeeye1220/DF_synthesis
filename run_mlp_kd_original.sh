#!/bin/bash


teacher="mlp_mixer_l16_21k"
student="mlp_mixer_b16_21k"


#python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ../old_DF_synthesis/Ablation/Fake_ours_resnet34_seed161_thres0.98_cifar10_iter5_TV --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt
#python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_100seed_th094_ours_resnet34_cifar10 --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt

# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir cifar100_inversion/dataset/denorm/final --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_NI_denorm_cifar10Official_norm.txt

#python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_diversity_5seed_th098_ours_resnet34_cifar10 --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/divloss_5seed_CI_"$dataset"_"$teacher"_"$student"_denorm.txt


#python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ../old_DF_synthesis/Fake_ours_resnet34_threshold0.94_iter5_cifar100 --teacher_weights ./cifar100_resnet34_7802.pth --csv_name CI_cifar100_"$teacher"_"$student"_nodenorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/CI_"$dataset"_"$teacher"_"$student"_nodenorm.txt

#python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ./DI_cifar100_nodenorm_7802_50k --teacher_weights ./cifar100_resnet34_7802.pth --csv_name cifar100_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt

#python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ./NI_cifar100_nodenorm_7802_50k --teacher_weights ./cifar100_resnet34_7802.pth --csv_name cifar100_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt



# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_ours_zo2m_mlp_mixer_cifar10_L_1k_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_1113 --teacher_weights ./classifier_pretrained_weights/Mixer-L_16_1k_9698_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_norm --t_arch "$teacher" --s_arch "$student" --epochs 200 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt



python kd_v3_original.py --ngpu "$1" --dataset 'cifar100_ori' --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt