#!/bin/bash


teacher_res="resnet34"
student_res="resnet18"

teacher="mlp_mixer_l16_1k"
student="mlp_mixer_b16_1k"

#python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ../old_DF_synthesis/Ablation/Fake_ours_resnet34_seed161_thres0.98_cifar10_iter5_TV --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt
#python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_100seed_th094_ours_resnet34_cifar10 --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt

# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir cifar100_inversion/dataset/denorm/final --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_NI_denorm_cifar10Official_norm.txt

#python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_diversity_5seed_th098_ours_resnet34_cifar10 --teacher_weights ./cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/divloss_5seed_CI_"$dataset"_"$teacher"_"$student"_denorm.txt


#python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ../old_DF_synthesis/Fake_ours_resnet34_threshold0.94_iter5_cifar100 --teacher_weights ./cifar100_resnet34_7802.pth --csv_name CI_cifar100_"$teacher"_"$student"_nodenorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/CI_"$dataset"_"$teacher"_"$student"_nodenorm.txt

#python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ./DI_cifar100_nodenorm_7802_50k --teacher_weights ./cifar100_resnet34_7802.pth --csv_name cifar100_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt

#python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ./NI_cifar100_nodenorm_7802_50k --teacher_weights ./cifar100_resnet34_7802.pth --csv_name cifar100_"$teacher"_"$student"_denorm --t_arch "$teacher" --s_arch "$student" --epochs 400 | tee KD_result/"$dataset"_"$teacher"_"$student"_denorm.txt



# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_ours_zo2m_mlp_mixer_cifar10_L_1k_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_1113 --teacher_weights ./classifier_pretrained_weights/Mixer-L_16_1k_9698_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_cos --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt



# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir Fake_ours_zo2m_mlp_mixer_cifar10_L_21k_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_1113 --teacher_weights ./classifier_pretrained_weights/Mixer-L16_21k_9844_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_cos --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt



# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir mlp_mixer_cifar10_L_1k_0 --teacher_weights ./classifier_pretrained_weights/Mixer-L_16_1k_9698_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_pii --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt


# python kd_v3.py --ngpu "$1" --dataset cifar100 --dir Fake_ours_zo2m_mlp_mixer_cifar100_L_1k_thres_0.94_lr_0.01_tv_1.3e_3_5seed --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_1k_8569_checkpoint.bin --csv_name cifar100_"$teacher"_"$student" --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt


# python kd_v3.py --ngpu "$1" --dataset cifar100 --dir Fake_ours_zo2m_mlp_mixer_cifar100_L_21k_thres_0.94_lr_0.01_tv_1.3e_3_5seed --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin --csv_name cifar100_"$teacher"_"$student" --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt

# Dataload : 224*224
# python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_10seed_th098_resnet34_cifar10_224_nearest --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_224_224 --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 


# python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_1seed_th098_resnet34_cifar10_T_nearest --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_nearest --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 
# python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_1seed_th098_resnet34_cifar10_T_bilinear --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_bilinear --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 
# python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_1seed_th098_resnet34_cifar10_T_bicubic --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_bicubic --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 
# python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_1seed_th098_resnet34_cifar10_T_area --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_area --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 
python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_1seed_th098_resnet34_cifar10_T_nearest-exact --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_nearest_exact --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 
# python kd_v3_resnet_2.py --ngpu "$1" --batch_size 64 --dataset cifar10 --dir ./Fake_90over_10seed_th098_resnet34_cifar10_224_nearest --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet_224_224 --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt 









# Dataload : 32*32
# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ./Fake_90over_10seed_th098_resnet34_cifar10_224_nearest --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt

# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ./DeepInversion/cifar10/runs/data_generation/try1/final_images --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher_res"_"$student_res"_resnet --t_arch "$teacher_res" --s_arch "$student_res" --epochs 400 | tee "$dataset"_"$teacher_res"_"$student_res"_ours_denorm_cifar10Official_norm.txt





# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ./DeepInversion/cifar10/runs/data_generation/try1/final_images --teacher_weights ./classifier_pretrained_weights/cifar10_resnet34_9557.pt --csv_name cifar10_"$teacher"_"$student"_itp_test --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt
# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ./Fake_denormours_zo2m_mlp_mixer_cifar10_L_21k_mean_thres_0.98_lr_0.01_tv_1.3e_3_1seedarea --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_area --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt
# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ./Fake_denormours_zo2m_mlp_mixer_cifar10_L_21k_mean_thres_0.98_lr_0.01_tv_1.3e_3_1seedbicubic --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_bicubic --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt
# python kd_v3.py --ngpu "$1" --dataset cifar10 --dir ./Fake_denormours_zo2m_mlp_mixer_cifar10_L_21k_mean_thres_0.98_lr_0.01_tv_1.3e_3_1seedbilinear --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin --csv_name cifar10_"$teacher"_"$student"_bilinear --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt
# python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ./mlp_mixer_32size_cifar100_L_21k_0 --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin --csv_name cifar100_"$teacher"_"$student"_pii --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt


# python kd_v3.py --ngpu "$1" --dataset cifar100 --dir ./Fake_85over_10seed_th094_mlp_mixer_cifar100_L_1knearest --teacher_weights ./classifier_pretrained_weights/Mixer_L16_cifar100_21k_9125_checkpoint.bin --csv_name cifar100_"$teacher"_"$student"_ours_1127_3 --t_arch "$teacher" --s_arch "$student" --epochs 30 | tee "$dataset"_"$teacher"_"$student"_ours_denorm_cifar10Official_norm.txt