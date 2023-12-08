import argparse
import logging
import os
import sys


from datasets import load_dataset, load_metric
from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import evaluate
import numpy as np
from transformers import DefaultDataCollator

import tqdm
from modeling import MlpMixer ,CONFIGS
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import timm

from torch.utils.data import DataLoader
from torchvision import transforms

# newly added : refer standard-dogs
from load import load_datasets
from datasets import Dataset


# from datasets import Dataset
from PIL import Image
import os
import scipy.io
from torchvision.transforms import ToTensor




def load_stanford_dogs_dataset(root, train=True, cropped=False):
    images_folder = os.path.join(root, 'Images')
    annotations_folder = os.path.join(root, 'Annotation')

    # 데이터셋 분할 및 라벨 로드
    split_file = 'train_list.mat' if train else 'test_list.mat'
    split_data = scipy.io.loadmat(os.path.join(root, split_file))
    split = [item[0][0] for item in split_data['annotation_list']]
    labels = [item[0] - 1 for item in split_data['labels']]

    # 이미지와 라벨 데이터 준비
    data = []
    for annotation, label in zip(split, labels):
        image_name = annotation + '.jpg'
        image_path = os.path.join(images_folder, image_name)
        if cropped:
            # 크롭된 이미지를 로드하는 로직 (여기서는 생략)
            pass
        else:
            image = Image.open(image_path).convert('RGB')
            image = ToTensor()(image)  # 이미지를 텐서로 변환
            data.append({'image': image, 'label': label})

    return Dataset.from_dict({'image': [d['image'] for d in data], 'label': [d['label'] for d in data]})






class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0, temperature=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature






class ImageDistilTrainer(Trainer):
    def __init__(self, *args,teacher_model=None, **kwargs):
        # if 'model' in kwargs:
        #     raise ValueError("`model` argument cannot be used, use `student_model` instead.")
        # if 'model_init' in kwargs:
        #     raise ValueError("`model_init` argument cannot be used with this Trainer.")
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # self.student = student_model
        self._move_model_to_device(self.teacher, self.model.device)
        # self.teacher.cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss()
        # self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(self.device)
        self.model.to(self.device)
        self.teacher.eval()
        # self.student.to(self.device)
        # self.teacher.eval()
        # self.student.train()
        # self.temperature = temperature
        # self.lambda_param = lambda_param

    def compute_loss(self, model, inputs, return_outputs=False):
        
        print(f"######################################################")
        # print(f"device:{self.device}")
        
        #student_output = self.student(**inputs)
        print("inputs : ",inputs)
        
        #student_output = student_output.cuda()


        outputs_student = model(**inputs)

        # outputs_student = model(inputs['pixel_values'])

        # student_output = self.student(inputs['pixel_values'])
        student_loss = outputs_student.loss

        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            # outputs_teacher = self.teacher(inputs['pixel_values'])
          #teacher_output = self.teacher(**inputs)
          #teacher_output = teacher_output.cuda()

        # Compute soft targets for teacher and student


        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # breakpoint()
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )

        




        # Return weighted student loss
        # loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        loss = self.args.alpha * self.criterion(outputs_student.logits,inputs['labels']) + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss




        # soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)

        # soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # print(f"soft_teacher: { soft_teacher}")
        # print(f"soft_student: { soft_student}")

        # # Compute the loss
        # distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # # Compute the true label loss
        # student_target_loss = student_output.loss

        # # Calculate final loss
        # #loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        # loss =(1. - self.lambda_param)* self.criterion(student_output.logits, inputs['labels']) + self.lambda_param * distillation_loss
        # return (loss, student_output) if return_outputs else loss



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=int, default=2)
    parser.add_argument("--teacher_id", type=str)
    parser.add_argument("--student_id", type=str)
    parser.add_argument("--dataset_id", type=str)
    parser.add_argument("--stanford_dataset_id", type=str)
    # parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--run_hpo", type=bool, default=True)
    parser.add_argument("--n_trials", type=int, default=50)

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default="t_vit_b16_1k_s_beit_b16_1k_kd_cifar10_5seed_temp_2_alpha_0.5_2")

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--hub_token", type=str, default=None)


    args, _ = parser.parse_known_args()


    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    
    dataset = load_dataset("imagefolder", data_dir=args.dataset_id,num_proc=2)

    # We should change here
    # test_dataset = load_dataset("Alanox/stanford-dogs",num_proc=2)
    # _, test_data, _ = load_datasets("stanford_dogs")    
    # test_loader = DataLoader(test_data, batch_size=args.per_device_eval_batch_size, shuffle=True)

    # # DataLoader에서 이미지와 라벨을 추출
    # all_images = []
    # all_labels = []
    # for batch in test_loader:
    #     images, labels = batch
    #     all_images.append(images)
    #     all_labels.append(labels)

    # # 모든 이미지와 라벨을 하나의 텐서로 스택
    # stacked_images = torch.cat(all_images, dim=0)
    # stacked_labels = torch.cat(all_labels, dim=0)

    # # Dataset.from_dict를 사용하여 새로운 데이터셋 생성
    # test_dataset = Dataset.from_dict({
    #     'image': stacked_images,
    #     'label': stacked_labels
    # })
    # 사용 예시
    # root = './data/StanfordDogs'
    # train_dataset = load_stanford_dogs_dataset(root, train=True, cropped=False)
    # test_dataset = load_stanford_dogs_dataset(root, train=False, cropped=False)
    test_dataset = load_dataset("iamgefolder", data_dir = args.stanford_dataset_id,num_proc=2)


    if args.teacher_id == "edumunozsala/vit_base-224-in21k-ft-cifar100":
        teacher_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    else:
        teacher_processor = AutoImageProcessor.from_pretrained(args.teacher_id)

    if args.teacher_id == "edumunozsala/vit_base-224-in21k-ft-cifar100":
        teacher_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    else:    
        teacher_extractor = AutoFeatureExtractor.from_pretrained(args.teacher_id)

    # # Load Datset
    def process(examples):
        processed_inputs = teacher_processor(examples["image"])
        return processed_inputs

    # def cifar_process(examples):
    #     processed_inputs = teacher_processor(examples["image"])
    #     return processed_inputs
    def cifar_process(examples):
        # 각 이미지를 개별적으로 처리
        processed_inputs = teacher_processor(examples["image"])
        # processed_images = [teacher_processor(image) for image in examples["image"]]
        # # 모든 처리된 이미지를 하나의 배치로 결합
        # processed_batch = {'pixel_values': torch.stack([x['pixel_values'] for x in processed_images])}
        return processed_batch

    processed_datasets = dataset.map(process, batched=True)
    processed_datasets = processed_datasets.rename_column("label", "labels")
    # processed_datasets = processed_datasets.remove_columns("name")
    # processed_datasets = processed_datasets.remove_columns("annotations")

    # FOR stanford_dogs -> made comment
    # breakpoint()
    processed_test_datasets = test_dataset.map(process, batched=True)
    processed_datasets = processed_datasets.rename_column("label", "labels")
    
    
    # processed_test_datasets = processed_test_datasets.remove_columns("name")
    # processed_test_datasets = processed_test_datasets.remove_columns("annotations")    
    # processed_test_datasets = processed_test_datasets.rename_column("target", "labels")

    



    # breakpoint()
    # breakpoint()
    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        # breakpoint()
        predictions, labels = eval_pred
        acc = accuracy_metric.compute(references=labels, predictions=np.argmax(predictions, axis=1))
        return {
            "accuracy": acc["accuracy"],
            "eval_accuracy": acc["accuracy"],
        }
    

    print(processed_datasets["train"].features)
    

    labels = processed_datasets["train"].features["labels"].names
    
    num_labels = len(labels)

    print(labels)

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label


    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
        learning_rate=float(args.learning_rate),
        seed=33,
        # logging & evaluation strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="epoch",  # to get more information to TB
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=args.push_to_hub,
        hub_strategy="every_save",
        # hub_model_id=args.hub_model_id,
        # hub_token=args.hub_token,
        # distilation parameters
        alpha=args.alpha,
        temperature=args.temperature,
    )


    # if args.

    # initialize models

    # if args.teacher_id == 'edadaltocg/vit_base_patch16_224_in21k_ft_cifar100':
    #     teacher_model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",
    #     pretrained=False)
    #     teacher_model.head = nn.Linear(model.head.in_features, 100)
    #     teacher_model.load_state_dict(
    #         torch.hub.load_state_dict_from_url(
    #             "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
    #             map_location="cpu",
    #             file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
    #         )
    #     )

    
    

    teacher_model = AutoModelForImageClassification.from_pretrained(
        args.teacher_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    )
        
   
    # teacher_model.cuda()
        # net.eval()

        # return net

    student_model = AutoModelForImageClassification.from_pretrained(
        args.student_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    )

    student_model.cuda()


    data_collator = DefaultDataCollator()

    # def student_init():
    #     return  AutoModelForImageClassification.from_pretrained(
    #     "google/vit-base-patch16-224",
    #     num_labels=num_labels,
    #     ignore_mismatched_sizes=True,
    #     id2label=id2label,
    #     label2id=label2id
    # )

    # student_model = student_init()
    trainer = ImageDistilTrainer(
        # model_init=student_init,
        model = student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_test_datasets["train"],
        data_collator=data_collator,
        tokenizer=teacher_extractor,
        compute_metrics=compute_metrics,
    )
    

    if 0:
        # to avoind unnecessary pushes of bad models
        training_args.push_to_hub = False
        training_args.output_dir = "./tmp_deit_tiny/hpo"
        training_args.logging_dir = "./tmp_deit_tiny/hpo/logs"

        # hpo space which replace the training_args
        def hp_space(trial):
            return {
                "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "alpha": trial.suggest_float("alpha", 0, 1),
                "temperature": trial.suggest_int("temperature", 2, 30),
            }

        best_run = trainer.hyperparameter_search(n_trials=args.n_trials, direction="maximize", hp_space=hp_space)

        # print best run
        print(best_run)
        # overwrite initial hyperparameters with from the best_run
        for k, v in best_run.hyperparameters.items():
            setattr(training_args, k, v)

        training_args.push_to_hub = args.push_to_hub
        training_args.output_dir = args.output_dir
        training_args.logging_dir = f"{args.output_dir}/logs"
    
    


    trainer.train()




    # trainer.evaluate(processed_test_datasets)
    # trainer.evaluate(test_dataset)

    trainer.save_model(args.output_dir)









# repo_name = "deit_b16_kd_214_cifar10_please"


# training_args = TrainingArguments(
#     output_dir="deit_b16_kd_214_cifar10_please",
#     per_device_train_batch_size=128,  # Set the train batch size
#     gradient_accumulation_steps=1,
#     per_device_eval_batch_size=128,   # Set the evaluation batch size
#     num_train_epochs=30,
#     fp16=True,
#     logging_dir=f"{repo_name}/logs",
#     logging_strategy="epoch",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     report_to="tensorboard",
#     # Set push_to_hub to False to not push your model to the Hugging Face Hub
#     push_to_hub=False,
#     local_rank=int(os.environ.get("LOCAL_RANK", "-1"))  # Add this line for distributed training
# )



# num_labels = len(processed_datasets["train"].features["label"].names)

# initialize models
# teacher_model = AutoModelForImageClassification.from_pretrained(
#     "tzhao3/DeiT-CIFAR10",
#     num_labels=num_labels,
#     ignore_mismatched_sizes=True
# ).cuda()

# training MobileNetV2 from scratch

# Load model directly


# extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")
# model = 

# student_model = AutoModelForImageClassification.from_pretrained(
#     "facebook/deit-tiny-patch16-224",
#     num_labels=num_labels,
#     ignore_mismatched_sizes=True    
# ).cuda()


# print(student_model)

# student_model.classifier = nn.Linear(student_model.classifier.in_features, num_labels).cuda()
# student_model.num_labels = num_labels

#accuracy = evaluate.load("accuracy")

# print(processed_test_datasets)


