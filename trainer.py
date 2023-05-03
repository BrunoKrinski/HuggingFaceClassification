import cv2
import argparse
import evaluate
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import DefaultDataCollator
from albumentations.pytorch import ToTensorV2
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_dir', type=str)
    return parser.parse_args()

def main():
    args = get_args()
        
    dataset = load_dataset("imagefolder", data_dir=f"data/{args.dataset}")
    print(dataset)
        
    label2id, id2label = dict(), dict()
    if len(dataset["train"].features) > 1:
        labels = dataset["train"].features["label"].names
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
    else:
        print("Dataset with just one label. Please use at least two labels!")
        return
            
    checkpoint = "google/vit-base-patch16-224-in21k"
    #checkpoint = "google/mobilenet_v2_1.0_224"
    #checkpoint = "microsoft/resnet-50"
    #checkpoint = "facebook/regnet-y-040"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
            
    size = (image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else image_processor.size["height"]
    )
    
    _train_transforms = albu.Compose([
        albu.LongestMaxSize(max_size=size),
        albu.PadIfNeeded(min_height=size, 
                         min_width=size,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0, 
                         mask_value=0),
        albu.Normalize(mean=image_processor.image_mean, 
                       std=image_processor.image_std),
        ToTensorV2()
    ])
    
    _valid_transforms = albu.Compose([
        albu.LongestMaxSize(max_size=size),
        albu.PadIfNeeded(min_height=size, 
                         min_width=size,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0, 
                         mask_value=0),
        albu.Normalize(mean=image_processor.image_mean, 
                       std=image_processor.image_std),
        ToTensorV2()
    ])
    
    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(image=np.asarray(img))["image"] for img in examples["image"]]
        del examples["image"]
        return examples
    
    def valid_transforms(examples):
        examples["pixel_values"] = [_valid_transforms(image=np.asarray(img))["image"] for img in examples["image"]]
        del examples["image"]
        return examples
    
    dataset["train"] = dataset["train"].with_transform(train_transforms)
    dataset["test"] = dataset["test"].with_transform(valid_transforms)
        
    data_collator = DefaultDataCollator()
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  
    )
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
if __name__ == '__main__':
    main()
