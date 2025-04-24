import os
import torch
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.augmentation import ImageAugmentation
from ultralytics import YOLO
import supervision as sv
torch.cuda.empty_cache()

PREPROCESS = True

def train(data_yaml_path):
    print("Training in progress...")
    
    model = YOLO("yolo11s-seg.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=960,
        lr0=0.0001,
        lrf=0.1,
        weight_decay=0.0005,
        patience=7,
        save_period=-1,
        val=True,
        batch=8,
        save=True
    )

    metrics = model.val()

def main():
    try:
        if(PREPROCESS):
            print("Data preprocessing for object detection training.")

            data_path = "/home/isha/repos/pallet-detection-segmentation/data/ground_segmentation/"
            train_path = os.path.join(data_path, "train")
            val_path = os.path.join(data_path, "valid")
            test_path = os.path.join(data_path, "test")

            augmentor = ImageAugmentation()
            images_dir = os.path.join(train_path, "images")
            labels_dir = os.path.join(train_path, "labels")

            shuffled_images, shuffled_labels = augmentor.augment_and_shuffle_data(images_dir, labels_dir)

            for img_path, label_path in zip(shuffled_images, shuffled_labels):
                print(f"Using augmented image: {img_path} with label: {label_path}")

        data_yaml_path = "/home/isha/repos/pallet-detection-segmentation/data/ground_segmentation/data.yaml"
        train(data_yaml_path)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
