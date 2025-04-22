import os
import cv2
from ultralytics import YOLO
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.augmentation import ImageAugmentation

PREPROCESS = True

def train(data_yaml_path):
    print("Training in progress...")
    model = YOLO("yolo11n.pt")

    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=420,
        lr0=0.0001,
        lrf=0.1,
#        patience=5,
        save=False
    )



def augment_and_shuffle_data(images_dir, labels_dir, augmentor):
    images = os.listdir(images_dir)
    labels = os.listdir(labels_dir)

    augmented_images = []
    augmented_labels = []

    for img_filename, label_filename in zip(images, labels):
        img_path = os.path.join(images_dir, img_filename)
        label_path = os.path.join(labels_dir, label_filename)
        
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Failed to load image {img_path}, skipping.")
            continue
        
        augmented_image_paths = augmentor.augment(image, img_path, label_path)

        for augmented_image_path in augmented_image_paths:
            augmented_images.append(augmented_image_path)
            augmented_labels.append(label_path)

    augmented_data_pairs = list(zip(augmented_images, augmented_labels))

    random.shuffle(augmented_data_pairs)

    shuffled_augmented_images = [pair[0] for pair in augmented_data_pairs]
    shuffled_augmented_labels = [pair[1] for pair in augmented_data_pairs]

    return shuffled_augmented_images, shuffled_augmented_labels

def main():
    try:
        if(PREPROCESS):
            print("Data preprocessing for object detection training.")

            data_path = "/home/ibhatt/repos/path-robotics/warehouse-detection-segmentation/data/pallets_original_labeled/"
            train_path = os.path.join(data_path, "train")
            val_path = os.path.join(data_path, "valid")
            test_path = os.path.join(data_path, "test")

            augmentor = ImageAugmentation()
            images_dir = os.path.join(train_path, "images")
            labels_dir = os.path.join(train_path, "labels")

            shuffled_images, shuffled_labels = augment_and_shuffle_data(images_dir, labels_dir, augmentor)

            for img_path, label_path in zip(shuffled_images, shuffled_labels):
                print(f"Using augmented image: {img_path} with label: {label_path}")

        data_yaml_path = "/home/ibhatt/repos/path-robotics/warehouse-detection-segmentation/data/pallets_original_labeled/data.yaml"
        train(data_yaml_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
