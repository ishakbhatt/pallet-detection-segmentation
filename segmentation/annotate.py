import os
import cv2
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def save_yolo_segmentation_labels(results, image_shape, label_file_path):
    """
    Save the labels to txt files. 
     - goes through the ground class id and polygon points from results
     - normalizes points (agnostic to different image sizes/resolutions)
     - puts them in one list so it's a polygon
     - writes that line to the txt
    """
    height, width = image_shape[:2]

    with open(label_file_path, "w") as f:
        for cls_id, polygon in zip(results.boxes.cls, results.masks.xy):
            cls_id = int(cls_id)
            normalized_poly = []

            for x, y in polygon:
                x_norm = x / width
                y_norm = y / height
                normalized_poly.extend([f"{x_norm:.16f}", f"{y_norm:.16f}"])

            line = f"{cls_id} " + " ".join(normalized_poly)
            f.write(line + "\n")

def annotate_by_inference(weights_path, data_path):
    """
    Runs instance segmentation inference and saves YOLOv11-style polygon labels.
    """
    test_images_path = '/home/isha/repos/pallet-detection-segmentation/data/unannotated/images'
    labels_output_dir = os.path.join(data_path, 'unannotated', 'masks')
    os.makedirs(labels_output_dir, exist_ok=True)

    model = YOLO(weights_path)
    test_images = sorted(os.listdir(test_images_path))

    for img_name in tqdm(test_images, desc="Annotating images"):
        # perform inference and then save the results as a txt for the label
        img_path = os.path.join(test_images_path, img_name)
        results = model.predict(source=img_path, conf=0.3, iou=0.8, max_det=300, device='cpu')[0]
        image = cv2.imread(img_path)
        label_file = os.path.join(labels_output_dir, Path(img_name).stem + ".txt")
        if results.masks is not None:
            save_yolo_segmentation_labels(results, image.shape, label_file)
        else:
            label_file = os.path.join(labels_output_dir, Path(img_name).stem + ".txt")
            open(label_file, 'w').close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Annotate images for ground segmentation based on a fine-tuned model.")
    
    parser.add_argument(
        '--weights_path',
        type=str,
        default='/home/isha/repos/pallet-detection-segmentation/models/segmentation/initial_best.pt',
        help='Path to initially trained segmentation model.'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/isha/repos/pallet-detection-segmentation/data',
        help='Path to all of the data.'
    )

    annotate_by_inference(parser.weights_path, data_path)

    os.makedirs(os.path.join(data_path, 'unannotated', 'images'))
    shutil.move(os.path.join(data_path, 'unannotated', '*.jpg'), os.path.join(data_path, 'unannotated', 'images'))

    # Now you can move this folder with annotated data wherever you want