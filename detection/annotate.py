import os
import torch
import nvidia_smi
import cv2
from ultralytics import YOLO
import supervision as sv

def annotate_by_inference(weights_path, data_path):
    """
    Runs inference to annotate new images
    """
    test_images_path = os.path.join(data_path, 'test', 'images')
    output_path = os.path.join(os.getcwd(), 'runs/detect/inference')
    os.makedirs(output_path, exist_ok=True)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    test_images = [f for f in Path(test_images_path).glob('*') 
        if f.suffix.lower() in ['.jpg']]

    try:
        model = YOLO(weights_path)
        for img in test_images:
            results = model.predict(source=str(img), conf=0.3, iou=0.8, max_det=300, device=0, save=True, save_txt=True, project=output_path, name='results')[0]
            image = cv2.imread(str(img))

            detections = sv.Detections.from_ultralytics(results)

            # Annotate image
            annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            
            # Save annotated image
            #output_img_path = os.path.join(output_path, 'results', f'annotated_{img_path.name}')
            #cv2.imwrite(os.path.join(output_img_path, 'test', 'annotated'), annotated_image)

            height, width = cv2.imread(image_path).shape[:2]
            label_file = os.path.join(output_dir, 'labels', os.path.splitext(filename)[0] + ".txt")

            with open(label_file, "w") as f:
                for box in results.boxes:
                    cls_id = int(box.cls)
                    x_center, y_center, box_w, box_h = box.xywhn[0]  # normalized xywh
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

if __name__ == "__main__":
    weights_path = '/home/isha/repos/pallets-detection-segmentation/runs/train23/best.pt'
    data_path = '/home/isha/repos/pallets-detection-segmentation/data/Pallets'
    annotate_by_inference(weights_path, data_path)