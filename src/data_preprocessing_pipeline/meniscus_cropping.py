import os
import cv2
from src.constants.constants import ARTIFACT_DIR, cropped_artifact_dataset, yolo_model, conf_threshold, resize_dim
from pathlib import Path
from src.logging_and_exception.exception import CustomException
import sys
from ultralytics import YOLO

class CroppingMeniscusConfig:
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path
        # Create artifact directory and cropped output directory using os.path.join
        self.artifact_dir = os.path.join(ARTIFACT_DIR)
        self.cropped_output_dir = os.path.join(self.artifact_dir, cropped_artifact_dataset)

        # Ensure directories exist
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.cropped_output_dir, exist_ok=True)

class CroppingMeniscus:
    def __init__(self, config: CroppingMeniscusConfig):
        self.input_dir = config.input_image_path
        self.output_dir = config.cropped_output_dir

    def iniate_cropping(self):
        try:
            # === LOAD YOLO MODEL ===
            model = YOLO(yolo_model)

            # === PROCESS EACH IMAGE ===
            for file in os.listdir(self.input_dir):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(self.input_dir, file)
                output_path = os.path.join(self.output_dir, file)

                img = cv2.imread(img_path)

                results = model(img, conf=conf_threshold)[0]
                boxes = results.boxes

                if boxes and len(boxes) > 0:
                    # Take the most confident detection
                    best_box = boxes[0]
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    cropped = img[y1:y2, x1:x2]

                    # Resize only detected ROI
                    resized = cv2.resize(cropped, resize_dim)
                    cv2.imwrite(output_path, resized)
                    print(f"Saved cropped and resized ROI: {file}")
                else:
                    # No detection → copy original image
                    # shutil.copy(img_path, output_path)
                    print(f"No detection: {file}")

            return self.output_dir
        except Exception as e:
            raise CustomException(e, sys)