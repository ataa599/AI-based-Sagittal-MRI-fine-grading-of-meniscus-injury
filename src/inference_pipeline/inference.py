import os
import json
from typing import List, Dict, Optional
import torch
import pydicom
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.deep_learning_architecture.training_utils.model import DenseNetSagittalModel
from ultralytics import YOLO
from tqdm import tqdm
from src.constants.constants import yolo_model


# Default constants
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_REGIONS = ["anterior horn", "body", "posterior horn"]

# Image transform used before feeding the classifier
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class InferenceEngine:
    """Modular inference engine for ROI detection (YOLO) + severity classification.

    Usage: create engine = InferenceEngine(model_path, yolo_path, device)
    then call engine.infer_folder(folder_path) to get results list.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.yolo_model_path = yolo_model
        self.device = DEFAULT_DEVICE
        self.regions = DEFAULT_REGIONS
        self.transform = DEFAULT_TRANSFORM

        # maps region -> position index expected by model
        # this map may need to be adapted for your model's training order
        self.region_position_map = {"posterior horn": 0, "anterior horn": 1, "body": 2}

        # load models
        self.model = self._load_model(self.model_path)
        self.yolo = self._load_yolo(self.yolo_model_path) if self.yolo_model_path else None

    def _load_model(self, model_path: str):
        model = DenseNetSagittalModel(num_classes=4)
        map_loc = torch.device(self.device)
        state = torch.load(model_path, map_location=map_loc)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _load_yolo(self, yolo_path: str):
        return YOLO(yolo_path)

    def _load_image(self, path: str) -> Optional[Image.Image]:
        try:
            if path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array
                img = Image.fromarray(arr).convert('RGB')
                return img
            else:
                return Image.open(path).convert('RGB')
        except Exception:
            return None

    def _detect_and_crop(self, pil_img: Image.Image) -> Optional[Image.Image]:
        # if YOLO present, run detection; otherwise return original
        if self.yolo is None:
            return pil_img
        try:
            results = self.yolo(pil_img)
            boxes = results[0].boxes
            if len(boxes) == 0:
                return pil_img
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
            return pil_img.crop((x1, y1, x2, y2))
        except Exception:
            return pil_img

    def _predict_for_position(self, img_tensor: torch.Tensor, position: int) -> Dict:
        """Run model on single image tensor for a specific region position.

        Returns dict with predicted class (int) and confidence (float).
        """
        img = img_tensor.unsqueeze(0).to(self.device)
        pos_tensor = torch.full((1,), position, dtype=torch.long, device=self.device)
        batch = {'image': img, 'position': pos_tensor}
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs['damage_logits'], dim=1)
            conf, pred = probs.max(dim=1)
        return {'pred_class': int(pred.item()), 'confidence': float(conf.item()), 'probs': probs.squeeze(0).cpu().tolist()}

    def infer_image(self, img_path: str) -> Optional[Dict]:
        """Infer a single image: detect ROI, then run classifier for each region and pick best region by confidence.

        Returns dict: {path, region, predicted_severity, confidence, probs}
        """
        pil = self._load_image(img_path)
        if pil is None:
            return None
        cropped = self._detect_and_crop(pil)
        img_t = self.transform(cropped)

        best = None
        for region in self.regions:
            pos = self.region_position_map.get(region, 0)
            res = self._predict_for_position(img_t, pos)
            if best is None or res['confidence'] > best['confidence']:
                best = {**res, 'region': region}

        if best is None:
            return None

        return {
            'image_path': img_path,
            'region': best['region'],
            'predicted_severity': best['pred_class'],
            'confidence': round(best['confidence'], 4)
        }

    def infer_folder(self, folder_path: str, extensions: List[str] = None, best_per_region: bool = True, top_k: int = 1) -> object:
        """Infer all images in a folder and return list of results.

        If best_per_region is True, returns a dict mapping each region to the top-k image(s) by confidence.
        Otherwise returns the full flat list of per-image results.
        """
        if extensions is None:
            extensions = ['.dcm', '.png', '.jpg', '.jpeg', '.tiff']
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in extensions]

        results = []
        for f in tqdm(files, desc='Inferring images'):
            try:
                r = self.infer_image(f)
                if r:
                    results.append(r)
            except Exception:
                continue

        posterior_horn = 0
        anterior_horn = 0
        body = 0
        posterior_horn_image = None
        anterior_horn_image = None
        body_image = None
        for result in results:
            if result["region"] == "posterior horn":
                # posterior_horn.append(result)
                if result["confidence"] > posterior_horn:
                    posterior_horn = result["confidence"]
                    posterior_horn_image = result
            elif result["region"] == "anterior horn":
                if result["confidence"] > anterior_horn:
                    anterior_horn = result["confidence"]
                    anterior_horn_image = result
            elif result["region"] == "body":
                if result["confidence"] > body:
                    body = result["confidence"]
                    body_image = result

        return posterior_horn_image, anterior_horn_image, body_image

        



