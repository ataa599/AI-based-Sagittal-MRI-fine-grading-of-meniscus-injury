from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import shutil, zipfile, os, base64
from src.inference_pipeline.inference import InferenceEngine
from typing import Dict, Any
import pydicom
from PIL import Image
from io import BytesIO
import numpy as np
from pipelines.preprocess_pipeline import PreprocessPipeline
from pipelines.training_pipeline import TrainingPipeline


app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


def enrich_with_image_and_metadata(image_info: Dict[str, Any]):
        dicom_path = image_info.get('image_path')
        image_base64 = convert_dicom_to_png(dicom_path)
        return {
            'region': image_info.get('region'),
            'predicted_severity': image_info.get('predicted_severity'),
            'confidence': image_info.get('confidence'),
            'image_base64': image_base64,
        }

def convert_dicom_to_png(dcm_path) -> 'Image.Image':
    """
    Convert a DICOM file into an 8-bit grayscale PNG.
    Uses safe min-max scaling (deterministic).
    """
    ds = pydicom.dcmread(str(dcm_path))      # read DICOM file
    arr = ds.pixel_array.astype(np.float32)  # convert to float for scaling

    # Min-max scaling into [0, 255]
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax == vmin:
        # Handle uniform images (avoid divide-by-zero)
        img_scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        img_scaled = np.clip(
            (arr - vmin) * (255.0 / (vmax - vmin)), 0, 255
        ).astype(np.uint8)

    img = Image.fromarray(img_scaled)

    buf = BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})

@app.get("/train/")
async def train_model():
    try:
        preprocessing_pipeline = PreprocessPipeline()
        augmented_images, augmented_csv, test_out, test_csv = preprocessing_pipeline.start_preprocessing_pipeline()
        
        training_pipeline = TrainingPipeline(augmented_images, test_out, augmented_csv, test_csv)
        training_pipeline.start_training_pipeline()
        
        return {"status": "success", "message": "Training completed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@app.post("/infer-folder/")
async def infer_folder(zip_file: UploadFile = File(...)):
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, zip_file.filename)
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    # Determine the extracted folder safely. The upload zip itself is in
    # temp_dir, so skip it when searching for the extracted directory.
    entries = [e for e in os.listdir(temp_dir) if e != os.path.basename(zip_path)]
    extracted_folder = None
    for e in entries:
        full = os.path.join(temp_dir, e)
        if os.path.isdir(full):
            extracted_folder = full
            break
    if extracted_folder is None:
        # try infering top-level folder name from Zip namelist
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            namelist = zip_ref.namelist()
        if namelist:
            top = namelist[0].split('/')[0]
            candidate = os.path.join(temp_dir, top)
            if os.path.isdir(candidate):
                extracted_folder = candidate
    if extracted_folder is None:
        # fallback: use temp_dir itself (extracted files at archive root)
        extracted_folder = temp_dir
    inferencing = InferenceEngine(model_path='best_f1_model.pth')
    posterior_horn_image, anterior_horn_image, body_image = inferencing.infer_folder(extracted_folder)

    result = {
        'posterior_horn_image': enrich_with_image_and_metadata(posterior_horn_image),
        'anterior_horn_image': enrich_with_image_and_metadata(anterior_horn_image),
        'body_image': enrich_with_image_and_metadata(body_image)
    }

    shutil.rmtree(temp_dir)
    return result
