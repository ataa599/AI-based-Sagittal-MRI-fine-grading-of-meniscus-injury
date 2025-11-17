# from typing import Annotated
# from typing import List
# from fastapi import FastAPI, File, UploadFile
# from src.inference_pipeline.inference import InferenceEngine
# import tempfile
# import shutil
# import os
# app = FastAPI()


# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}


# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     return {"filename": file.filename}

# @app.post("/infer/")
# async def infer_image(file: UploadFile):
#     try:
#         # Save the uploaded file to a temporary location
#         temp_file_path = f"temp_{file.filename}"
#         with open(temp_file_path, "wb") as temp_file:
#             temp_file.write(await file.read())

#         # Initialize the inference engine with the trained model path
#         model_path = "best_f1_model.pth"  # Update with your actual model path
#         inference_engine = InferenceEngine(model_path=model_path)

#         # Perform inference on the uploaded image
#         results = inference_engine.infer_image(temp_file_path)

#         # Clean up the temporary file
        
#         os.remove(temp_file_path)

#         return {"inference_results": results}
#     except Exception as e:
#         return {"error": str(e)}
    

# @app.post("/infer_folder/")
# async def infer_folder(files: List[UploadFile] = File(...)):
#     # create a temp dir for this request
#     temp_dir = tempfile.mkdtemp(prefix="infer_")
#     try:
#         # save all uploaded files
#         saved_paths = []
#         for upload in files:
#             save_path = os.path.join(temp_dir, upload.filename)
#             with open(save_path, "wb") as f:
#                 f.write(await upload.read())
#             saved_paths.append((upload.filename, save_path))

#         # initialize inference engine once
#         model_path = "best_f1_model.pth"
#         engine = InferenceEngine(model_path=model_path)

#         results = {}
#         for orig_name, path in saved_paths:
#             try:
#                 res = engine.infer_image(path)   # use your existing infer method
#                 results[orig_name] = {"ok": True, "result": res}
#             except Exception as e:
#                 results[orig_name] = {"ok": False, "error": str(e)}

#         return {"results": results}
#     finally:
#         # cleanup the temp dir (remove files)
#         shutil.rmtree(temp_dir, ignore_errors=True)



from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import shutil, zipfile, os, base64
from src.inference_pipeline.inference import InferenceEngine
from typing import Dict, Any
import pydicom
from PIL import Image
from io import BytesIO

app = FastAPI()

# Serve static files (including frontend.html)
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Serve the frontend at the root URL
@app.get("/")
def read_frontend():
    return FileResponse("frontend.html")


def convert_dicom_to_png(dcm_path) -> 'Image.Image':
    """
    Convert a DICOM file into an 8-bit grayscale PNG.
    Uses safe min-max scaling (deterministic).
    """
    import pydicom
    import numpy as np
    from PIL import Image
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


@app.post("/infer-folder/")
async def infer_folder(zip_file: UploadFile = File(...)):
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, zip_file.filename)
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    # Assuming the folder to infer is the first extracted folder
    extracted_folder = os.path.join(temp_dir, os.listdir(temp_dir)[0])
    inferencing = InferenceEngine(model_path='best_f1_model.pth')
    posterior_horn_image, anterior_horn_image, body_image = inferencing.infer_folder(extracted_folder)

    def enrich_with_image_and_metadata(image_info: Dict[str, Any]):
        dicom_path = image_info.get('image_path')
        image_base64 = convert_dicom_to_png(dicom_path)
        return {
            'region': image_info.get('region'),
            'predicted_severity': image_info.get('predicted_severity'),
            'confidence': image_info.get('confidence'),
            'image_base64': image_base64,
        }

    result = {
        'posterior_horn_image': enrich_with_image_and_metadata(posterior_horn_image),
        'anterior_horn_image': enrich_with_image_and_metadata(anterior_horn_image),
        'body_image': enrich_with_image_and_metadata(body_image)
    }

    shutil.rmtree(temp_dir)
    return result
