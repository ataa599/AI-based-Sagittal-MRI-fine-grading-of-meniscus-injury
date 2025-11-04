from src.constants.constants import metadata_path, image_root, regions, ARTIFACT_DIR, NEW_DATASET_DIR, NEW_DATASET_IMAGES, NEW_METADATA_CSV 
from src.logging_and_exception.exception import CustomException
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
import sys

class NewDatasetConfig:
    """Configuration for data preprocessing pipeline"""
    def __init__(self):
        self.metadata_path = metadata_path
        self.image_root = image_root
        self.artifact_dir = Path(ARTIFACT_DIR)
        self.new_dataset_dir = self.artifact_dir / NEW_DATASET_DIR
        self.output_dir = self.new_dataset_dir / NEW_DATASET_IMAGES
        self.csv_output_path = self.new_dataset_dir / NEW_METADATA_CSV

        # Create all necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # The parent of csv_output_path is new_dataset_dir, which is already created above

class NewDataset:
    def __init__(self, config: NewDatasetConfig):
        self.config = config
        self.metadata_path = config.metadata_path
        self.image_root = config.image_root 
        self.output_dir = config.output_dir
        self.csv_output_path = config.csv_output_path

    def convert_dicom_to_png(self, dcm_path) -> Image.Image:
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

        return Image.fromarray(img_scaled)

    

    def create_dataset(self):
        
        try:
            # Read patient metadata, force ID as string for consistency
            metadata_df = pd.read_csv(metadata_path, dtype={"ID": str})
            metadata_df = metadata_df.drop_duplicates(subset="ID", keep="first").set_index("ID")

            rows = []  # will hold all dataset entries

            # Deterministic list of patient IDs: directories only, case-insensitive sort
            patient_ids = [d.name for d in Path(image_root).iterdir() if d.is_dir()]
            patient_ids = sorted(patient_ids, key=str.lower)

            for patient_id in tqdm(patient_ids, desc="Processing Patients"):
                patient_path = Path(image_root) / patient_id / "Best"
                if not patient_path.exists():
                    continue
                if patient_id not in metadata_df.index:
                    continue

                patient_meta = metadata_df.loc[patient_id]

                for region_name, grade_column, (ah, b, ph) in regions:
                    dcm_folder = patient_path / region_name / "Sagittal"
                    if not dcm_folder.exists():
                        continue
                    dcm_files = [*dcm_folder.glob("*.DCM"), *dcm_folder.glob("*.dcm")]
                    if not dcm_files:
                        continue
                    dcm_files = sorted(dcm_files, key=lambda p: p.name.lower())
                    dcm_file = dcm_files[0]

                    out_name = f"{patient_id}__{region_name.replace(' ', '_')}__{dcm_file.stem}.png"
                    output_path = self.output_dir / out_name

                    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    png_image = self.convert_dicom_to_png(dcm_file)
                    png_image.save(output_path)

                    rows.append({
                        "Patient ID": patient_id,
                        "Image Name": output_path.name,
                        "Anterior Horn": ah,
                        "Body": b,
                        "Posterior Horn": ph,
                        "Gender": patient_meta.get("Gender"),
                        "Age": patient_meta.get("Age"),
                        "Height": patient_meta.get("Height"),
                        "Weight": patient_meta.get("Weight"),
                        "Severity of Injury": patient_meta.get(grade_column),
                    })

            # ========================
            # BUILD FINAL CSV
            # ========================
            df_out = pd.DataFrame(rows)
            if not df_out.empty:
                df_out["__region_order"] = df_out["Body"] * 1 + df_out["Posterior Horn"] * 2
                df_out = df_out.sort_values(
                    by=["Patient ID", "__region_order", "Image Name"],
                    ascending=[True, True, True],
                    kind="mergesort"
                ).drop(columns="__region_order").reset_index(drop=True)

            # Ensure parent directory exists and save final metadata CSV
            self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(self.csv_output_path, index=False)
            print(f"Saved {len(df_out)} entries to CSV at:\n{self.csv_output_path}")
            return self.output_dir, self.csv_output_path
        
        except Exception as e:
            raise CustomException(e, sys)