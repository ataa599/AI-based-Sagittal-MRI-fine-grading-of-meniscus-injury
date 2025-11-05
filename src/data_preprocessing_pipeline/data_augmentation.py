import os
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import random
from src.constants.constants import ARTIFACT_DIR, AUGMENTED_DATASET_DIR, augmentation_strategies, augmented_images_dir, augmented_metadata_csv, chunk_size
from src.logging_and_exception.exception import CustomException
import sys
import shutil



class DataAugmentationConfig:
    def __init__(self, input_image_path, input_metadata_csv):
        self.input_image_path = input_image_path
        self.input_metadata_csv = input_metadata_csv
        

class DataAugmentation:
    def __init__(self, config: DataAugmentationConfig):
        self.input_dir = config.input_image_path
        self.metadata_csv = config.input_metadata_csv
        self.artifact_dir = os.path.join(ARTIFACT_DIR)
        self.augmented_output_dir = os.path.join(self.artifact_dir, AUGMENTED_DATASET_DIR)
        self.augmented_images_path = os.path.join(self.augmented_output_dir, augmented_images_dir)
        self.augmented_metadata_csv = os.path.join(self.augmented_output_dir, augmented_metadata_csv)

        # Ensure directories exist
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.augmented_output_dir, exist_ok=True)

    # ---------- Global determinism ----------
    def set_global_determinism(seed: int = 42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Helps PyTorch determinism on CUDA (if you later move this to GPU code)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    set_global_determinism(42)

    # ---------- Deterministic helpers ----------
    def stable_hash_to_int(self, *parts, mod=None):
        """Hash arbitrary strings to a positive int; optionally reduce modulo."""
        msg = "|".join(str(p) for p in parts)
        h = int(hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16], 16)
        return h if mod is None else h % mod

    def per_sample_seed(self, severity_class, img_name, aug_index):
        return self.stable_hash_to_int("aug", severity_class, img_name, aug_index, mod=2**31 - 1)

    def deterministic_strategy_index(self, severity_class, img_name, aug_index, n_strategies):
        return self.stable_hash_to_int("strat", severity_class, img_name, aug_index, mod=n_strategies)
    
    def moving_train_images_into_augmented_folder(self, source_folder, destination_folder):
        try:
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # image files only
                        src_path = os.path.join(root, file)
                        
                        # Preserve subdirectory structure
                        relative_path = os.path.relpath(src_path, source_folder)
                        dest_path = os.path.join(destination_folder, relative_path)

                        # Create subfolders if needed
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                        if not os.path.exists(dest_path):  # Don't overwrite
                            shutil.copy2(src_path, dest_path)
                            print(f" Copied: {relative_path}")
                        else:
                            print(f" Skipped (already exists): {relative_path}")
        except Exception as e:
            raise CustomException(e, sys)


    def augment_class_images(self, severity_class, target_count):
        try:
            img_dir = Path(self.input_dir)
            augmented_img_dir = Path(self.augmented_images_path)
            output_csv_path = Path(self.augmented_metadata_csv)
            original_csv_path = Path(self.metadata_csv)

            augmented_img_dir.mkdir(parents=True, exist_ok=True)

            # Always read the CURRENT output CSV if exists so reruns are idempotent.
            # This is reproducible as long as files/CSV content are unchanged between runs.
            try:
                if output_csv_path.exists():
                    df = pd.read_csv(output_csv_path)
                    print(f"Loaded existing augmented CSV: {output_csv_path}")
                else:
                    df = pd.read_csv(original_csv_path)
                    print(f"Loaded original CSV: {original_csv_path}")
            except PermissionError:
                print(f"Permission denied: {output_csv_path} is open or locked.\n Close the file and try again.")
                return

            # Types + deterministic ordering
            for col in ["Severity of Injury", "Image Name"]:
                if col in df.columns:
                    if col == "Severity of Injury":
                        df[col] = df[col].astype(int)
                    else:
                        df[col] = df[col].astype(str)

            df = df.sort_values(["Image Name"], kind="mergesort").reset_index(drop=True)

            class_df = df[df["Severity of Injury"] == severity_class].copy()
            current_count = len(class_df)
            required_augmented = target_count - current_count

            if required_augmented <= 0:
                print(f" No augmentation needed for class {severity_class} (already has {current_count})")
                return

            print(f" Augmenting class {severity_class} from {current_count} → {target_count} (adding {required_augmented})")



            # Build a repeated pool deterministically
            if current_count == 0:
                print(f" No originals for class {severity_class}; cannot augment.")
                return

            reps = (required_augmented // current_count) + 1
            repeated_df = pd.concat([class_df] * reps, ignore_index=True)
            repeated_df = repeated_df.sample(frac=1.0, random_state=999).reset_index(drop=True)
            repeated_df = repeated_df.iloc[:required_augmented].reset_index(drop=True)

            augmented_rows = []
            n_strat = len(augmentation_strategies)

            # Track per-class running index for filename stability
            running_idx = 0

            for i in tqdm(range(0, required_augmented, chunk_size), desc=f"Class {severity_class}"):
                chunk_df = repeated_df.iloc[i:i + chunk_size]

                for _, row in chunk_df.iterrows():
                    img_name = str(row["Image Name"])
                    img_path = img_dir / img_name

                    if not img_path.exists():
                        print(f" Missing image: {img_path}")
                        continue

                    # Choose strategy deterministically
                    strat_idx = self.deterministic_strategy_index(severity_class, img_name, running_idx, n_strat)
                    strategy = augmentation_strategies[strat_idx]

                    # Set per-sample seeds so every random op inside transforms is fixed
                    seed_val = self.per_sample_seed(severity_class, img_name, running_idx)
                    random.seed(seed_val)
                    np.random.seed(seed_val)
                    torch.manual_seed(seed_val)

                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f" Error reading {img_path}: {e}")
                        running_idx += 1
                        continue

                    try:
                        aug_image = strategy(image)
                    except Exception as e:
                        print(f" Augmentation failed for {img_path}: {e}")
                        running_idx += 1
                        continue

                    # Deterministic filename
                    stem = Path(img_name).stem
                    ext = Path(img_name).suffix or ".png"
                    new_filename = f"{stem}__aug{severity_class}_{running_idx:05d}{ext}"
                    new_img_path = augmented_img_dir / new_filename

                    try:
                        aug_image.save(new_img_path)
                    except PermissionError:
                        print(f" Permission error saving image: {new_img_path}")
                        running_idx += 1
                        continue

                    # New metadata row (deterministic)
                    new_row = row.copy()
                    new_row["Image Name"] = new_filename
                    augmented_rows.append(new_row)

                    running_idx += 1  # advance after successful attempt

            # Save to CSV (append deterministically at the end)
            aug_df = pd.DataFrame(augmented_rows)

            final_df = pd.concat([df, aug_df], ignore_index=True)
            # Keep a deterministic order in the final CSV
            final_df = final_df.sort_values(["Image Name"], kind="mergesort").reset_index(drop=True)

            try:
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                final_df.to_csv(output_csv_path, index=False)
                print(f"\n Augmentation complete for class {severity_class}")
                print(f" Augmented images saved to: {augmented_img_dir}")
                print(f" Updated CSV saved to: {output_csv_path}")
            except PermissionError:
                print(f" Cannot write to {output_csv_path}. Please close the file and try again.")   
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_augmentation(self):
        try:
            # Load metadata to determine class distribution
            metadata_df = pd.read_csv(self.metadata_csv)

            class_counts = metadata_df["Severity of Injury"].value_counts().to_dict()
            max_count = max(class_counts.values())

            print(f"Current class distribution: {class_counts}")
            print(f"Targeting {max_count} images per class after augmentation.\n")

            for severity_class in sorted(class_counts.keys()):
                self.augment_class_images(severity_class, max_count)

            # Finally, move all original training images into the augmented folder as well
            self.moving_train_images_into_augmented_folder(self.input_dir, self.augmented_images_path)

            return self.augmented_metadata_csv, self.augmented_images_path

        except Exception as e:
            raise CustomException(e, sys)