from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.constants.constants import output_path, random_state, n_splits, train, test, ARTIFACT_DIR
import os
import shutil

class SplittingDatasetConfig:
    def __init__(self, dataset_csv_path, cropped_dataset_path):
        self.dataset_csv_path = dataset_csv_path
        self.cropped_dataset_path = cropped_dataset_path

class SplittingDataset:
    def __init__(self, config: SplittingDatasetConfig):
        self.dataset_csv_path = config.dataset_csv_path
        self.image_root = Path(config.cropped_dataset_path)  # <-- fix here
        self.artifact_dir = Path(ARTIFACT_DIR)
        self.output_path = self.artifact_dir / output_path               # <-- and here
        self.train_out = self.output_path / train
        self.test_out  = self.output_path / test    

    # --- Helper: copy only existing images and collect the ones actually copied ---
    def copy_images(self, df_split: pd.DataFrame, out_dir: Path):
        kept_rows = []
        for img_name in df_split["Image Name"]:
            src = self.image_root / img_name
            dst = out_dir / img_name
            if src.exists():
                shutil.copy2(src, dst)
                kept_rows.append(img_name)
            else:
                print(f"⚠ Missing: {src}")
        # Return filtered DataFrame containing only rows with existing files
        return df_split[df_split["Image Name"].isin(kept_rows)].reset_index(drop=True)

    def split_dataset(self):
        # --- Step 1: Load dataset (deterministic order) ---
        df = pd.read_csv(self.dataset_csv_path)

        # Enforce types and deterministic sort so SGKF sees a stable order
        df["Severity of Injury"] = df["Severity of Injury"].astype(int)
        df["Patient ID"] = df["Patient ID"].astype(str)
        df["Image Name"] = df["Image Name"].astype(str)

        df = df.sort_values(by=["Patient ID", "Image Name"], kind="mergesort").reset_index(drop=True)

        # Extract labels and groups
        y = df["Severity of Injury"].values
        groups = df["Patient ID"].values

        # --- Step 2: Stratified Group K-Fold Split (deterministic with fixed seed & sorted df) ---
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(sgkf.split(df, y, groups))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        print("Train class counts:\n", train_df["Severity of Injury"].value_counts().sort_index())
        print("Test class counts:\n",  test_df["Severity of Injury"].value_counts().sort_index())

        # --- Step 3: Prepare output folders (clean for reproducibility) ---
        # train_out = output_root / "train"
        # test_out  = output_root / "test"
        for d in (self.train_out, self.test_out):
            d_path = Path(d)
            if d_path.exists():
                shutil.rmtree(d_path)
            d_path.mkdir(parents=True, exist_ok=True)

        # --- Step 4: Copy images and write CSVs that reflect the files actually copied ---
        train_df_copied = self.copy_images(train_df, self.train_out)
        test_df_copied  = self.copy_images(test_df,  self.test_out)

        # Save split CSVs (reflecting exactly what's in the folders)
        split_dir = self.output_path
        split_dir.mkdir(parents=True, exist_ok=True)
        train_csv = split_dir / "train_dataset.csv"
        test_csv  = split_dir / "test_dataset.csv"

        train_df_copied.to_csv(train_csv, index=False)
        test_df_copied.to_csv(test_csv, index=False)

        # print("Done splitting images into train/ and test/")
        # print(f"Train CSV: {train_csv} ({len(train_df_copied)} rows)")
        # print(f"Test  CSV: {test_csv} ({len(test_df_copied)} rows)")
        return self.train_out, self.test_out, train_csv, test_csv
