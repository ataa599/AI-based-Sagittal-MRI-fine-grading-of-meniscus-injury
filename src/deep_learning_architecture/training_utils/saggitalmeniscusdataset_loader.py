import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class SagittalMeniscusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Deterministic ordering before filtering
        df = df.sort_values(["Patient ID", "Image Name"], kind="mergesort").reset_index(drop=True)

        # Filter rows where image file exists
        valid_rows = []
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, str(row['Image Name']))
            if os.path.exists(img_path):
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_name = str(row['Image Name'])
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        damage_level = int(row['Severity of Injury'])

        # Determine anatomical region
        if row['Anterior Horn'] == 1:
            position = 1
        elif row['Posterior Horn'] == 1:
            position = 0
        elif row['Body'] == 1:
            position = 2
        else:
            raise ValueError(f"Invalid row {idx}: no region marked as 1")

        return {
            'image': image,
            'damage_level': damage_level,
            'position': torch.tensor(position, dtype=torch.long)
        }

def get_sagittal_data_loaders(csv_file_train, img_dir_train,csv_file_test, img_dir_test, batch_size=32):
    df_train = pd.read_csv(csv_file_train)
    df_test = pd.read_csv(csv_file_test)

    # Keep only rows that have one anatomical region marked as 1
    df_train = df_train[(df_train['Anterior Horn'] == 1) | (df_train['Posterior Horn'] == 1) | (df_train['Body'] == 1)]
    df_test = df_test[(df_test['Anterior Horn'] == 1) | (df_test['Posterior Horn'] == 1) | (df_test['Body'] == 1)]

    # Drop missing values
    df_train = df_train.dropna(subset=['Image Name', 'Severity of Injury'])
    df_test = df_test.dropna(subset=['Image Name', 'Severity of Injury'])


    # # Train-validation split with stratification
    # train_df, val_df = train_test_split(
    #     df, train_size=train_ratio,
    #     stratify=df['Severity of Injury'],
    #     random_state=42
    # )

    train_df = df_train
    val_df = df_test

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = SagittalMeniscusDataset(train_df, img_dir_train, transform=train_transform)
    val_dataset = SagittalMeniscusDataset(val_df, img_dir_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
