import torch
from torchvision import transforms


ARTIFACT_DIR: str = "Artifacts"
random_state = 42


"""DATA CREATION CONSTANTS"""
metadata_path = r"C:\Users\attas\Documents\Queens University Belfast\Thesis\data\data\Notes.csv"
image_root = r"C:\Users\attas\Documents\Queens University Belfast\Thesis\data\data\Image"
NEW_DATASET_DIR: str = "New_Dataset"
NEW_DATASET_IMAGES = "new_dataset_images"
NEW_METADATA_CSV = "new_dataset_metadata.csv"
# ========================
# REGION DEFINITIONS
# ========================
# Each region:
#   - folder name inside "Best"
#   - metadata grade column
#   - one-hot encoding [Anterior, Body, Posterior]
regions = [
    ("anterior horn", "Anterior_Horn_Injury_Grade", (1, 0, 0)),
    ("body",          "Body_Injury_Grade",         (0, 1, 0)),
    ("posterior horn","Posterior_Horn_Injury_Grade",(0, 0, 1)),
]


"""DATASET CROPPING CONSTANTS"""
yolo_model = 'best.pt'
cropped_artifact_dataset: str = "cropped_roi"
conf_threshold = 0.3                 # Minimum confidence for detection
resize_dim = (224, 224)              # Target size (width, height)

"""DATA SPLITTING CONSTANTS"""
output_path = "split_dataset"
n_splits = 5
train = "train"
test = "test"

"""DATA AUGMENTATION CONSTANTS"""
AUGMENTED_DATASET_DIR: str = "augmented_dataset"
augmented_images_dir = "augmented_images"
augmented_metadata_csv = "augmented_metadata.csv"
chunk_size=50

# Define augmentation strategies (same content, order matters for determinism)
augmentation_strategies = [
        transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.92, 1.0), ratio=(0.97, 1.03)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.08, contrast=0.12),
        ]),
        transforms.Compose([
            transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        ]),
        transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.04, p=0.5),
            transforms.ColorJitter(contrast=0.15),
        ]),
        transforms.Compose([
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.05),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((224, 224), scale=(0.94, 1.0), ratio=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.01, 0.05), ratio=(0.3, 3.3), value='random'),
            transforms.ToPILImage(),
        ]),
        transforms.Compose([
            transforms.RandomAffine(degrees=6, translate=(0.01, 0.01)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ColorJitter(brightness=0.06, contrast=0.1),
        ]),
    ]



batch_size = 16
num_epochs = 4
batch_size = 16
num_classes = 4
lr_scheduler = 'multistep'
use_pretrained = False
epochs = 2
save_all_epochs = False


