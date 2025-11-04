


ARTIFACT_DIR: str = "Artifacts"


"""DATA CREATION CONSTANTS"""
metadata_path = r"C:\Users\attas\Documents\Queens University Belfast\Thesis\data\data\Notes.csv"
image_root = r"C:\Users\attas\Documents\Queens University Belfast\Thesis\data\data\Image"
NEW_DATASET_DIR: str = "New_Dataset"
NEW_DATASET_IMAGES = "new_dataset_images"
NEW_METADATA_CSV = "new_dataset_metadata.csv"

"""DATASET CROPPING CONSTANTS"""
yolo_model = 'best.pt'
cropped_artifact_dataset: str = "cropped_roi"
conf_threshold = 0.3                 # Minimum confidence for detection
resize_dim = (224, 224)              # Target size (width, height)

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


