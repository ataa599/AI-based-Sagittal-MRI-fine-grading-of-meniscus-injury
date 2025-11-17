from src.data_preprocessing_pipeline.creating_new_dataset import NewDatasetConfig, NewDataset
from src.data_preprocessing_pipeline.meniscus_cropping import CroppingMeniscusConfig, CroppingMeniscus
from src.data_preprocessing_pipeline.splitting_dataset import SplittingDatasetConfig, SplittingDataset
from src.data_preprocessing_pipeline.data_augmentation import DataAugmentationConfig, DataAugmentation
from src.logging_and_exception.exception import CustomException
from src.logging_and_exception.logger import logging
import sys


class PreprocessPipeline:
    def __init__(self):
        self.create_dataset_config = NewDatasetConfig()
        self.create_dataset = NewDataset(self.create_dataset_config)

    def start_preprocessing_pipeline(self):
        try:
            # Create Dataset
            logging.info("Starting dataset creation process")
            output_dir, csv_output_path = self.create_dataset.create_dataset()
            logging.info("Dataset creation process completed successfully")

            # Crop Meniscus
            cropping_config = CroppingMeniscusConfig(input_image_path=output_dir)
            cropping_meniscus = CroppingMeniscus(config=cropping_config)
            logging.info("Starting meniscus cropping process")
            cropped_dataset = cropping_meniscus.iniate_cropping() 
            logging.info("Meniscus cropping process completed successfully")

            # Split the Dataset
            datasetsplitting_config = SplittingDatasetConfig(csv_output_path, cropped_dataset)
            splitting_dataset = SplittingDataset(config=datasetsplitting_config)
            logging.info("Starting dataset splitting process")
            train_out, test_out, train_csv, test_csv = splitting_dataset.split_dataset()    
            logging.info("Dataset splitting process completed successfully")
            
            # Data Augmentation
            data_augmentation_config = DataAugmentationConfig(input_image_path=train_out, input_metadata_csv=train_csv)
            data_augmentation = DataAugmentation(config=data_augmentation_config)
            logging.info("Starting data augmentation process")
            augmented_csv, augmented_images = data_augmentation.initiate_augmentation()
            logging.info("Data augmentation process completed successfully")

            return augmented_images, augmented_csv, test_out, test_csv


            
        except Exception as e:
            raise CustomException(e, sys)