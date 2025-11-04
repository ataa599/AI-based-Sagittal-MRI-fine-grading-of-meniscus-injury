import sys
from src.logging_and_exception.exception import CustomException
from src.logging_and_exception.logger import logging
from src.data_preprocessing_pipeline.creating_new_dataset import NewDatasetConfig, NewDataset
from src.data_preprocessing_pipeline.meniscus_cropping import CroppingMeniscusConfig, CroppingMeniscus


if __name__ == "__main__":
    create_dataset_config = NewDatasetConfig()
    create_dataset = NewDataset(create_dataset_config)
    logging.info("Starting dataset creation process")
    output_dir, csv_output_path = create_dataset.create_dataset()
    print(f"Dataset images saved to: {output_dir}")
    print(f"Dataset metadata CSV saved to: {csv_output_path}")
    logging.info("Dataset creation process completed successfully")

    cropping_config = CroppingMeniscusConfig(input_image_path=output_dir)
    cropping_meniscus = CroppingMeniscus(config=cropping_config)
    logging.info("Starting meniscus cropping process")
    cropped_dataset = cropping_meniscus.iniate_cropping() 
    logging.info("Meniscus cropping process completed successfully")
    print(f"Cropped dataset saved to: {cropped_dataset}")



    # try:
    #     logging.info("Starting the application")
    #     a = 1 / 0  # This will raise a ZeroDivisionError
    # except Exception as e:
    #     logging.error("An error occurred")
    #     raise CustomException(e, sys)