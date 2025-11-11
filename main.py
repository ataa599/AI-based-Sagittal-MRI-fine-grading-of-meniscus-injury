import sys
from src.logging_and_exception.exception import CustomException
from src.logging_and_exception.logger import logging
from src.data_preprocessing_pipeline.creating_new_dataset import NewDatasetConfig, NewDataset
from src.data_preprocessing_pipeline.meniscus_cropping import CroppingMeniscusConfig, CroppingMeniscus
from src.data_preprocessing_pipeline.splitting_dataset import SplittingDatasetConfig, SplittingDataset
from src.data_preprocessing_pipeline.data_augmentation import DataAugmentationConfig, DataAugmentation
from src.deep_learning_architecture.training import TrainingConfig, Trainer
from src.inference_pipeline.inference import InferenceEngine

if __name__ == "__main__":
    # create_dataset_config = NewDatasetConfig()
    # create_dataset = NewDataset(create_dataset_config)
    # logging.info("Starting dataset creation process")
    # output_dir, csv_output_path = create_dataset.create_dataset()
    # print(f"Dataset images saved to: {output_dir}")
    # print(f"Dataset metadata CSV saved to: {csv_output_path}")
    # logging.info("Dataset creation process completed successfully")

    # cropping_config = CroppingMeniscusConfig(input_image_path=output_dir)
    # cropping_meniscus = CroppingMeniscus(config=cropping_config)
    # logging.info("Starting meniscus cropping process")
    # cropped_dataset = cropping_meniscus.iniate_cropping() 
    # logging.info("Meniscus cropping process completed successfully")
    # print(f"Cropped dataset saved to: {cropped_dataset}")

    # datasetsplitting_config = SplittingDatasetConfig(csv_output_path, cropped_dataset)
    # splitting_dataset = SplittingDataset(config=datasetsplitting_config)
    # logging.info("Starting dataset splitting process")
    # train_out, test_out, train_csv, test_csv = splitting_dataset.split_dataset()    
    # logging.info("Dataset splitting process completed successfully")
    # print(f"Train images saved to: {train_out}")    
    # print(f"Test images saved to: {test_out}")
    # print(f"Train CSV saved to: {train_csv}")
    # print(f"Test CSV saved to: {test_csv}")

    # data_augmentation_config = DataAugmentationConfig(input_image_path=train_out, input_metadata_csv=train_csv)
    # data_augmentation = DataAugmentation(config=data_augmentation_config)
    # logging.info("Starting data augmentation process")
    # augmented_csv, augmented_images = data_augmentation.initiate_augmentation()
    # logging.info("Data augmentation process completed successfully")
    # print(f"Augmented images saved to: {augmented_images}")
    # print(f"Augmented CSV saved to: {augmented_csv}")

    # augmented_images = "Artifacts\\augmented_dataset\\augmented_images"
    # augmented_csv = "Artifacts\\augmented_dataset\\augmented_metadata.csv"
    # test_out = "Artifacts\\split_dataset\\test"
    # test_csv = "Artifacts\\split_dataset\\test_dataset.csv"

    # model_training_config = TrainingConfig(augmented_images, test_out, augmented_csv, test_csv)
    # training = Trainer(config=model_training_config)
    # logging.info("Starting model training process")
    # training_result_dir = training.initiate_training()
    # logging.info("Model training process completed successfully")
    # print(f"Model training artifacts saved to: {training_result_dir}")

    training_result_dir = "Artifacts\\Training_Results"


    # infer_folder_dir = "C:/Users/attas/Documents/Queens University Belfast/Thesis/data/data/Image/ANON0B3727ED53/Sagittal"
    infer_folder_dir = "C:/Users/attas/Documents/Queens University Belfast\Thesis/data/data/Image/ANON1BC4C7A93D/Sagittal"
    # inferencing = InferenceEngine(model_path=f"{training_result_dir}\\best_f1_model.pth")
    inferencing = InferenceEngine(model_path='best_f1_model.pth')
    logging.info("Starting inference process on test dataset")
    posterior_horn_image, anterior_horn_image, body_image = inferencing.infer_folder(infer_folder_dir)
    logging.info("Inference process completed successfully")
    print(f"Inference results: {posterior_horn_image}, {anterior_horn_image}, {body_image}")








    # try:
    #     logging.info("Starting the application")
    #     a = 1 / 0  # This will raise a ZeroDivisionError
    # except Exception as e:
    #     logging.error("An error occurred")
    #     raise CustomException(e, sys)