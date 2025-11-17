from src.deep_learning_architecture.training import TrainingConfig, Trainer
import sys
from src.logging_and_exception.exception import CustomException
from src.logging_and_exception.logger import logging


class TrainingPipeline:
    def __init__(self, augmented_images, test_out, augmented_csv, test_csv):
        self.config = TrainingConfig(augmented_images, test_out, augmented_csv, test_csv)

    def start_training_pipeline(self):
        try:
            training = Trainer(config=self.config)
            logging.info("Starting model training process")
            training.initiate_training()
            logging.info("Model training process completed successfully")
        except Exception as e:
            raise CustomException(e, sys)