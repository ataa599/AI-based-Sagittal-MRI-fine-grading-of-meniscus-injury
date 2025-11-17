from pipelines.preprocess_pipeline import PreprocessPipeline
from pipelines.training_pipeline import TrainingPipeline


if __name__ == "__main__":

    preprocessing_pipeline = PreprocessPipeline()
    augmented_images, augmented_csv, test_out, test_csv = preprocessing_pipeline.start_preprocessing_pipeline()
    
    training_pipeline = TrainingPipeline(augmented_images, test_out, augmented_csv, test_csv)
    training_pipeline.start_training_pipeline()