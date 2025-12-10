# AI-based Sagittal MRI Fine Grading of Meniscus Injury

This application uses deep learning to analyze sagittal MRI slices of the knee and provide fine-grained classification of meniscus injuries to assist clinicians and reduce workload. The system processes DICOM images and identifies three anatomical regions of the meniscus:

- **Posterior horn:** the back portion of the meniscus
- **Body:** the central portion of the meniscus
- **Anterior horn:** the front portion of the meniscus

For each region, the AI model predicts injury severity and provides a confidence score to aid clinical diagnosis.
This project provides an end-to-end solution for preprocessing, training, and inference for fine grading of meniscus injury using sagittal MRI scans. It exposes a FastAPI service for training and inference, and organizes code and artifacts to follow repeatable MLOps practices.

## Demo Video
![Watch the demo video](./demo.gif)

## Live Demo
The application is hosted here:

- Live Space: [AI-based Sagittal MRI Fine Grading of Meniscus Injury](https://huggingface.co/spaces/ataa599/AI-based-Sagittal-MRI-Fine-Grading-of-Meniscus-Injury)

Note: This is a computationally intensive pipeline (YOLO ROI detection + DenseNet classification on DICOMs). On CPU-only hardware (no GPUs available on the hosted space), a single end-to-end run may take approximately 15–20 minutes to execute. Please be patient while the inference completes.

## Architecture
- FastAPI app (`app.py`) serving web UI and APIs
- Preprocessing pipeline (`pipelines/preprocess_pipeline.py`) to create, crop, split, and augment data
- Training pipeline (`pipelines/training_pipeline.py`) to train models
- Inference pipeline (`src/inference_pipeline/inference.py`) for predictions
- Deep learning modules (`src/deep_learning_architecture/`) including model, trainer, and config
- Artifacts folder structure for datasets, splits, and training results
- Logging and exception handling (`src/logging_and_exception/`)


## Pipelines
1. Preprocess

```mermaid
flowchart TD
    A([Start]) --> B[Create Dataset]
    B -->|output_dir, csv_output_path| C[Crop Meniscus]
    C -->|cropped_dataset| D[Split Dataset]
    D -->|train_out, test_out, train_csv, test_csv| E[Data Augmentation]
    Custom_YOLO_Model -->|Fine-tuned YOLO model for Meniscus Detection| C

    E -->|augmented_images, augmented_csv| F([Return])
    D --> F

    F --> O1[augmented_images]
    F --> O2[augmented_csv]
    F --> O3[test_out]
    F --> O4[test_csv]

    %% Error handling
    B -.->|on error| X{Exception}
    C -.->|on error| X
    D -.->|on error| X
    E -.->|on error| X
    X --> Y[CustomException]
```

2. Training

```mermaid
flowchart TD
    A([Start]) --> B[Initialize TrainingConfig]
    
    subgraph Inputs
        I1[(augmented_images)]
        I2[(test_out)]
        I3[(augmented_csv)]
        I4[(test_csv)]
    end
    I1 --> B
    I2 --> B
    I3 --> B
    I4 --> B

    B --> C[Create Trainer]
    C --> D[Start Training]
    D --> E([Training completed])
    E --> F([Save best model classifier for inference])

    %% Error handling
    B -.->|on error| X{Exception}
    C -.->|on error| X
    D -.->|on error| X
    X --> Y[CustomException]
```
3. Inferencing
```mermaid
flowchart TD
    A([Start]) --> B{Request}
    B -->|infer-hardcoded| D[Load Sagittal.zip]

    C --> E[Save to temp dir]
    D --> E
    E --> F[Extract ZIP]
    F --> G[Resolve extracted folder]

    G --> H[Init InferenceEngine]

    subgraph InferenceEngine
        H --> I[Load DenseNet weights: best_f1_model.pth]
        H --> J{YOLO path available?}
        J -->|yes| K[Load YOLO model]
        J -->|no| L[Skip detection]
    end

    G --> M[List images by extensions]
    M --> N[For each image]
    N --> O[Load DICOM images ]
    O --> P{YOLO available}
    P -->|yes| Q[Detect ROI and crop]
    P -->|no| R[Use original image]
    Q --> S[Transform -> 224x224 tensor]
    R --> S

    S --> T[Predict per region]
    T --> U[Select region with max confidence]
    U --> V[Collect path, region, severity and confidence]
    V --> W{More images?}
    W -->|yes| N
    W -->|no| X[Pick top image per region by confidence]

    X --> Y[Convert selected images to base64 PNG]
    Y --> Z[Build response JSON]
    Z --> AA[Cleanup temp dir]
    AA --> AB([End])

    subgraph Outputs
        Z --> O1[posterior_horn_image]
        Z --> O2[anterior_horn_image]
        Z --> O3[body_image]
        O1 --> F1{region, predicted_severity, confidence, image_base64}
        O2 --> F2{region, predicted_severity, confidence, image_base64}
        O3 --> F3{region, predicted_severity, confidence, image_base64}
    end
```

## MLOps Practices
- Versioned Artifacts: Output data and models stored under `Artifacts/` with clear subfolders (dataset, splits, results)
- Reproducible Pipelines: Deterministic preprocessing steps encapsulated in configs (`*Config` dataclasses) and pipeline classes
- Configuration Management: Central config modules (`src/.../training_utils/config.py`) and pipeline configs
- Logging & Exceptions: Uniform logging and `CustomException` for robust error handling
- Environment Management: `requirements.txt` and optional `venv` for consistent environments
- Containerization: `Dockerfile` for reproducible builds and deployment
- Notebooks to Production: `notebooks/testing.ipynb` for exploration, pipelines for production-ready flow
- Separation of Concerns: Clear module boundaries (preprocess, training, inference, web)


## Data & Artifacts
- `Artifacts/augmented_dataset/`: augmented images and metadata
- `Artifacts/split_dataset/`: train/test splits and CSVs
- `Artifacts/Training_Results/`: models (`best_acc_model.pth`, `best_f1_model.pth`) and logs

## Folder Guide
- `app.py`: FastAPI server and endpoints
- `pipelines/`: Preprocess and training orchestration
- `src/`: Modules for data processing, models, training, logging, inference
- `static/`, `templates/`: Frontend assets for the web UI
- `notebooks/`: Experiments and sanity checks
