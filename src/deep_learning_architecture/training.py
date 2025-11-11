from src.constants.constants import ARTIFACT_DIR, batch_size, epochs, save_all_epochs
from src.logging_and_exception.exception import CustomException
import sys
from pathlib import Path
import torch
from src.deep_learning_architecture.training_utils.training_logger import Logger, log
from datetime import datetime
import os
from src.deep_learning_architecture.training_utils.saggitalmeniscusdataset_loader import get_sagittal_data_loaders
from src.deep_learning_architecture.training_utils.model import DenseNetSagittalModel
from src.deep_learning_architecture.training_utils.config import DenseNetSagittalConfig
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from src.deep_learning_architecture.training_utils.trainer import train_epoch, validate

class TrainingConfig:
    def __init__(self, train_img_path, test_img_path, train_csv_path, test_csv_path):   
        self.train_img_path = train_img_path
        self.test_img_path = test_img_path
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.train_img_path = config.train_img_path
        self.test_img_path = config.test_img_path
        self.train_csv_path = config.train_csv_path
        self.test_csv_path = config.test_csv_path
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.artifact_dir = Path(ARTIFACT_DIR)
        self.save_dir = self.artifact_dir / "Training_Results"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        


    def initiate_training(self):
        try:
            # 0. Configs
            dense_net_model_config = DenseNetSagittalConfig()
            # 1. Create log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.save_dir, f'training_log_{timestamp}.txt')
            sys.stdout = Logger(log_file)

            log(f'Using device: {self.device}')
            log(f"Results will be saved to: {self.save_dir}")
            log(f"Log file: {log_file}")

            # 3. Load data
            log(f"Starting data loading...")
            train_loader, val_loader = get_sagittal_data_loaders(
                self.train_csv_path, self.train_img_path, self.test_csv_path, self.test_img_path, batch_size
            )

            log(f"Using single-view (sagittal) dataset with data augmentation")
            log(f"Training batches: {len(train_loader)}, Batch size: {batch_size}, Estimated training samples: ~{len(train_loader)*batch_size}")
            log(f"Validation batches: {len(val_loader)}, Batch size: {batch_size}, Estimated validation samples: ~{len(val_loader)*batch_size}")

            # 4. Create model with tuned variant & pretrained flag
            model = DenseNetSagittalModel(num_classes=4, pretrained=dense_net_model_config.use_pretrained, densenet_variant=dense_net_model_config.chosen_variant).to(self.device)
            log(f"Model created: DenseNet-{dense_net_model_config.chosen_variant} (Sagittal Meniscus Injury Classification Model)")
            log(f"Number of classes: {4}")
            log(f"Using pretrained weights: {dense_net_model_config.chosen_variant}")

            # 5. Loss & Optimizer (tuned)
            criterion = nn.CrossEntropyLoss(label_smoothing=dense_net_model_config.label_smoothing)
            log(f"Using CrossEntropyLoss with label_smoothing={dense_net_model_config.label_smoothing:.4f}")

            if dense_net_model_config.optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=dense_net_model_config.learning_rate, weight_decay=dense_net_model_config.weight_decay)
                log(f"Using AdamW optimizer, lr={dense_net_model_config.learning_rate:.8f}, weight_decay={dense_net_model_config.weight_decay:.8f}")
            else:
                # fallback (shouldn't happen with tuned params)
                optimizer = optim.SGD(model.parameters(), lr=dense_net_model_config.learning_rate, momentum=0.9, weight_decay=dense_net_model_config.weight_decay, nesterov=True)
                log("Using SGD optimizer, momentum=0.9, nesterov=True")

            # 6. Scheduler (tuned multistep with fractional milestones)
            if dense_net_model_config.scheduler_name == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
                log(f"Using cosine annealing LR scheduler, T_max={epochs}")
            else:  # multistep
                m1 = max(1, int(round(epochs * dense_net_model_config.ms_m1_fraction)))
                m2 = max(m1 + 1, int(round(epochs * dense_net_model_config.ms_m2_fraction)))
                milestones = [m1, m2]
                scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=dense_net_model_config.ms_gamma)
                log(f"Using multi-step LR scheduler, milestones={milestones}, gamma={dense_net_model_config.ms_gamma:.6f}")

            best_val_acc = 0
            best_macro_f1 = 0
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            macro_f1s = []

            log("\nTuned hyperparameters being used:")
            log(f"  optimizer={dense_net_model_config.optimizer_name}, lr={dense_net_model_config.learning_rate}, weight_decay={dense_net_model_config.weight_decay}")
            log(f"  label_smoothing={dense_net_model_config.label_smoothing}, use_pretrained={dense_net_model_config.use_pretrained}, variant={dense_net_model_config.chosen_variant}")
            log(f"  scheduler={dense_net_model_config.scheduler_name} (gamma={dense_net_model_config.ms_gamma if dense_net_model_config.scheduler_name=='multistep' else '—'})")

            # 8. Train
            log(f"\nStarting model training for {epochs} epochs...\n")
            for epoch in range(epochs):
                log(f'\nEpoch {epoch+1}/{epochs}')
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, self.device, epoch)
                val_loss, val_acc, macro_f1, val_preds, val_labels = validate(model, val_loader, criterion, self.device, save_dir=self.save_dir)

                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                log(f"Current learning rate: {current_lr:.6f}")

                train_losses.append(train_loss); val_losses.append(val_loss)
                train_accs.append(train_acc);    val_accs.append(val_acc)
                macro_f1s.append(macro_f1)

                log(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
                log(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Macro F1: {macro_f1:.4f}')

                epoch_results = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'macro_f1': macro_f1,
                    'learning_rate': current_lr
                }

                if save_all_epochs:
                    epoch_model_path = os.path.join(self.save_dir, f'epoch_{epoch+1}_model.pth')
                    torch.save(model.state_dict(), epoch_model_path)
                    log(f'Epoch {epoch+1} model weights saved')

                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    log(f'New best macro F1: {best_macro_f1:.4f}, saving model...')
                    best_f1_model_path = os.path.join(self.save_dir, f'best_f1_model.pth')
                    torch.save(model.state_dict(), best_f1_model_path)

                    best_f1_results_file = os.path.join(self.save_dir, 'best_f1_results.txt')
                    with open(best_f1_results_file, 'w', encoding='utf-8') as f:
                        for key, value in epoch_results.items():
                            f.write(f"{key}: {value}\n")
                    log(f'Detailed results for best macro F1 model saved')

                    _ = validate(model, val_loader, criterion, self.device, print_results=True, save_dir=self.save_dir)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    log(f'New best validation accuracy: {best_val_acc:.2f}%, saving model...')
                    best_model_path = os.path.join(self.save_dir, f'best_acc_model.pth')
                    torch.save(model.state_dict(), best_model_path)

                    best_acc_results_file = os.path.join(self.save_dir, 'best_acc_results.txt')
                    with open(best_acc_results_file, 'w', encoding='utf-8') as f:
                        for key, value in epoch_results.items():
                            f.write(f"{key}: {value}\n")
                    log(f'Detailed results for best accuracy model saved')

                    _ = validate(model, val_loader, criterion, self.device, print_results=True, save_dir=self.save_dir)
                
            # 9. Save training history
            history_file = os.path.join(self.save_dir, 'training_history.txt')
            with open(history_file, 'w', encoding='utf-8') as f:
                for epoch in range(len(train_losses)):
                    f.write(f"Epoch {epoch+1}:\n")
                    f.write(f"  train_loss: {train_losses[epoch]:.4f}\n")
                    f.write(f"  val_loss: {val_losses[epoch]:.4f}\n")
                    f.write(f"  train_acc: {train_accs[epoch]:.2f}%\n")
                    f.write(f"  val_acc: {val_accs[epoch]:.2f}%\n")
                    f.write(f"  macro_f1: {macro_f1s[epoch]:.4f}\n")
                    f.write("-" * 30 + "\n")

            # 10. Training completed
            log(f'\nModel training complete!')
            log(f'Best validation accuracy: {best_val_acc:.2f}%')
            log(f'Best macro-averaged F1: {best_macro_f1:.4f}')
            log(f'Models saved to: {self.save_dir}')
            log(f'Complete training history saved to: {history_file}')

            # 11. Final Evaluation
            log(f"\nFinal evaluation using best accuracy model:")
            model.load_state_dict(torch.load(os.path.join(self.save_dir, f'best_acc_model.pth')))
            _ = validate(model, val_loader, criterion, self.device, print_results=True, save_dir=self.save_dir)

            log(f"\nFinal evaluation using best macro F1 model:")
            model.load_state_dict(torch.load(os.path.join(self.save_dir, f'best_f1_model.pth')))
            _, _, _, all_preds_f1, all_labels_f1 = validate(model, val_loader, criterion, self.device, print_results=True, save_dir=self.save_dir)
            return self.save_dir

        except CustomException as e:
            raise CustomException(e, sys)


    