from tqdm import tqdm
from deep_learning_architecture.training_utils.training_logger import log
from sklearn.metrics import precision_recall_fscore_support, f1_score
import os
import torch
import torch.nn.functional as F
# =========================
# Train / Validate (unchanged)
# =========================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        inputs = {
            'image': batch['image'].to(device),
            'position': batch['position'].to(device)
        }
        labels = batch['damage_level'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs['damage_logits'], labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs['damage_logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            label_idx = labels[i].item()
            pred_idx = predicted[i].item()
            class_total[label_idx] += 1
            if label_idx == pred_idx:
                class_correct[label_idx] += 1

        pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100.*correct/total})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    log("\nTraining set class-wise accuracy:")
    for i in range(4):
        acc = 100 * class_correct[i] / max(class_total[i], 1)
        log(f"Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, print_results=False, save_dir=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]

    all_preds = []
    all_labels = []
    all_probs = []
    all_positions = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            inputs = {
                'image': batch['image'].to(device),
                'position': batch['position'].to(device)
            }

            damage_level = batch['damage_level'].to(device)
            position = batch['position']

            outputs = model(inputs)

            loss = criterion(outputs['damage_logits'], damage_level)
            running_loss += loss.item()

            _, predicted = outputs['damage_logits'].max(1)
            probs = F.softmax(outputs['damage_logits'], dim=1)

            total += damage_level.size(0)
            correct += predicted.eq(damage_level).sum().item()

            for i in range(damage_level.size(0)):
                label_idx = damage_level[i].item()
                pred_idx = predicted[i].item()
                class_total[label_idx] += 1
                if label_idx == pred_idx:
                    class_correct[label_idx] += 1

                all_preds.append(pred_idx)
                all_labels.append(label_idx)
                all_probs.append(probs[i].cpu().numpy())
                all_positions.append(position[i].item())

            pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100.*correct/total})

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3], average=None
    )
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    log("\nValidation metrics:")
    log(f"Overall accuracy: {val_acc:.2f}%")
    log(f"Macro-averaged F1 score: {macro_f1:.4f}")
    log("\nDetailed class metrics:")
    log(f"{'Class':<6} {'Precision(P)':<12} {'Recall(R)':<12} {'F1 Score':<12} {'Support':<10} {'Correct':<10}")
    log('-' * 70)

    for i in range(4):
        log(f"{i:<6} {precision[i]:.4f}      {recall[i]:.4f}      {f1[i]:.4f}      {class_total[i]:<10} {class_correct[i]:<10}")

    log("\nValidation class-wise accuracy:")
    for i in range(4):
        acc = 100 * class_correct[i] / max(class_total[i], 1)
        log(f"Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    anterior_correct = 0
    anterior_total = 0
    posterior_correct = 0
    posterior_total = 0
    body_correct = 0
    body_total = 0

    for i in range(len(all_labels)):
        if all_positions[i] == 1:  # Anterior
            anterior_total += 1
            if all_labels[i] == all_preds[i]:
                anterior_correct += 1
        elif all_positions[i] == 0: # Posterior
            posterior_total += 1
            if all_labels[i] == all_preds[i]:
                posterior_correct += 1
        elif all_positions[i] == 2:
            body_total += 1
            if all_labels[i] == all_preds[i]:
                body_correct += 1

    anterior_acc = 100 * anterior_correct / max(anterior_total, 1)
    posterior_acc = 100 * posterior_correct / max(posterior_total, 1)
    body_acc = 100 * body_correct / max(body_total, 1)

    log(f"\nAnterior accuracy: {anterior_acc:.2f}% ({anterior_correct}/{anterior_total})")
    log(f"Posterior accuracy: {posterior_acc:.2f}% ({posterior_correct}/{posterior_total})")
    log(f"Body accuracy: {body_acc:.2f}% ({body_correct}/{body_total})")

    if print_results and save_dir:
        result_file = os.path.join(save_dir, 'validation_results.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("Validation prediction details:\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Index':<6} {'Position':<8} {'True Label':<10} {'Predicted':<10} {'Correct':<10} {'Class Probabilities'}\n")
            f.write("-" * 80 + "\n")

            for i in range(len(all_labels)):
                is_correct = all_labels[i] == all_preds[i]
                probs_str = " ".join([f"{p:.3f}" for p in all_probs[i]])
                if all_positions[i] == 1:
                    part = "Anterior"
                elif all_positions[i] == 0:
                    part = "Posterior"
                elif all_positions[i] == 2:
                    part = "Body"
                else:
                    part = "Unknown"

                f.write(f"{i+1:<6} {part:<8} {all_labels[i]:<10} {all_preds[i]:<10} {'✓' if is_correct else '✗':<10} {probs_str}\n")

            f.write("=" * 80 + "\n")
            f.write(f"Total samples: {len(all_labels)}\n")
            f.write(f"Correct samples: {sum(1 for i in range(len(all_labels)) if all_labels[i] == all_preds[i])}\n")
            f.write(f"Overall accuracy: {val_acc:.2f}%\n")
            f.write(f"Macro-averaged F1 score: {macro_f1:.4f}\n")
            f.write("\nDetailed class metrics:\n")
            f.write(f"{'Class':<6} {'Precision(P)':<12} {'Recall(R)':<12} {'F1 Score':<12} {'Support':<10} {'Correct':<10}\n")
            f.write('-' * 70 + "\n")
            for i in range(4):
                f.write(f"{i:<6} {precision[i]:.4f}      {recall[i]:.4f}      {f1[i]:.4f}      {class_total[i]:<10} {class_correct[i]:<10}\n")
            f.write(f"\nAnterior accuracy: {anterior_acc:.2f}% ({anterior_correct}/{anterior_total})\n")
            f.write(f"Posterior accuracy: {posterior_acc:.2f}% ({posterior_correct}/{posterior_total})\n")

        log(f"Detailed results saved to: {result_file}")

    return val_loss, val_acc, macro_f1, all_preds, all_labels