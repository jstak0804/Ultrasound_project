import os
import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score,
    roc_curve, auc
)
from torchmetrics.classification import F1Score, Precision, Recall

from functions_for_train import (
    find_image_sizes, EarlyStopping,
    calculate_optimal_size, get_unique_filename,
    save_results_to_csv
)
from model import EnhancedResNet, CustomModel
from Grad_Cam import (
    log_gradcam_examples, log_gradcam_to_wandb,
    overlay_heatmap_on_image, generate_gradcam_heatmap,
    visualize_cam, show_cam_on_image, GradCAM
)

# ----------------------------
# Ensure plots directory exists
# ----------------------------
PLOTS_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_roc_f1_curves(y_true, y_prob, epoch, out_dir=PLOTS_DIR):
    """
    y_true: list of true labels (0 or 1)
    y_prob: list or array of predicted probabilities shape (N,) or (N,1) or (N,2)
    """
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    probs = np.array(y_prob)

    # extract positive-class probability
    if probs.ndim == 2 and probs.shape[1] > 1:
        pos_probs = probs[:, 1]
    else:
        pos_probs = probs.ravel()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, f"roc_curve_epoch_{epoch}.png"), dpi=300)
    plt.close()

    # F1-Confidence Curve
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [
        f1_score(y_true, (pos_probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"F1-Confidence Curve (Epoch {epoch})")
    plt.savefig(os.path.join(out_dir, f"f1_conf_curve_epoch_{epoch}.png"), dpi=300)
    plt.close()

# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser(description='Train binary US classifier with ROC & F1 curves')
parser.add_argument('--epochs',     type=int,   default=30,               help='Number of epochs')
parser.add_argument('--batch_size', type=int,   default=32,               help='Batch size (-1 for auto)')
parser.add_argument('--seeds',      nargs='+',   type=int, default=[24],   help='Random seeds')
args = parser.parse_args()

# ----------------------------
# WandB init
# ----------------------------
wandb.init(project="LiverUS_binary_classification", entity="ddurbozak")
wandb.config.update({
    "learning_rate": 3e-5,
    "epochs":        args.epochs,
    "batch_size":    args.batch_size
})

# ----------------------------
# Data transforms & loaders
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder("Liver_US/train_2", transform=train_transform)
test_dataset  = datasets.ImageFolder("Liver_US/test_2",  transform=test_transform)

batch_size = args.batch_size if args.batch_size != -1 else len(train_dataset)//100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
dataloaders = {"train": train_loader, "test": test_loader}

# ----------------------------
# Training / Evaluation
# ----------------------------
def train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    results = []
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_test_acc = 0.0
    best_epoch    = 0
    best_cm       = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ["train", "test"]:
            is_train = (phase == "train")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            correct      = 0
            f1_m = F1Score(num_classes=2, average='macro', task='multiclass').to(device)
            prec_m = Precision(num_classes=2, average='macro', task='multiclass').to(device)
            rec_m  = Recall(num_classes=2, average='macro', task='multiclass').to(device)

            all_labels = []
            all_preds  = []
            all_probs  = []

            for inputs, labels in tqdm(dataloaders[phase],
                                      desc=f"{phase.capitalize()} Epoch {epoch+1}",
                                      unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    if not is_train:
                        probs = torch.softmax(outputs, dim=1)
                        all_probs.extend(probs.cpu().numpy())

                    _, preds = torch.max(outputs, 1)
                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct      += (preds == labels).sum().item()

                f1_m.update(outputs, labels)
                prec_m.update(outputs, labels)
                rec_m.update(outputs, labels)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            # epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = correct / len(dataloaders[phase].dataset)
            epoch_f1   = f1_m.compute().item()
            epoch_prec = prec_m.compute().item()
            epoch_rec  = rec_m.compute().item()

            # log basic metrics
            wandb.log({
                f"{phase.capitalize()} Loss": epoch_loss,
                f"{phase.capitalize()} Acc":  epoch_acc,
                f"{phase.capitalize()} F1":   epoch_f1,
                f"{phase.capitalize()} Prec": epoch_prec,
                f"{phase.capitalize()} Rec":  epoch_rec
            })

            print(f"{phase.capitalize()} | Loss: {epoch_loss:.4f} "
                  f"Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} "
                  f"Prec: {epoch_prec:.4f} Rec: {epoch_rec:.4f}")

            if phase == "test":
                # ensure plots dir
                os.makedirs(PLOTS_DIR, exist_ok=True)

                # Confusion Matrix
                cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
                print("Confusion Matrix:\n", cm)
                disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","Malignant"])
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(os.path.join(PLOTS_DIR, f"conf_mat_epoch_{epoch+1}.png"), dpi=300)
                wandb.log({f"Conf Mat Epoch {epoch+1}": wandb.Image(os.path.join(PLOTS_DIR, f"conf_mat_epoch_{epoch+1}.png"))})
                plt.close()

                # ROC & F1-confidence
                save_roc_f1_curves(all_labels, all_probs, epoch+1, out_dir=PLOTS_DIR)
                wandb.log({
                    f"ROC Curve Epoch {epoch+1}":     wandb.Image(os.path.join(PLOTS_DIR, f"roc_curve_epoch_{epoch+1}.png")),
                    f"F1-Confidence Epoch {epoch+1}": wandb.Image(os.path.join(PLOTS_DIR, f"f1_conf_curve_epoch_{epoch+1}.png"))
                })

                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    best_epoch    = epoch+1
                    best_cm       = cm

        # save to CSV
        results.append([
            epoch+1,
            # train metrics
            epoch_loss if phase=="train" else None,
            epoch_acc  if phase=="train" else None,
            epoch_f1   if phase=="train" else None,
            epoch_prec if phase=="train" else None,
            epoch_rec  if phase=="train" else None,
            # test metrics
            epoch_loss if phase=="test" else None,
            epoch_acc  if phase=="test" else None,
            epoch_f1   if phase=="test" else None,
            epoch_prec if phase=="test" else None,
            epoch_rec  if phase=="test" else None,
        ])
        save_results_to_csv(results, f"results_seed_{best_epoch}.csv")

        if early_stopping(epoch_loss if phase=="test" else None, model):
            break

    print(f"\nBest Test Acc: {best_test_acc:.4f} at Epoch {best_epoch}")
    print("Best Confusion Matrix:\n", best_cm)
    return best_test_acc, best_epoch, best_cm

# ----------------------------
# Main: run across seeds
# ----------------------------
best_overall_acc  = 0.0
best_overall_seed = None
seed_results      = []

for seed in args.seeds:
    print(f"\n=== Seed {seed} ===")
    set_seed(seed)
    model     = CustomModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

    acc, epoch, cm = train_and_evaluate_model(
        model, dataloaders, criterion, optimizer, num_epochs=args.epochs
    )
    seed_results.append((seed, acc))
    if acc > best_overall_acc:
        best_overall_acc  = acc
        best_overall_seed = seed

print(f"\nOverall Best Seed: {best_overall_seed} with Acc {best_overall_acc:.4f}")
print("Seed-wise results:")
for s, a in seed_results:
    print(f"  Seed {s}: Acc = {a:.4f}")
