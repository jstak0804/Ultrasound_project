import time
import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import datetime
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torchmetrics.classification import F1Score, Precision, Recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# [중요] model.py가 같은 폴더에 있어야 합니다.
from model import CustomModel 

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU Cache 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 1. Helper Functions & Classes
# ---------------------------------------------------------

def set_seed(random_seed):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def create_result_dir(model_name, base_dir="Result"):
    """결과 저장 폴더 생성"""
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    result_dir = os.path.join(base_dir, date_str, model_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def save_results_to_csv(results, filename, columns=None):
    """학습 로그 CSV 저장"""
    if columns is None:
        columns = [
            "Epoch", 
            "Train Loss", "Train Accuracy", "Train F1", "Train Precision", "Train Recall",
            "Val Loss", "Val Accuracy", "Val F1", "Val Precision", "Val Recall"
        ]
    results_df = pd.DataFrame(results, columns=columns)
    if os.path.exists(filename):
        results_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        results_df.to_csv(filename, index=False)

# ---------------------------------------------------------
# 2. EarlyStopping Class (직접 정의)
# ---------------------------------------------------------
class EarlyStopping:
    """Validation Loss가 더 이상 줄어들지 않으면 학습 중단"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0 # 개선됨 -> 초기화

# ---------------------------------------------------------
# 3. Grad-CAM Class & Visualization
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_forward_hook(self.save_gradient_hook)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient_hook(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            self.gradients = grad
        output.register_hook(_store_grad)

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        targets = torch.argmax(output, dim=1)
            
        one_hot_output = torch.zeros_like(output)
        for i in range(len(targets)):
            one_hot_output[i][targets[i]] = 1
            
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        
        cam = cam.detach().cpu().numpy()
        result_cams = []
        for i in range(len(cam)):
            c = cam[i, 0, :, :]
            if np.max(c) > np.min(c):
                c = (c - np.min(c)) / (np.max(c) - np.min(c))
            else:
                c = np.zeros_like(c)
            result_cams.append(c)
        return np.array(result_cams)

def show_cam_on_image(img, mask):
    """히트맵 시각화"""
    heatmap = plt.get_cmap('jet')(mask)[..., :3]
    heatmap = np.float32(heatmap)
    cam = 0.5 * heatmap + 0.5 * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# ---------------------------------------------------------
# 4. Main Training Function
# ---------------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, seed, result_dir, num_epochs=25, num_classes=2):
    results = []
    # Patience=5 설정 (5번 개선 안되면 중단)
    early_stopping = EarlyStopping(patience=5, verbose=True) 
    
    best_val_acc = 0.0
    best_epoch_acc = 0
    best_model_path = os.path.join(result_dir, f"best_model_seed_{seed}.pth")

    for epoch in range(num_epochs):
        epoch_metrics = {}
        
        for phase in ['train', 'val']: 
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            correct_predictions = 0
            
            # Metrics Init
            metric_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device)
            metric_precision = Precision(num_classes=num_classes, average='macro', task='multiclass').to(device)
            metric_recall = Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)
            
            # TQDM Progress Bar
            pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]", leave=True)
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Metrics Update
                    metric_f1.update(outputs, labels)
                    metric_precision.update(outputs, labels)
                    metric_recall.update(outputs, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
                    correct_predictions += torch.sum(preds == labels.data).item()
                    
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Epoch Stats
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct_predictions / len(dataloaders[phase].dataset)
            epoch_f1 = metric_f1.compute().item()
            epoch_precision = metric_precision.compute().item()
            epoch_recall = metric_recall.compute().item()

            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc
            epoch_metrics[f'{phase}_f1'] = epoch_f1
            epoch_metrics[f'{phase}_precision'] = epoch_precision
            epoch_metrics[f'{phase}_recall'] = epoch_recall

            if phase == 'val':
                print(f" -> Val Acc: {epoch_acc:.4f} | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f}")
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_epoch_acc = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
                    print(f"    (Best Model Saved!)")
            
            wandb.log({
                f"{phase} loss": epoch_loss, 
                f"{phase} acc": epoch_acc, 
                "epoch": epoch+1
            })

        # Early Stopping Check
        early_stopping(epoch_metrics['val_loss'], model)
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping triggered at Epoch {epoch+1}")
            break

        # Save Logs (11 Columns)
        results.append([
            epoch + 1,
            epoch_metrics['train_loss'], epoch_metrics['train_acc'], epoch_metrics['train_f1'], epoch_metrics['train_precision'], epoch_metrics['train_recall'],
            epoch_metrics['val_loss'], epoch_metrics['val_acc'], epoch_metrics['val_f1'], epoch_metrics['val_precision'], epoch_metrics['val_recall']
        ])
        save_results_to_csv(results, os.path.join(result_dir, f"training_log_seed_{seed}.csv"))

    print(f"\nTraining Finished. Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch_acc}")
    return best_model_path, best_epoch_acc

# ---------------------------------------------------------
# 5. Final Evaluation Function
# ---------------------------------------------------------
def run_final_evaluation(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss, correct_predictions = 0.0, 0
    f1_metric = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device)
    precision_metric = Precision(num_classes=num_classes, average='macro', task='multiclass').to(device)
    recall_metric = Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Final Test Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            f1_metric.update(outputs, labels)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data).item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = correct_predictions / len(dataloader.dataset)
    test_f1 = f1_metric.compute().item()
    test_precision = precision_metric.compute().item()
    test_recall = recall_metric.compute().item()
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return test_loss, test_acc, test_f1, test_precision, test_recall, cm

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seeds', nargs='+', type=int, default=[24])
    args = parser.parse_args()

    # WandB
    wandb.init(project="US_classification_Final_Code", entity="ddurbozak")
    wandb.config.update(args)

    model_name = "CustomModel_Final"
    result_dir = create_result_dir(model_name)
    print(f"Results will be saved in: {result_dir}")

    # Data Paths (경로 확인 필수)
    train_path = "train_clean"
    val_path = "val_clean"
    test_path = "test_clean"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\nLoading Datasets...")
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names} (N={num_classes})")

    # Loaders
    bs = args.batch_size if args.batch_size != -1 else 32
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    
    dataloaders = {'train': train_loader, 'val': val_loader}

    seed_results = []

    # --- Seed Loop ---
    for seed in args.seeds:
        print(f"\n{'='*40}\n STARTING SEED: {seed}\n{'='*40}")
        set_seed(seed)
        
        # Init Model
        model = CustomModel(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.NAdam(model.parameters(), lr=1e-5)

        # 1. Train
        best_model_path, best_epoch = train_model(
            model, dataloaders, criterion, optimizer, seed, result_dir, 
            num_epochs=args.epochs, num_classes=num_classes
        )

        # 2. Final Test
        if os.path.exists(best_model_path):
            print(f"\n[Test] Loading Best Model (Seed {seed})...")
            model.load_state_dict(torch.load(best_model_path))
            
            t_loss, t_acc, t_f1, t_prec, t_rec, t_cm = run_final_evaluation(
                model, test_loader, criterion, device, num_classes
            )
            
            seed_results.append([seed, best_epoch, t_loss, t_acc, t_f1, t_prec, t_rec])

            # 3. Save Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=t_cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
            plt.title(f"Confusion Matrix (Seed {seed})")
            cm_path = os.path.join(result_dir, f"CM_Seed_{seed}.png")
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close()
            wandb.log({f"CM Seed {seed}": wandb.Image(cm_path)})

            # 4. Grad-CAM
            print(f"[Grad-CAM] Generating images...")
            cam_dir = os.path.join(result_dir, f"GradCAM_Seed_{seed}")
            os.makedirs(cam_dir, exist_ok=True)
            
            # Target Layer 자동 탐지 (CustomModel 구조에 맞춰 조정 필요)
            target_layer = model.conv2d if hasattr(model, 'conv2d') else model.backbone[-1]
            gradcam = GradCAM(model, target_layer)
            
            cnt = 0
            for inputs, labels in test_loader:
                if cnt >= 20: break
                inputs = inputs.to(device)
                cams = gradcam(inputs)
                
                for i in range(len(inputs)):
                    if cnt >= 20: break
                    # Denormalize
                    img = inputs[i].cpu().permute(1, 2, 0).numpy()
                    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    
                    vis = show_cam_on_image(img, cams[i])
                    
                    gt = class_names[labels[i].item()]
                    pred = class_names[model(inputs[i:i+1]).argmax(1).item()]
                    fname = os.path.join(cam_dir, f"img{cnt}_GT-{gt}_Pred-{pred}.jpg")
                    plt.imsave(fname, vis)
                    cnt += 1

            del model
            torch.cuda.empty_cache()
        else:
            print("Error: Best model not found.")

    # 5. Final Summary
    print(f"\n{'='*40}\n FINAL RESULTS SUMMARY \n{'='*40}")
    df = pd.DataFrame(seed_results, columns=["Seed", "Best Epoch", "Loss", "Acc", "F1", "Prec", "Rec"])
    print(df.to_string(index=False))
    
    summary_path = os.path.join(result_dir, "final_summary_results.csv")
    df.to_csv(summary_path, index=False)
    wandb.save(summary_path)
    wandb.finish()