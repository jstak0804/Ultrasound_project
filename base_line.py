import time
import argparse
import os
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import datetime
import warnings
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc  # [수정] 가비지 컬렉션 모듈 추가
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchmetrics.classification import F1Score, Precision, Recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CustomModel import
try:
    from model import CustomModel
except ImportError:
    CustomModel = None
    print("[Info] 'model.py' not found. 'custom_model' will be skipped if selected.")

# 경고 무시
warnings.filterwarnings("ignore")

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU Cache 초기 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

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
# 2. EarlyStopping
# ---------------------------------------------------------
class EarlyStopping:
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
            self.counter = 0

# ---------------------------------------------------------
# 3. Grad-CAM (CNN 기반)
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
    heatmap = plt.get_cmap('jet')(mask)[..., :3]
    heatmap = np.float32(heatmap)
    cam = 0.5 * heatmap + 0.5 * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# ---------------------------------------------------------
# 4. Model Loader (Pretrained=False 적용)
# ---------------------------------------------------------
def get_model(model_name, num_classes, pretrained=False):
    model = None
    target_layer = None
    name = model_name.lower()
    
    # weights=None -> Random Initialization (가중치 초기화됨)
    
    if name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        target_layer = model.features[-3]
        
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1]
        
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1]
        
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        target_layer = model.features[-1]
        
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        target_layer = model.features.denseblock4 
    
    elif name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        target_layer = model.features[-1] 

    elif name == "custom_model":
        if CustomModel is None: raise ValueError("CustomModel class not found.")
        model = CustomModel(num_classes=num_classes)
        target_layer = model.conv2d if hasattr(model, 'conv2d') else list(model.children())[-2]

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model, target_layer

# ---------------------------------------------------------
# 5. Integrated Train Function
# ---------------------------------------------------------
def train_and_evaluate(model, dataloaders, criterion, optimizer, num_epochs, result_dir, num_classes, seed):
    early_stopping = EarlyStopping(patience=7, verbose=True)
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(result_dir, f"best_model.pth")
    results = []
    
    metrics = {
        'f1': F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device),
        'prec': Precision(num_classes=num_classes, average='macro', task='multiclass').to(device),
        'rec': Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)
    }

    for epoch in range(num_epochs):
        epoch_stats = {}
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            
            running_loss = 0.0
            correct = 0
            for m in metrics.values(): m.reset()
            
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
                        
                    for m in metrics.values(): m.update(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    correct += torch.sum(preds == labels.data).item()
                    
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct / len(dataloaders[phase].dataset)
            epoch_f1 = metrics['f1'].compute().item()
            epoch_prec = metrics['prec'].compute().item()
            epoch_rec = metrics['rec'].compute().item()
            
            epoch_stats[f'{phase}_loss'] = epoch_loss
            epoch_stats[f'{phase}_acc'] = epoch_acc
            epoch_stats[f'{phase}_f1'] = epoch_f1
            epoch_stats[f'{phase}_precision'] = epoch_prec
            epoch_stats[f'{phase}_recall'] = epoch_rec

            if phase == 'val':
                print(f" -> Val Acc: {epoch_acc:.4f} | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f}")
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
                    print("    (Best Model Saved!)")
            
            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc, "epoch": epoch+1})

        early_stopping(epoch_stats['val_loss'], model)
        
        results.append([
            epoch + 1,
            epoch_stats['train_loss'], epoch_stats['train_acc'], epoch_stats['train_f1'], epoch_stats['train_precision'], epoch_stats['train_recall'],
            epoch_stats['val_loss'], epoch_stats['val_acc'], epoch_stats['val_f1'], epoch_stats['val_precision'], epoch_stats['val_recall']
        ])
        save_results_to_csv(results, os.path.join(result_dir, f"training_log.csv"))

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    return best_model_path, best_epoch

# ---------------------------------------------------------
# 6. Final Test Function
# ---------------------------------------------------------
def run_final_test(model, dataloader, criterion, num_classes):
    model.eval()
    running_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    
    f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device)
    prec = Precision(num_classes=num_classes, average='macro', task='multiclass').to(device)
    rec = Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Final Test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            f1.update(outputs, labels)
            prec.update(outputs, labels)
            rec.update(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (running_loss / len(dataloader.dataset), 
            correct / len(dataloader.dataset), 
            f1.compute().item(), 
            prec.compute().item(), 
            rec.compute().item(),
            confusion_matrix(all_labels, all_preds, labels=range(num_classes)))

# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    default_models = [
        
        "custom_model"
    ]
    
    if CustomModel is None and "custom_model" in default_models:
        default_models.remove("custom_model")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--models', nargs='+', default=default_models, help="Models to train")
    
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Data Loading
    train_path = "train_clean"
    val_path = "val_clean"
    test_path = "test_clean"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Data...")
    try:
        train_ds = datasets.ImageFolder(train_path, transform=train_transform)
        val_ds = datasets.ImageFolder(val_path, transform=val_test_transform)
        test_ds = datasets.ImageFolder(test_path, transform=val_test_transform)
        
        dataloaders = {
            'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        class_names = train_ds.classes
        num_classes = len(class_names)
        print(f"Classes: {class_names}, Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    overall_results = []

    print(f"Selected Models (Pretrained=False): {args.models}")
    
    # 2. Iterate Over Models
    for model_name in args.models:
        print(f"\n\n{'='*50}\n STARTING MODEL: {model_name} (Random Init)\n{'='*50}")
        
        # WandB 초기화
        wandb.init(project="US_Model_Comparison_NoPretrain", entity="ddurbozak", name=f"{model_name}_Seed{args.seed}", reinit=True)
        wandb.config.update(args)
        
        result_dir = create_result_dir(model_name + "_NoPretrain")
        
        try:
            # Model Instantiation (여기서 새로운 객체 생성 = 가중치 초기화)
            model, target_layer = get_model(model_name, num_classes, pretrained=False)
            model = model.to(device)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            wandb.finish()
            continue

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.NAdam(model.parameters(), lr=1e-5)

        # Train
        best_model_path, best_epoch = train_and_evaluate(
            model, dataloaders, criterion, optimizer, args.epochs, result_dir, num_classes, args.seed
        )

        # Final Test
        if os.path.exists(best_model_path):
            print(f"[Test] Loading best model for {model_name}...")
            model.load_state_dict(torch.load(best_model_path))
            
            t_loss, t_acc, t_f1, t_prec, t_rec, cm = run_final_test(model, test_loader, criterion, num_classes)
            
            overall_results.append({
                "Model": model_name,
                "Best Epoch": best_epoch,
                "Test Acc": t_acc,
                "Test F1": t_f1,
                "Test Prec": t_prec,
                "Test Rec": t_rec
            })

            # Save Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
            plt.title(f"CM: {model_name} (No Pretrain)")
            cm_path = os.path.join(result_dir, "confusion_matrix.png")
            plt.savefig(cm_path, bbox_inches='tight')
            wandb.log({"Confusion Matrix": wandb.Image(cm_path)})
            plt.close()

            # Run Grad-CAM
            print("[Grad-CAM] Generating images...")
            try:
                cam_dir = os.path.join(result_dir, "GradCAM")
                os.makedirs(cam_dir, exist_ok=True)
                gradcam = GradCAM(model, target_layer)
                
                cnt = 0
                for inputs, labels in test_loader:
                    if cnt >= 4: break
                    inputs = inputs.to(device)
                    cams = gradcam(inputs)
                    
                    for i in range(len(inputs)):
                        if cnt >= 4: break
                        img = inputs[i].cpu().permute(1, 2, 0).numpy()
                        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                        img = np.clip(img, 0, 1)
                        vis = show_cam_on_image(img, cams[i])
                        
                        fname = os.path.join(cam_dir, f"img{cnt}_{model_name}.jpg")
                        plt.imsave(fname, vis)
                        wandb.log({f"GradCAM_{cnt}": wandb.Image(vis)})
                        cnt += 1
            except Exception as e:
                print(f"[Warning] Grad-CAM failed for {model_name}: {e}")
        else:
            print("Best model not found.")

        wandb.finish()

        # [수정된 부분] 메모리 및 가중치 초기화 (다음 루프를 위해 정리)
        # model 객체 삭제 및 가비지 컬렉션 수행으로 GPU 메모리 확보
        del model
        del optimizer
        del criterion
        if 'gradcam' in locals(): 
            del gradcam
        
        gc.collect()             # Python 메모리 쓰레기 수집
        torch.cuda.empty_cache() # GPU 캐시 비우기
        print(f"[Info] Memory cleared for next model.\n")

    # 3. Final Output (Table)
    print(f"\n\n{'='*60}")
    print(" FINAL COMPARISON RESULTS (No Pretraining) ")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(overall_results)
    
    if not summary_df.empty:
        format_mapping = {col: "{:.4f}" for col in ["Test Acc", "Test F1", "Test Prec", "Test Rec"]}
        for col, fmt in format_mapping.items():
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(lambda x: fmt.format(x))
            
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join("Result", f"Comparison_NoPretrain_Summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("No results to show.")