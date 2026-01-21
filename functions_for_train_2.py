import numpy as np
import torch
import os
import pandas as pd # (save_results_to_csv를 위해)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path  # <--- 이 부분이 핵심입니다.

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- (파일에 다른 함수들이 있다면 그대로 둡니다) ---

def get_unique_filename(directory, basename, extension):
    """지정된 디렉토리에서 중복되지 않는 파일 이름을 생성합니다."""
    counter = 1
    filename = f"{basename}{extension}"
    filepath = os.path.join(directory, filename)
    
    while os.path.exists(filepath):
        filename = f"{basename}_{counter}{extension}"
        filepath = os.path.join(directory, filename)
        counter += 1
    return filepath

def save_results_to_csv(results, filename):
    """결과 리스트를 DataFrame으로 변환하여 CSV 파일로 저장합니다."""
    columns = [
        'Epoch', 
        'Train Loss', 'Train Acc', 'Train F1', 'Train Precision', 'Train Recall',
        'Test Loss', 'Test Acc', 'Test F1', 'Test Precision', 'Test Recall'
    ]
    # 결과 길이에 맞게 컬럼 조정
    if results and len(results[0]) != len(columns):
        # K-Fold 결과용 (Iteration, Fold, Acc, F1, P, R)
        columns = ["Iteration", "Fold", "Accuracy", "F1_Score", "Precision", "Recall"]
        if len(results[0]) != len(columns):
             # Epoch별 (Epoch, Train Loss, Train Acc, Val Loss, Val Acc)
             # 필요한 경우 여기에 다른 형식의 컬럼을 추가
             columns = columns[:len(results[0])] # 임시 조치
             
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# ... (find_image_sizes, calculate_optimal_size 등 다른 함수들) ...