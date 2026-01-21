import os
from torchvision import datasets, transforms
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np


def find_image_sizes(root):
    widths, heights = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif')):
                img_path = os.path.join(dirpath, filename)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}")
    return widths, heights

def save_results_to_csv(results, path):
    """
    Save training results to a CSV file.

    Args:
        results (list): List of results to be saved.
        path (str): Path to the CSV file.
    """
    results_df = pd.DataFrame(results, columns=[
    "Epoch", "Train Loss", "Train Accuracy", "Train F1", "Train Precision", "Train Recall",
    "Test Loss", "Test Accuracy", "Test F1", "Test Precision", "Test Recall"
])


    results_df.to_csv(path, index=False)
    print(f'Results saved to {path}')


def get_unique_filename(filepath):
    """
    Check if the file exists and return a unique filename by appending a number.
    """
    if not os.path.isfile(filepath):
        return filepath

    base, extension = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{extension}"

    while os.path.isfile(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{extension}"

    return new_filepath
def calculate_optimal_size(widths, heights):
    min_width, max_width = min(widths), max(widths)
    min_height, max_height = min(heights), max(heights)
    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)

    optimal_width = round(avg_width)
    optimal_height = round(avg_height)

    return (min_width, min_height), (max_width, max_height), (optimal_width, optimal_height)
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 모델의 개선이 없더라도 기다릴 에포크 수
            verbose (bool): 개선이 이루어질 때마다 메시지를 출력할지 여부
            delta (float): 개선으로 간주하기 위한 최소 변화량
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

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
        '''Validation loss가 감소했을 때 모델을 저장'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Patience counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

