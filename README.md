# Liver Ultrasound Classification Project (Ultrasound_project)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-yellow.svg)](https://wandb.ai/)

This project focuses on the **automated classification of liver tumors into benign and malignant categories** using deep learning techniques on ultrasound imagery. The core of this research lies in implementing a custom attention mechanism to capture the subtle textures and patterns characteristic of medical ultrasound data.

간 초음파 영상을 분석하여 종양의 **양성(Benign) 및 악성(Malignant) 여부를 분류**하는 딥러닝 프로젝트입니다. 의료 영상의 미세한 특징을 효과적으로 포착하기 위해 설계된 커스텀 어텐션 구조와 최신 의료 인공지능 프레임워크를 활용합니다.

---

## 📌 Model Architecture (`custommodel`)

The `custommodel` in `model.py` is specifically designed to handle the complexity of medical images through a refined feature-refinement process.



* **Backbone**: Utilizes a pre-trained **ResNet-50** for robust initial feature extraction.
* **Feature Compression**: The 2048-channel output from the backbone is compressed to 1024 channels via a 3x3 convolution to improve computational efficiency.
* **Parallel Attention Structure**:
    * **Branch A**: Sequential processing through a **SEBlock** followed by **ChannelAttention**.
    * **Branch B**: Sequential processing through **ChannelAttention** followed by a **SEBlock**.
    * The outputs from both branches are **concatenated**, restoring the feature map to 2048 channels.
* **Spatial Refinement**: A **SpatialAttention** module is applied to the concatenated features to emphasize relevant spatial regions (e.g., tumor boundaries).
* **Final Output**: Global Average Pooling (GAP) followed by a Linear layer to produce the final classification logic.

---

## 📂 Project Structure

| File Name | Description |
| :--- | :--- |
| **`Liver_train.py`** | Main execution script for the training pipeline. |
| **`model.py`** | Definition of the `custommodel` and related neural network blocks. |
| **`base_line.py`** | Baseline training script for initial performance benchmarking. |
| **`train_binary2.py`** | Optimized training script specifically for binary classification tasks. |
| **`functions_for_train.py`** | Utility functions for data preprocessing, loss calculation, and metrics. |

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/jstak0804/Ultrasound_project.git](https://github.com/jstak0804/Ultrasound_project.git)
cd Ultrasound_project
pip install -r requirements.txt
