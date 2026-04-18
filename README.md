# Anti_Spoofing_FaceRec

# 🛡️ Face Anti-Spoofing: Ensemble Deep Learning Solution

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/Library-timm-FF6F00)

Repository ini berisi solusi untuk kompetisi **Face Anti-Spoofing**. Proyek ini bertujuan untuk membedakan antara wajah asli (*live*) dan serangan pemalsuan (*spoof*) seperti foto cetak, tampilan layar elektronik, atau masker, menggunakan pendekatan **Ensemble Model** dari tiga arsitektur CNN modern.

## 🌟 Ringkasan Teknik
Solusi ini menggabungkan kekuatan tiga model *State-of-the-Art* dengan strategi training yang berbeda untuk mendapatkan generalisasi terbaik:
- **ResNet50 & EfficientNetV2-S**: Fokus pada stabilitas dan ekstraksi fitur yang efisien.
- **ConvNeXt-Small**: Arsitektur modern yang dioptimalkan untuk menangkap pola tekstur halus (seperti moiré atau serat kertas).

## 📂 Struktur Proyek & Workflow

### 1. `FINAL_RESNET.ipynb` (Stage 1: Fine-Tuning)
Notebook ini menangani pelatihan dua model pertama dengan teknik:
* **Progressive Unfreezing**: Melatih model secara bertahap dari *head* hingga ke seluruh *backbone*.
* **Progressive Resizing**: Training dimulai dari resolusi 128x128 kemudian ditingkatkan ke 256x256 untuk stabilitas konvergensi.
* **Fixes**: Penanganan bug pada *unfreeze schedule* dan optimasi *early stopping*.

### 2. `face_antispoofing_v4_convnext.ipynb` (Stage 2: Advanced Training)
Fokus pada model **ConvNeXt-Small** dengan fitur tambahan:
* **SAM (Sharpness-Aware Minimization)**: Optimizer untuk mencari *flat minima* guna mencegah overfitting pada dataset kecil.
* **Triple Pooling**: Menggabungkan GAP (Global Average), GMP (Global Max), dan GeM (Generalized Mean) Pooling.
* **EMA (Exponential Moving Average)**: Menghaluskan bobot model agar lebih robust terhadap noise selama training.

### 3. `ensemble_inference_only.py` (Stage 3: Production/Submission)
Skrip final untuk melakukan prediksi pada data test:
* **Weighted Ensemble**: Menggabungkan output ketiga model dengan bobot tertentu.
* **5-Pass TTA (Test Time Augmentation)**: Melakukan augmentasi (flip, rotasi, dll) saat inference untuk mengurangi varians error.

## 🛠️ Tech Stack & Library
- **Framework**: PyTorch
- **Model Library**: `timm` (PyTorch Image Models)
- **Optimizer**: AdamW & SAM
- **Augmentation**: RandAugment, Mixup, CutMix
- **Metrics**: Accuracy, F1-Score, Confusion Matrix
