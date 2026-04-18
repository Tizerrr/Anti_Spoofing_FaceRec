"""
Face Anti-Spoofing v5 — Ensemble Inference Only
Gunakan ini jika sudah punya .pth dari ResNet50, EfficientNetV2-S, dan ConvNeXt-Small.
Tidak ada training, langsung load → evaluasi val → inference test set.
"""

# !pip install timm --quiet

import random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ─────────────────────────────────────────────────────────────
# KONFIGURASI — sesuaikan path .pth kamu di sini
# ─────────────────────────────────────────────────────────────
TRAIN_DIR = Path('dataset/train')
TEST_DIR  = Path('dataset/test')
OUTPUT_CSV = Path('submission_v5_ensemble_2.csv')

# Path ke checkpoint yang sudah ada
# Untuk ConvNeXt: isi CKPT_CONVNXT dengan path terbaik (standard atau EMA)
CKPT_RESNET  = Path('NEWbest_resnet50_v3.pth')        # ← ganti sesuai nama file kamu
CKPT_EFFNET  = Path('best_efficientnet_v3.pth')    # ← ganti sesuai nama file kamu
CKPT_CONVNXT = Path('best_convnext_v4.pth')    # ← ganti sesuai nama file kamu

# Backbone ConvNeXt yang dipakai saat training
CONVNXT_BACKBONE = 'convnext_small.fb_in22k_ft_in1k'

CLASS_NAMES = [
    'realperson', 'fake_printed', 'fake_screen',
    'fake_mask', 'fake_mannequin', 'fake_unknown'
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX2CLASS   = {i: c for c, i in CLASS2IDX.items()}

BATCH_SIZE  = 32
NUM_WORKERS = 0
VAL_SPLIT   = 0.2   # harus sama dengan saat training agar val set identik


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples; self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img) if self.transform else img, label

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.paths = sorted([p for p in Path(test_dir).iterdir()
                             if p.suffix.lower() in ['.jpg','.jpeg','.png']])
        self.ids   = [p.stem for p in self.paths]
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img) if self.transform else img, self.ids[idx]

def load_samples(train_dir):
    samples = []
    for cls in CLASS_NAMES:
        d = Path(train_dir) / cls
        if not d.exists(): print(f'⚠ skip {d}'); continue
        for ext in ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG']:
            for p in d.glob(ext):
                samples.append((str(p), CLASS2IDX[cls]))
    return samples

# Recreate val split dengan seed yang sama persis seperti saat training
all_samples = load_samples(TRAIN_DIR)
paths_all   = [s[0] for s in all_samples]
labels_all  = [s[1] for s in all_samples]

_, paths_va, _, labels_va = train_test_split(
    paths_all, labels_all, test_size=VAL_SPLIT,
    random_state=SEED, stratify=labels_all
)
val_samples = list(zip(paths_va, labels_va))
print(f'Val samples: {len(val_samples)} (stratified, seed={SEED})')


# ─────────────────────────────────────────────────────────────
# Transforms — TTA deterministik (5 pass)
# ─────────────────────────────────────────────────────────────
def get_val_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def get_tta_transforms(img_size=224):
    norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    sz2  = int(img_size * 1.15)
    return [
        T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm]),
        T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose([T.Resize((sz2, sz2)), T.CenterCrop(img_size), T.ToTensor(), norm]),
        T.Compose([T.Resize((sz2, sz2)), T.CenterCrop(img_size), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose([T.Resize((img_size, img_size)), T.RandomRotation((90, 90)), T.ToTensor(), norm]),
    ]


# ─────────────────────────────────────────────────────────────
# Arsitektur model (harus identik dengan saat training)
# ─────────────────────────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p)

class SpoofingHead(nn.Module):
    def __init__(self, in_features, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.net(x)

class SpoofingHeadLN(nn.Module):
    def __init__(self, in_features, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.net(x)

class ResNet50Spoof(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gem = GeM()
        self.head = SpoofingHead(bb.fc.in_features * 3, num_classes)
    def forward(self, x):
        f = self.backbone(x)
        return self.head(torch.cat([self.gap(f), self.gmp(f), self.gem(f)], dim=1))

class EfficientNetV2Spoof(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=False,
            num_classes=0, global_pool='')
        in_feat = self.backbone.num_features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gem = GeM()
        self.head = SpoofingHead(in_feat * 3, num_classes)
    def forward(self, x):
        f = self.backbone(x)
        return self.head(torch.cat([self.gap(f), self.gmp(f), self.gem(f)], dim=1))

class ConvNeXtSpoof(nn.Module):
    def __init__(self, backbone_name=CONVNXT_BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        in_feat = self.backbone.num_features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gem = GeM()

        # ⬇️ LANGSUNG Sequential, jangan pakai SpoofingHeadLN wrapper
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_feat * 3),
            nn.Linear(in_feat * 3, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        x = torch.cat([self.gap(f), self.gmp(f), self.gem(f)], dim=1)
        return self.head(x)
    
# ─────────────────────────────────────────────────────────────
# Load checkpoint — otomatis handle format berbeda
# ─────────────────────────────────────────────────────────────
def load_checkpoint(model, ckpt_path):
    """
    Mendukung dua format checkpoint:
    - Dict dengan key 'model_state_dict' (format training kita)
    - Raw state_dict langsung
    Juga otomatis handle prefix 'module.' dari DataParallel.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        meta  = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        meta  = {}
    else:
        state = ckpt
        meta  = {}

    # Hapus prefix 'module.' jika model disimpan dari DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return meta

print('\nMemuat checkpoint...')
model_resnet  = ResNet50Spoof().to(DEVICE)
model_effnet  = EfficientNetV2Spoof().to(DEVICE)
model_convnxt = ConvNeXtSpoof().to(DEVICE)

meta_r = load_checkpoint(model_resnet,  CKPT_RESNET)
meta_e = load_checkpoint(model_effnet,  CKPT_EFFNET)
meta_c = load_checkpoint(model_convnxt, CKPT_CONVNXT)

print(f'ResNet50       loaded  | {meta_r}')
print(f'EfficientNetV2 loaded  | {meta_e}')
print(f'ConvNeXt       loaded  | {meta_c}')


# ─────────────────────────────────────────────────────────────
# Evaluasi di val set → tentukan bobot ensemble
# ─────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_p, all_l, probs_list = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, leave=False):
            probs = torch.softmax(model(imgs.to(DEVICE)), 1)
            probs_list.append(probs.cpu().numpy())
            all_p.extend(probs.argmax(1).cpu().numpy())
            all_l.extend(labels.numpy())
    probs_arr = np.concatenate(probs_list)
    f1  = f1_score(all_l, all_p, average='macro', zero_division=0)
    acc = accuracy_score(all_l, all_p)
    return all_l, all_p, probs_arr, f1, acc

val_ds  = FaceDataset(val_samples, get_val_transform(224))
val_ldr = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print('\nEvaluasi di val set...')
gt, preds_r, probs_r, f1_r, acc_r = evaluate(model_resnet,  val_ldr)
gt, preds_e, probs_e, f1_e, acc_e = evaluate(model_effnet,  val_ldr)
gt, preds_c, probs_c, f1_c, acc_c = evaluate(model_convnxt, val_ldr)

w_r, w_e, w_c = f1_r, f1_e, f1_c
probs_ens = (w_r*probs_r + w_e*probs_e + w_c*probs_c) / (w_r + w_e + w_c)
preds_ens = probs_ens.argmax(1)
f1_ens    = f1_score(gt, preds_ens, average='macro', zero_division=0)
acc_ens   = accuracy_score(gt, preds_ens)

print('\n' + '='*65)
print(f'  ResNet50         : Acc={acc_r:.4f}  F1={f1_r:.4f}  w={w_r:.4f}')
print(f'  EfficientNetV2-S : Acc={acc_e:.4f}  F1={f1_e:.4f}  w={w_e:.4f}')
print(f'  ConvNeXt-Small   : Acc={acc_c:.4f}  F1={f1_c:.4f}  w={w_c:.4f}')
print(f'  ── Triple Ensemble: Acc={acc_ens:.4f}  F1={f1_ens:.4f} ⭐')
print('='*65)

print('\n📋 Classification Report — Triple Ensemble')
print(classification_report(gt, preds_ens, target_names=CLASS_NAMES, digits=4))

# Confusion matrix 4 panel
fig, axes = plt.subplots(1, 4, figsize=(28, 6))
fig.suptitle('Confusion Matrix — Triple Ensemble', fontsize=14, fontweight='bold')
for ax, (title, preds) in zip(axes, [
    ('ResNet50',          preds_r),
    ('EfficientNetV2-S',  preds_e),
    ('ConvNeXt-Small',    preds_c),
    ('Triple Ensemble ⭐', preds_ens),
]):
    cm = confusion_matrix(gt, preds).astype(float)
    cm /= cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap='Blues', ax=ax, linewidths=0.5)
    f1_  = f1_score(gt, preds, average='macro', zero_division=0)
    acc_ = accuracy_score(gt, preds)
    ax.set_title(f'{title}\nAcc={acc_:.4f}  F1={f1_:.4f}', fontsize=10, fontweight='bold')
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('confusion_matrix_ensemble.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────────────
# Inference Test Set — TTA × 3 model
# ─────────────────────────────────────────────────────────────
def predict_ensemble_tta(models_weights, test_dir, img_size=224):
    """5-pass TTA × 3 model = 15 forward passes per sample."""
    tta_tfms  = get_tta_transforms(img_size)
    avg_probs = None
    all_ids   = None
    total_w   = 0.0

    for t_idx, tfm in enumerate(tta_tfms):
        ds     = TestDataset(test_dir, transform=tfm)
        loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        for model, w, name in models_weights:
            model.eval()
            print(f'  TTA {t_idx+1}/{len(tta_tfms)} × {name}...')
            p_list, id_list = [], []
            with torch.no_grad():
                for imgs, ids in tqdm(loader, leave=False):
                    p = torch.softmax(model(imgs.to(DEVICE)), 1).cpu().numpy()
                    p_list.append(p); id_list.extend(ids)
            p_arr = np.concatenate(p_list) * w
            avg_probs = p_arr if avg_probs is None else avg_probs + p_arr
            all_ids   = id_list
            total_w  += w

    avg_probs  /= total_w
    pred_labels = [IDX2CLASS[i] for i in avg_probs.argmax(1)]
    return all_ids, pred_labels, avg_probs

print('\n🔍 Inference test set (5 TTA × 3 model = 15 passes)...')
models_weights = [
    (model_resnet,  w_r, 'ResNet50'),
    (model_effnet,  w_e, 'EfficientNetV2-S'),
    (model_convnxt, w_c, 'ConvNeXt-Small'),
]
test_ids, test_preds, test_probs = predict_ensemble_tta(models_weights, TEST_DIR)

submission_df = pd.DataFrame({'id': test_ids, 'label': test_preds})
submission_df = submission_df.sort_values('id').reset_index(drop=True)
submission_df.to_csv(OUTPUT_CSV, index=False)

invalid = submission_df[~submission_df['label'].isin(set(CLASS_NAMES))]
print(f'\n✅ Saved → {OUTPUT_CSV}')
print(f'   Rows: {len(submission_df)} | Nulls: {submission_df.isnull().sum().sum()} | Invalid: {len(invalid)}')
print(f'   Labels: {sorted(submission_df["label"].unique())}')
print(submission_df.head(6).to_string(index=False))
