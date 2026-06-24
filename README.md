# 🌍 Multiclass Satellite Image Classification using Transfer Learning (VGG16)

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-EuroSAT_RGB-1D9E75?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-10-7F77DD?style=for-the-badge)
![Val Accuracy](https://img.shields.io/badge/Val_Accuracy-86.46%25-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This project implements **Multiclass Satellite Image Classification** using **Transfer Learning** with the pre-trained **VGG16** model on the **EuroSAT RGB** dataset. The goal is to classify satellite images into **10 land-use categories** by leveraging features already learned from 14 million ImageNet images — and then fine-tuning those features for the satellite domain.

> *"Instead of training a deep neural network from scratch, Transfer Learning lets us stand on the shoulders of giants."*

**Final Result:** Achieved **86.46% validation accuracy** using only 166,026 trainable parameters — just 1.1% of the full VGG16 model.

---

## 📊 Results

| Phase | Layers Trained | Train Accuracy | Val Accuracy | Val Loss |
|---|---|---|---|---|
| Phase 1 — Transfer Learning | Custom Dense Head only | 83.31% | 81.67% | 0.5276 |
| Phase 2 — Fine-Tuning | Last 4 VGG16 layers + Dense Head | 83.54% | **86.46%** | **0.3820** |
| **Improvement** | — | +0.23% | **+4.79%** | **-0.1456** |

### 🔮 Prediction Results

| Test Image | Actual Class | Predicted Class | Confidence |
|---|---|---|---|
| Before fine-tuning | Pasture | Pasture ✅ | 90.56% |
| After fine-tuning | AnnualCrop | AnnualCrop ✅ | **100.00%** |

---

## 🗂️ Dataset — EuroSAT RGB

| Property | Details |
|---|---|
| **Source** | [Kaggle — EuroSAT RGB](https://www.kaggle.com/datasets/nilesh789/eurosat-rgb) |
| **Total Images** | 27,000 satellite images |
| **Image Size** | 64×64 px (resized to 224×224 for VGG16) |
| **Classes** | 10 land-use categories |
| **Train Split** | 80% → 21,600 images |
| **Validation Split** | 20% → 5,400 images |

### 🏷️ 10 Land-Use Classes

| # | Class | Description |
|---|---|---|
| 1 | AnnualCrop | Seasonal crop fields |
| 2 | Forest | Dense forested areas |
| 3 | HerbaceousVegetation | Grasslands and shrubs |
| 4 | Highway | Roads and motorways |
| 5 | Industrial | Factories and warehouses |
| 6 | Pasture | Open grazing land |
| 7 | PermanentCrop | Vineyards, orchards |
| 8 | Residential | Housing and urban areas |
| 9 | River | Waterways and streams |
| 10 | SeaLake | Seas, lakes, reservoirs |

---

## 🧠 Model Architecture

### Transfer Learning Strategy

```
INPUT IMAGE (224×224×3)
        ↓
┌──────────────────────────────────────┐
│         VGG16 (Pre-trained)          │
│   14M images · 1000 classes          │
│                                      │
│  block1_conv1 → block4_pool  FROZEN  │ ← Phase 1 & 2
│  block5_conv1 → block5_pool  FROZEN  │ ← Phase 1 only
│                               TUNED  │ ← Phase 2 fine-tuning
└──────────────────────────────────────┘
        ↓
   GlobalAveragePooling2D()    512 features
        ↓
   Dense(256, ReLU)
        ↓
   BatchNormalization()
        ↓
   Dropout(0.4)
        ↓
   Dense(128, ReLU)
        ↓
   Dropout(0.3)
        ↓
   Dense(10, Softmax)          ← Our custom classification head
        ↓
OUTPUT: 10 class probabilities
```

### Model Size

| Component | Parameters |
|---|---|
| Total parameters | 14,881,226 (56.77 MB) |
| Trainable (Phase 1) | 166,026 (648 KB) — only 1.1% of model |
| Non-trainable (frozen) | 14,715,200 (56.13 MB) |

### Why VGG16?
- Pre-trained on **14 million images** across 1000 categories
- Strong at extracting low-level (edges, textures) to high-level (shapes, structures) features
- Proven performance on image classification with limited domain-specific data

### Why GlobalAveragePooling2D over Flatten?
An earlier version of this project used `Flatten() → Dense(64)` which produced only **~17% accuracy**. Replacing it with `GlobalAveragePooling2D() → Dense(256) → Dense(128)` improved accuracy to **81.67%** — a 64% absolute improvement from architecture redesign alone.

---

## 🔄 Training Pipeline

### Phase 1 — Transfer Learning (Frozen VGG16)

| Parameter | Value |
|---|---|
| Optimizer | Adam (default lr = 1e-3) |
| Loss | Categorical Crossentropy |
| Epochs | 10 (EarlyStopping triggered at Epoch 4) |
| Best Epoch | Epoch 1 — restored by EarlyStopping |
| Trainable Layers | Custom Dense Head only (166,026 params) |
| Batch Size | 32 |

> **EarlyStopping** detected val_accuracy peaked at Epoch 1 (81.67%) and automatically restored the best weights before overfitting could set in.

### Phase 2 — Fine-Tuning (Last 4 VGG16 Layers Unfrozen)

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr = 1e-5) |
| Loss | Categorical Crossentropy |
| Epochs | 5 (consistent improvement every epoch) |
| Unfrozen Layers | block5_conv1, block5_conv2, block5_conv3, block5_pool |
| Reason for Low LR | Prevent catastrophic forgetting of ImageNet weights |

> Val accuracy improved **every single epoch**: 85.93% → 86.02% → 86.07% → 86.20% → **86.46%**
> Val loss dropped steadily: 0.3965 → 0.3923 → 0.3890 → 0.3853 → **0.3820**

---

## 📁 Project Structure

```
satellite-image-classification-vgg16/
│
├── 📓 notebook/
│   └── satellite_image_classification.ipynb
│
├── 📁 saved_models/
│   └── .gitkeep               ← model files excluded (too large)
│
├── 📁 sample_outputs/
│   ├── training_accuracy.png  ← Phase 1 accuracy & loss curves
│   ├── finetune_accuracy.png  ← Phase 2 accuracy & loss curves
│   ├── prediction_pasture.png ← Pasture prediction (90.56%)
│   └── prediction_crop.png   ← AnnualCrop prediction (100%)
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/arti2441/satellite-image-classification-vgg16
cd satellite-image-classification-vgg16
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset (Kaggle API)

Make sure your `kaggle.json` API key is placed at `~/.kaggle/`

```bash
kaggle datasets download -d nilesh789/eurosat-rgb
unzip eurosat-rgb.zip
```

### 4. Run the Notebook

Open in **Google Colab** (recommended):

```bash
jupyter notebook notebook/satellite_image_classification.ipynb
```

---

## 🔮 How to Predict on a New Image

```python
# Use the built-in predict_image() function
predict_image(
    image_path="path/to/your/satellite_image.jpg",
    class_names=CLASSES,
    model=model
)
```

**Example Output:**
```
Actual Class    : Pasture
Predicted Class : Pasture
Confidence      : 90.56%
```

---

## 💡 Key Concepts Demonstrated

- **Transfer Learning** — Reusing ImageNet-trained VGG16 features for satellite imagery classification
- **Feature Extraction** — 14.7M frozen parameters acting as a powerful fixed feature extractor
- **Fine-Tuning** — Gently updating block5 VGG16 layers with lr=1e-5 for domain adaptation
- **GlobalAveragePooling2D** — Efficient spatial feature summarization (25,088 → 512 features)
- **BatchNormalization** — Stabilizes training and speeds up convergence
- **EarlyStopping** — Automatically restored best weights, preventing overfitting
- **ModelCheckpoint** — Saved model after every epoch for safe recovery
- **Dropout Regularization** — 40% and 30% dropout to prevent co-adaptation of neurons

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 | Programming language |
| TensorFlow / Keras | 2.20.0 | Model building and training |
| VGG16 (ImageNet weights) | — | Pre-trained base model |
| OpenCV (cv2) | 4.7+ | Image reading and BGR→RGB conversion |
| NumPy | 1.23+ | Array operations |
| Matplotlib | 3.7+ | Training curves and prediction visualization |
| Scikit-learn | 1.2+ | Utilities |
| Kaggle API | 2.0+ | Dataset download |

---

## 🌍 Real-World Applications

- **Precision agriculture** — identifying AnnualCrop, PermanentCrop, Pasture for yield estimation
- **Urban monitoring** — tracking Residential and Industrial zone expansion over time
- **Environmental conservation** — detecting Forest loss and SeaLake/River changes
- **Disaster response** — rapid post-flood or post-wildfire land-use assessment
- **Infrastructure planning** — automated Highway and road network identification

---

## 📚 References

- [VGG16 Paper — Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [EuroSAT Dataset Paper (Helber et al., 2019)](https://arxiv.org/abs/1709.00029)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [Kaggle EuroSAT RGB Dataset](https://www.kaggle.com/datasets/nilesh789/eurosat-rgb)

---

## 👩‍💻 Author

**Arti Anil Nighote**
B.Tech — Artificial Intelligence & Data Science (Final Year)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arti-nighote-508055360/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/arti2441)
[![HackerRank](https://img.shields.io/badge/HackerRank-2EC866?style=flat-square&logo=hackerrank&logoColor=white)](https://www.hackerrank.com/profile/nighotearti)

---

<p align="center"><em>"A model trained on 14 million natural images learned to recognize satellite landscapes it had never seen — with 86% accuracy, in under 30 minutes of training. That's the power of Transfer Learning."</em></p>
