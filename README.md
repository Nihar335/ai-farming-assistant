# 🌾 AgroVaidya — AI-Powered Personal Farming Assistant

> A computer vision system that detects crop diseases and pest infestations from leaf images using MobileNetV3-Large, with LLM-powered treatment recommendations for smallholder farmers.

---

## 📌 Project Overview

Smallholder farmers lose over **35% of their crop yield** annually due to undetected diseases and pests. AgroVaidya puts an AI-powered agronomist in every farmer's pocket — just take a photo of your crop leaf and get an instant diagnosis with a treatment plan.

The system combines a fine-tuned **MobileNetV3-Large CNN** for visual disease classification with an **LLM reasoning layer** that converts predictions into actionable, farmer-friendly advice.

---

## ✨ Features

- 🔍 **Crop disease detection** across 38 disease classes with **98% accuracy**
- 🌿 **Confusion matrix & training curves** to validate and visualize model performance
- 🤖 **LLM-powered recommendations** — treatment plans, pesticide advice, and prevention tips in plain language
- 📱 **Custom HTML/CSS frontend** connected to a Flask REST API backend
- 🧪 **Transfer learning** — fine-tuned MobileNetV3-Large on the PlantVillage dataset

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python, HTML, CSS |
| Deep Learning | PyTorch, torchvision |
| Computer Vision | OpenCV, Albumentations |
| Model | MobileNetV3-Large (Transfer Learning) |
| Evaluation | scikit-learn, Matplotlib, Seaborn |
| LLM Integration | Anthropic API |
| Backend | Flask REST API |
| Frontend | HTML / CSS / JavaScript |

---

## 📁 Project Structure

```
agrovaidya/
│
├── agrovaidya.html         ← Frontend web interface
├── train.py                ← MobileNetV3 training script
├── requirements.txt        ← All dependencies
│
├── model/
│   ├── plant_disease_model.pth   ← Saved model weights
│   └── class_names.json          ← 38 class labels
│
├── Figure_1.png            ← Training accuracy & loss curves
└── Figure_2.png            ← Confusion matrix (38 classes)
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Nihar335/agrovaidya.git
cd agrovaidya
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (or skip if .pth file is available)
```bash
python train.py
```

### 4. Add your API key
Create a `.env` file in the root folder:
```
ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Run the Flask API
```bash
python app.py
```

### 6. Open the frontend
Open `agrovaidya.html` in your browser.

---

## 🧠 How It Works

```
Farmer captures leaf image
        ↓
Image preprocessing (resize 224×224, normalize)
        ↓
MobileNetV3-Large CNN classifies disease (38 classes)
        ↓
Confidence score + predicted class returned
        ↓
LLM generates treatment recommendation
        ↓
Farmer receives diagnosis + advice on screen
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Dataset | PlantVillage (54,000+ images) |
| Number of Classes | 38 disease classes |
| Classification Accuracy | **98%** |
| Base Model | MobileNetV3-Large (ImageNet pre-trained) |
| Training Strategy | Transfer Learning (fine-tuned classifier head) |
| Epochs | 15 |
| Optimizer | Adam (lr=0.001, StepLR scheduler) |
| Final Val Loss | ~0.06 |

---

## 📈 Training Results

**Accuracy per Epoch** — Model reaches 98% validation accuracy by epoch 15

**Loss per Epoch** — Validation loss drops from 0.16 → 0.06 over 15 epochs showing strong convergence with no overfitting

---

## 📦 Dataset

This project uses the **PlantVillage Dataset** — 54,000+ labelled leaf images across 38 disease classes covering tomato, potato, corn, apple, grape, and more.

Download it from:
- [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)

> ⚠️ Dataset is not included in this repository due to size. Download and place it in the path defined in `train.py`.

---

## 🎯 Real-World Impact

- 🇮🇳 India has **140 million farming households**
- Only **1 agronomist per 1,000 farmers** is available
- AgroVaidya provides instant expert-level diagnosis via any smartphone camera
- Supports **local language output** through LLM integration

---

## 👨‍💻 Author

**Nihar Kanakam**
B.Tech Computer Science | Lovely Professional University
Project Duration: Mar 2026 – May 2026
Course: Computer Vision

---

## 📄 License

This project is for academic purposes only.
