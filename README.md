# 🌾 AI-Powered Personal Farming Assistant

> A computer vision system that detects crop diseases and pest infestations from leaf images using ResNet-50, with LLM-powered treatment recommendations for smallholder farmers.

---

## 📌 Project Overview

Smallholder farmers lose over **35% of their crop yield** annually due to undetected diseases and pests. This project puts an AI-powered agronomist in every farmer's pocket — just take a photo of your crop leaf and get an instant diagnosis with a treatment plan.

The system combines a fine-tuned **ResNet-50 CNN** for visual disease classification with an **LLM reasoning layer** that converts predictions into actionable, farmer-friendly advice.

---

## ✨ Features

- 🔍 **Crop disease detection** across 38 disease classes with 90%+ accuracy
- 🌿 **Grad-CAM heatmaps** to visually highlight the diseased region on the leaf
- 🤖 **LLM-powered recommendations** — treatment plans, pesticide advice, and prevention tips in plain language
- 📱 **Streamlit web interface** — upload a photo and get results instantly
- 🧪 **Transfer learning** — fine-tuned ResNet-50 on the PlantVillage dataset

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, torchvision |
| Computer Vision | OpenCV, Albumentations, Pillow |
| Model | ResNet-50 (Transfer Learning) |
| Explainability | Grad-CAM |
| LLM Integration | Anthropic API |
| Evaluation | scikit-learn, Matplotlib |
| Web App | Streamlit |

---

## 📁 Project Structure

```
ai-farming-assistant/
│
├── app.py                  ← Streamlit web application
├── model.py                ← ResNet-50 model definition
├── train.py                ← Model training script
├── predict.py              ← Prediction + Grad-CAM generation
├── requirements.txt        ← All dependencies
│
├── notebooks/
│   └── EDA.ipynb           ← Exploratory data analysis
│
└── models/
    └── resnet50_plantvillage.pth  ← Saved model weights
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Nihar335/ai-farming-assistant.git
cd ai-farming-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the root folder:
```
ANTHROPIC_API_KEY=your_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🧠 How It Works

```
Farmer captures leaf image
        ↓
Image preprocessing (resize 224×224, normalize)
        ↓
ResNet-50 CNN classifies disease (38 classes)
        ↓
Grad-CAM generates heatmap on diseased region
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
| Classification Accuracy | 90%+ |
| Base Model | ResNet-50 (ImageNet pre-trained) |
| Training Strategy | Transfer Learning (fine-tuned last layers) |

---

## 📦 Dataset

This project uses the **PlantVillage Dataset** — 54,000+ labelled leaf images across 38 disease classes covering tomato, potato, corn, apple, and more.

Download it from:
- [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)

> ⚠️ Dataset is not included in this repository due to size. Download and place it in a `/dataset` folder.

---

## 🎯 Real-World Impact

- 🇮🇳 India has **140 million farming households**
- Only **1 agronomist per 1,000 farmers** is available
- This system provides instant expert-level diagnosis via any smartphone camera
- Supports **local language output** through LLM integration

---

## 👨‍💻 Author

**Nihar**
B.Tech Computer Science | [Lovely Professional University]
Project Duration: Mar 2026 – May 2026
Course: Computer Vision

---

## 📄 License

This project is for academic purposes only.
```
