# 🌾 AgroVaidya — AI-Powered Crop Disease Detection & Farming Assistant

An AI-powered farming assistant that detects crop diseases from leaf images using **MobileNetV3-Large** and provides instant farmer-friendly treatment recommendations through a web interface.

Built as a Computer Vision project to help farmers quickly identify plant diseases and receive actionable guidance using image-based diagnosis.

---

## 📌 Project Overview

Crop diseases and pest infections can significantly reduce crop yield when not identified early. AgroVaidya helps solve this problem by allowing farmers to upload a leaf image and instantly receive:

- Disease prediction
- Confidence score
- Treatment suggestions
- Prevention guidance

The system combines **Computer Vision + Deep Learning + Web API integration** to deliver a simple and practical farming support tool.

---

## ✨ Features

- 🔍 Detects crop diseases from leaf images
- 🌿 Supports **38 disease classes**
- 📊 Achieves **~98% validation accuracy** on PlantVillage dataset
- 🤖 Generates treatment and prevention recommendations
- 🌐 Custom web interface for image upload and results
- ⚡ Flask backend for model prediction API
- 📈 Model evaluation using confusion matrix and training curves

---

## 🛠️ Tech Stack

### Languages
- Python
- HTML
- CSS
- JavaScript

### Deep Learning & Computer Vision
- PyTorch
- TorchVision
- OpenCV
- Albumentations

### Model
- MobileNetV3-Large (Transfer Learning)

### Backend
- Flask REST API

### Evaluation
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```bash
agrovaidya/
│
├── api/
│   ├── api.py
│   └── start_agrovaidya.bat
│
├── model/
│   ├── plant_disease_model.pth
│   └── class_names.json
│
├── agrovaidya.html
├── train.py
├── requirements.txt
├── README.md
```

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/Nihar335/ai-farming-assistant.git
cd ai-farming-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask API

```bash
python api/api.py
```

### 4. Open frontend

Open:

```bash
agrovaidya.html
```

in browser.

### 5. Upload crop leaf image

- Select image
- Click predict
- View:
  - disease name
  - confidence score
  - treatment recommendation

---

## 🧠 Workflow

```text
Leaf image upload
        ↓
Image preprocessing
        ↓
MobileNetV3-Large prediction
        ↓
Disease classification
        ↓
Treatment recommendation
        ↓
Result shown on web interface
```

---

## 📊 Model Performance

| Metric | Value |
|---|---:|
| Dataset | PlantVillage |
| Classes | 38 |
| Model | MobileNetV3-Large |
| Accuracy | ~98% |
| Framework | PyTorch |
| Training Type | Transfer Learning |

---

## 📦 Dataset

Dataset used:

**PlantVillage Dataset**

Includes labeled crop leaf images across multiple crops and disease categories.

Dataset is not uploaded in this repository due to file size.

---

## 🎯 Project Outcome

- Built a practical crop disease detection system
- Improved disease identification accuracy using transfer learning
- Created a farmer-friendly web interface
- Delivered fast predictions with treatment guidance

---

## 👨‍💻 Author

**Nihar Kanakam**  
B.Tech Computer Science  
Lovely Professional University

---

## 📄 License

Academic project for learning and demonstration purposes.
