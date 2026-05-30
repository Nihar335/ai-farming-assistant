from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import io
import os
import requests

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Computer Vision Project\model\plant_disease_model.pth"
CLASSES_PATH = r"C:\Computer Vision Project\model\class_names.json"
IMG_SIZE     = 224
# ── Groq configuration ─────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── Load class names ────────────────────────
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# ── Load model ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*50}")
print(f"  AgroVaidya — Disease Detection API")
print(f"{'='*50}")
print(f"  Device     : {device}")
if device.type == "cuda":
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
print(f"  Classes    : {num_classes}")

# Rebuild model architecture
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes),
)

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()
print(f"  Model loaded (val_acc={checkpoint['val_acc']*100:.2f}%)")
print(f"{'='*50}\n")
print("  API is running at http://localhost:5000")
print("  Ready to accept image predictions!\n")

# ── Image transform ─────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Helper: format class name ───────────────
def format_class_name(raw_name):
    """Convert 'Tomato___Early_blight' → ('Tomato', 'Early Blight')"""
    parts = raw_name.split("___")
    crop    = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return crop, disease

# ── Helper: confidence label ─────────────────
def confidence_label(conf):
    if conf >= 0.95: return "Very High"
    if conf >= 0.85: return "High"
    if conf >= 0.70: return "Medium"
    return "Low"

# ────────────────────────────────────────────
#  ROUTES
# ────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "app":     "AgroVaidya Disease Detection API",
        "classes": num_classes,
        "device":  str(device),
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a POST request with an image file.
    Returns predicted disease, confidence, crop name, and top 3 predictions.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send image as form-data with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read and preprocess image
        img_bytes = file.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor    = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs     = model(tensor)
            probs       = torch.softmax(outputs, dim=1)[0]
            top3_probs, top3_idxs = torch.topk(probs, 3)

        # Top prediction
        top_idx   = top3_idxs[0].item()
        top_conf  = top3_probs[0].item()
        top_class = class_names[top_idx]
        crop, disease = format_class_name(top_class)

        # Top 3 predictions
        top3 = []
        for prob, idx in zip(top3_probs, top3_idxs):
            c, d = format_class_name(class_names[idx.item()])
            top3.append({
                "crop":       c,
                "disease":    d,
                "confidence": round(prob.item() * 100, 2),
            })

        is_healthy = "healthy" in disease.lower()

        return jsonify({
            "success":          True,
            "crop":             crop,
            "disease":          disease,
            "is_healthy":       is_healthy,
            "confidence":       round(top_conf * 100, 2),
            "confidence_label": confidence_label(top_conf),
            "raw_class":        top_class,
            "top3":             top3,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    """Returns all 38 disease classes the model can detect."""
    formatted = []
    for name in class_names:
        crop, disease = format_class_name(name)
        formatted.append({"crop": crop, "disease": disease, "raw": name})
    return jsonify({"total": num_classes, "classes": formatted})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded", "device": str(device)})


# ────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    system_msg = data.get("systemMsg", "")
    user_msg = data.get("userMsg", "")

    if not GROQ_API_KEY:
        return jsonify({"error": "Groq API key missing"}), 500

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    }

    try:
        r = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload
        )

        reply = r.json()["choices"][0]["message"]["content"]

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)