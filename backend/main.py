"""
PneumoScan AI — FastAPI Backend
Serves ResNet-18, ViT-Small, and Ensemble models for chest X-ray classification.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
from PIL import Image
import io
import numpy as np
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PneumoScan AI API",
    description="Chest X-ray classification using ResNet-18, ViT-Small, and Ensemble models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CLASS_NAMES = ["COVID19", "Normal", "Pneumonia", "Tuberculosis"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update these paths to where your .pth files are saved
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
RESNET_PATH = os.path.join(MODELS_DIR, "best_resnet18_advanced.pth")
VIT_PATH = os.path.join(MODELS_DIR, "best_vit_advanced.pth")

# Ensemble weights (best from your training: ResNet 0.5 / ViT 0.5 or your best combo)
ENSEMBLE_WEIGHTS = [0.5, 0.5]

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
class AdvancedResNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
resnet_model = None
vit_model = None
models_loaded = False
load_error = None


def load_models():
    global resnet_model, vit_model, models_loaded, load_error
    try:
        logger.info(f"Loading models from: {MODELS_DIR}")
        logger.info(f"Device: {DEVICE}")

        # Load ResNet-18
        resnet_model = AdvancedResNet(num_classes=NUM_CLASSES, dropout=0.5).to(DEVICE)
        resnet_model.load_state_dict(
            torch.load(RESNET_PATH, map_location=DEVICE)
        )
        resnet_model.eval()
        logger.info("✅ ResNet-18 loaded")

        # Load ViT-Small
        vit_model = timm.create_model(
            'vit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES
        ).to(DEVICE)
        vit_model.load_state_dict(
            torch.load(VIT_PATH, map_location=DEVICE)
        )
        vit_model.eval()
        logger.info("✅ ViT-Small loaded")

        models_loaded = True
        logger.info("🚀 All models ready!")

    except FileNotFoundError as e:
        load_error = f"Model file not found: {e}. Copy your .pth files to {MODELS_DIR}/"
        logger.error(load_error)
    except Exception as e:
        load_error = str(e)
        logger.error(f"Error loading models: {e}")


@app.on_event("startup")
async def startup_event():
    load_models()


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes):
    """Convert raw bytes to PIL Image, ensure RGB."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def run_resnet(image: Image.Image):
    tensor = resnet_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = resnet_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def run_vit(image: Image.Image):
    tensor = vit_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = vit_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def build_result(probs: np.ndarray, model_name: str, inference_ms: float):
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return {
        "model": model_name,
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASS_NAMES, probs)
        },
        "inference_ms": round(inference_ms, 1)
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "app": "PneumoScan AI",
        "version": "1.0.0",
        "models_loaded": models_loaded,
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "error": load_error,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict/resnet")
async def predict_resnet(file: UploadFile = File(...)):
    """Run ResNet-18 inference on uploaded chest X-ray."""
    if not models_loaded:
        raise HTTPException(503, detail=load_error or "Models not loaded")

    contents = await file.read()
    try:
        image = preprocess_image(contents)
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    t0 = time.time()
    probs = run_resnet(image)
    ms = (time.time() - t0) * 1000

    return build_result(probs, "ResNet-18 Advanced", ms)


@app.post("/predict/vit")
async def predict_vit(file: UploadFile = File(...)):
    """Run ViT-Small inference on uploaded chest X-ray."""
    if not models_loaded:
        raise HTTPException(503, detail=load_error or "Models not loaded")

    contents = await file.read()
    try:
        image = preprocess_image(contents)
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    t0 = time.time()
    probs = run_vit(image)
    ms = (time.time() - t0) * 1000

    return build_result(probs, "ViT-Small (4× Data)", ms)


@app.post("/predict/ensemble")
async def predict_ensemble(file: UploadFile = File(...)):
    """Run Ensemble (ResNet + ViT) inference on uploaded chest X-ray."""
    if not models_loaded:
        raise HTTPException(503, detail=load_error or "Models not loaded")

    contents = await file.read()
    try:
        image = preprocess_image(contents)
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    t0 = time.time()
    resnet_probs = run_resnet(image)
    vit_probs = run_vit(image)
    ensemble_probs = (
        ENSEMBLE_WEIGHTS[0] * resnet_probs +
        ENSEMBLE_WEIGHTS[1] * vit_probs
    )
    ms = (time.time() - t0) * 1000

    result = build_result(ensemble_probs, "Ensemble (ResNet-18 + ViT-Small)", ms)
    result["component_predictions"] = {
        "resnet": build_result(resnet_probs, "ResNet-18", 0)["probabilities"],
        "vit": build_result(vit_probs, "ViT-Small", 0)["probabilities"],
    }
    result["ensemble_weights"] = {"resnet": ENSEMBLE_WEIGHTS[0], "vit": ENSEMBLE_WEIGHTS[1]}
    return result


@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    """Run all three models and return combined results."""
    if not models_loaded:
        raise HTTPException(503, detail=load_error or "Models not loaded")

    contents = await file.read()
    try:
        image = preprocess_image(contents)
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    t0 = time.time()
    resnet_probs = run_resnet(image)
    vit_probs = run_vit(image)
    ensemble_probs = (
        ENSEMBLE_WEIGHTS[0] * resnet_probs +
        ENSEMBLE_WEIGHTS[1] * vit_probs
    )
    total_ms = (time.time() - t0) * 1000

    return {
        "resnet": build_result(resnet_probs, "ResNet-18 Advanced", total_ms * 0.45),
        "vit": build_result(vit_probs, "ViT-Small (4× Data)", total_ms * 0.45),
        "ensemble": build_result(ensemble_probs, "Ensemble", total_ms),
        "total_inference_ms": round(total_ms, 1)
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
