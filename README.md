# PneumoScan AI — Full Stack Deployment Guide

```
FastAPI (Python) + Angular (TypeScript) · Local Machine Deployment
Models: ResNet-18 (97.15%) · ViT-Small (96.63%) · Ensemble (97.50%)
```

---

## Project Structure

```
pneumoscan/
├── backend/
│   ├── main.py                  ← FastAPI app (all endpoints)
│   ├── requirements.txt         ← Python dependencies
│   ├── .env                     ← Environment config
│   └── models/                  ← ⚠️ PUT YOUR .pth FILES HERE
│       ├── best_resnet18_advanced.pth
│       └── best_vit_advanced.pth
│
├── frontend-angular/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.component.ts     ← Main component logic
│   │   │   ├── app.component.html   ← UI template
│   │   │   ├── app.component.scss   ← Styles
│   │   │   ├── app.config.ts        ← Angular providers
│   │   │   └── pneumo.service.ts    ← API service
│   │   ├── main.ts
│   │   ├── index.html
│   │   └── styles.scss
│   ├── angular.json
│   ├── package.json
│   ├── tsconfig.json
│   └── tsconfig.app.json
│
├── start_backend.sh             ← Linux/Mac: run backend
├── start_backend.bat            ← Windows: run backend
├── start_frontend.sh            ← Linux/Mac: run frontend
└── README.md
```

---

## Step 1 — Copy Your Trained Models

Download your `.pth` files from Google Drive and paste them in:

```
backend/models/best_resnet18_advanced.pth
backend/models/best_vit_advanced.pth
```

> **Quick copy from Google Drive on Colab:**
> ```python
> import shutil
> shutil.copy('/content/drive/MyDrive/ChestXray_Project_X/models/best_resnet18_advanced.pth', '.')
> shutil.copy('/content/drive/MyDrive/ChestXray_Project_X/models/best_vit_advanced.pth', '.')
> ```

---

## Step 2 — Start the FastAPI Backend

### Linux / macOS
```bash
chmod +x start_backend.sh
./start_backend.sh
```

### Windows
```cmd
Double-click start_backend.bat
```

### Manual (any OS)
```bash
cd backend
python -m venv venv

# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

**Backend runs at:** `http://localhost:8000`  
**API docs (Swagger UI):** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`

---

## Step 3 — Start the Angular Frontend

### Prerequisites
Install Node.js (v18+) from https://nodejs.org

### Linux / macOS
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

### Manual
```bash
cd frontend-angular
npm install          # first time only (~2 minutes)
npm start            # starts dev server
```

**Frontend runs at:** `http://localhost:4200`

---

## Step 4 — Open the App

Open your browser: **http://localhost:4200**

The nav bar will show **"Models Ready"** when both servers are running and models are loaded.

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | App info |
| GET | `/health` | Backend & model status |
| POST | `/predict/ensemble` | **Best** — ResNet + ViT ensemble |
| POST | `/predict/resnet` | ResNet-18 only |
| POST | `/predict/vit` | ViT-Small only |
| POST | `/predict/all` | Run all 3, compare side-by-side |

### Example with curl
```bash
curl -X POST http://localhost:8000/predict/ensemble \
  -F "file=@chest_xray.jpg"
```

### Example Response
```json
{
  "model": "Ensemble (ResNet-18 + ViT-Small)",
  "prediction": "Pneumonia",
  "confidence": 94.23,
  "probabilities": {
    "COVID19": 1.12,
    "Normal": 2.45,
    "Pneumonia": 94.23,
    "Tuberculosis": 2.20
  },
  "inference_ms": 187.4,
  "ensemble_weights": { "resnet": 0.5, "vit": 0.5 }
}
```

---

## Changing Ensemble Weights

In `backend/main.py`, find this line and update with your best weights from training:

```python
ENSEMBLE_WEIGHTS = [0.5, 0.5]   # [resnet_weight, vit_weight]
# e.g. if ResNet 0.6 / ViT 0.4 gave better results:
ENSEMBLE_WEIGHTS = [0.6, 0.4]
```

---

## GPU vs CPU

The backend automatically detects GPU. To verify:
```
GET http://localhost:8000/health
→ { "cuda_available": true, "device": "cuda" }
```

To force CPU (if GPU causes issues):
```bash
CUDA_VISIBLE_DEVICES="" python main.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Models not loaded` in UI | Check models/ folder has both .pth files |
| `Backend Offline` in nav | Make sure `python main.py` is running |
| CORS error in browser | Ensure Angular runs on port 4200 |
| `ModuleNotFoundError: timm` | Run `pip install timm` in venv |
| Port 8000 already in use | `lsof -i :8000` then kill the PID |
| Slow inference (CPU) | Normal — ResNet ~200ms, ViT ~800ms on CPU |

---

## Production Build (optional)

```bash
# Build Angular for production
cd frontend-angular
npm run build
# Output: dist/pneumoscan-frontend/

# Serve static files from FastAPI
# Add to main.py:
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="../frontend-angular/dist/pneumoscan-frontend/browser", html=True))
```

---

## Class Labels

The model was trained with these exact class folder names:
```
COVID19 · Normal · Pneumonia · Tuberculosis
```
These match `CLASS_NAMES` in `main.py`. Do not rename them.

---

*PneumoScan AI — For educational and research use only. Not a substitute for professional medical diagnosis.*
