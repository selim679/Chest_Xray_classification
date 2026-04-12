#!/bin/bash
set -e

echo "=========================================="
echo " PneumoScan AI — Starting Backend"
echo "=========================================="

cd "$(dirname "$0")/backend"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt -q

# Remind about model files
if [ ! -d "models" ]; then
    mkdir -p models
    echo ""
    echo "⚠️  IMPORTANT: Copy your .pth model files to:"
    echo "   $(pwd)/models/"
    echo "   - best_resnet18_advanced.pth"
    echo "   - best_vit_advanced.pth"
    echo ""
fi

echo ""
echo "🚀 Starting FastAPI on http://localhost:8000"
echo "📖 API docs at http://localhost:8000/docs"
echo ""
python main.py
