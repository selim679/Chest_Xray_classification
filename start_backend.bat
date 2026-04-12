@echo off
echo ==========================================
echo  PneumoScan AI - Starting Backend
echo ==========================================

cd /d "%~dp0backend"

REM Create virtual env if missing
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

REM Install deps
echo Installing dependencies...
pip install -r requirements.txt --quiet

REM Create models dir hint
if not exist "models" (
    mkdir models
    echo [IMPORTANT] Copy your .pth files to: %~dp0backend\models\
    echo   - best_resnet18_advanced.pth
    echo   - best_vit_advanced.pth
)

echo.
echo Starting FastAPI on http://localhost:8000
echo API docs at http://localhost:8000/docs
echo.
python main.py
pause
