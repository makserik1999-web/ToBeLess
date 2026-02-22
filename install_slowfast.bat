@echo off
echo ========================================
echo SlowFast Dependencies Installation
echo RTX 4060 GPU Version (CUDA 11.8)
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Please make sure Python is installed and in PATH.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created!
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo [INFO] Installing requirements (this may take 5-10 minutes)...
echo [INFO] Downloading PyTorch with CUDA 11.8 support (~2GB)
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed!
    echo.
    echo Possible issues:
    echo - Internet connection
    echo - Disk space (need ~5GB free)
    echo - Conflicting packages
    echo.
    pause
    exit /b 1
)

REM Verify GPU is available
echo.
echo [INFO] Verifying GPU setup...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if errorlevel 1 (
    echo.
    echo [WARNING] Could not verify GPU setup
    echo.
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Test SlowFast: python download_slowfast_model.py
echo 2. Run the app: python app.py
echo.
echo Your RTX 4060 should give ~25-30 FPS performance!
echo.
pause
