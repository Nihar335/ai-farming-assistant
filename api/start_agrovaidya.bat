@echo off
title AgroVaidya - Disease Detection API
color 0A

echo.
echo ==================================================
echo    AgroVaidya - Disease Detection API
echo ==================================================
echo.

:: Change to the project directory
cd /d "C:\Computer Vision Project\api"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python not found! Make sure Python is installed and in PATH.
    pause
    exit /b 1
)

:: Check if the model file exists
if not exist "C:\Computer Vision Project\model\plant_disease_model.pth" (
    color 0C
    echo [ERROR] Model file not found at:
    echo         C:\Computer Vision Project\model\plant_disease_model.pth
    echo.
    echo Make sure your model file is in the correct location.
    pause
    exit /b 1
)

:: Check if the class names file exists
if not exist "C:\Computer Vision Project\model\class_names.json" (
    color 0C
    echo [ERROR] class_names.json not found at:
    echo         C:\Computer Vision Project\model\class_names.json
    pause
    exit /b 1
)

echo [OK] Project folder found
echo [OK] Model files found
echo.
echo  Starting API server...
echo  Once loaded, open: http://localhost:5000
echo.
echo  Press Ctrl+C to stop the server
echo ==================================================
echo.

:: Start the Flask API
python api.py

:: If it crashes, don't close immediately
echo.
echo ==================================================
echo  [!] Server stopped or crashed.
echo ==================================================
pause
