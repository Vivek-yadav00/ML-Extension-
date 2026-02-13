@echo off
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please ensure python and pip are in your PATH.
    pause
    exit /b %errorlevel%
)

echo Starting server...
python -m uvicorn app.main:app --reload
pause
