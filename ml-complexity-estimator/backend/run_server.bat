@echo off
echo Installing dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies. Please ensure Python is installed and in your PATH.
    pause
    exit /b %ERRORLEVEL%
)

echo Starting server...
python -m uvicorn app.main:app --host 127.0.0.1 --reload
pause
