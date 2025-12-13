@echo off
echo Starting ToBeLess AI Backend...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the Flask application
python app.py

pause
