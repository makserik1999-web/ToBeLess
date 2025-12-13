@echo off
echo Installing requirements in virtual environment...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
pip install flask-cors
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
pause
