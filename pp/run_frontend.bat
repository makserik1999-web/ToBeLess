@echo off
echo Starting ToBeLess AI Frontend...
echo.

REM Check if node_modules exists
if not exist "node_modules\" (
    echo Installing dependencies...
    call npm install
    echo.
)

REM Start development server
echo Starting Vite dev server...
call npm run dev

pause
