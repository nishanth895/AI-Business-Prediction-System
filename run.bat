@echo off
REM ============================================================
REM  One-click setup + run for Windows 11
REM  Double-click this file or run from PowerShell: .\run.bat
REM ============================================================
cd /d "%~dp0"

REM 1) Create virtual env if missing
if not exist venv (
    echo [1/5] Creating virtual environment ...
    python -m venv venv
)

REM 2) Activate + upgrade pip
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

REM 3) Install dependencies
echo [2/5] Installing dependencies ...
pip install -r requirements.txt

REM 4) Train models + generate reports
echo [3/5] Training models + generating report artefacts ...
python train_model.py

REM 5) Generate charts
echo [4/5] Generating result charts ...
python generate_charts.py

REM 6) Run tests
echo [5/5] Running test suite ...
python run_tests.py

echo.
echo ============================================================
echo  Launching Streamlit dashboard. Browser will open shortly.
echo  Press Ctrl+C in this window to stop the dashboard.
echo ============================================================
streamlit run src/app.py

pause
