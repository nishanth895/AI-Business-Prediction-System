#!/usr/bin/env bash
# ============================================================
#  One-click setup + run for Linux / macOS
# ============================================================
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "[1/5] Creating virtual environment ..."
    python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip

echo "[2/5] Installing dependencies ..."
pip install -r requirements.txt

echo "[3/5] Training models + generating report artefacts ..."
python train_model.py

echo "[4/5] Generating result charts ..."
python generate_charts.py

echo "[5/5] Running test suite ..."
python run_tests.py

echo
echo "============================================================"
echo " Launching Streamlit dashboard. Browser will open shortly."
echo " Press Ctrl+C to stop."
echo "============================================================"
streamlit run src/app.py
