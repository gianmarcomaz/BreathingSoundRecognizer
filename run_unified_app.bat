@echo off
echo ========================================
echo Unified Neonatal Monitoring App
echo ========================================
echo.
echo Starting Streamlit app...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
    echo.
)

REM Check if required modules are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install streamlit opencv-python sounddevice
    echo.
)

REM Run the app
streamlit run unified_neonatal_app.py

pause
