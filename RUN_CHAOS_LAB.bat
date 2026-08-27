@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
echo Installing or checking dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
for /f %%P in ('.venv\Scripts\python.exe tools\find_free_port.py') do set PORT=%%P
if not defined PORT (
  echo No free localhost port found in 8501-8520.
  pause
  exit /b 1
)
echo Starting Nonlinear Dynamics ^& Chaos Lab at http://localhost:%PORT%
.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port %PORT%
endlocal
