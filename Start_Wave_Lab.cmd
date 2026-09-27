@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_wave_lab.py
) else (
  python run_wave_lab.py
)
if errorlevel 1 (
  echo Please follow README.md to install Python and requirements.txt.
  pause
)
