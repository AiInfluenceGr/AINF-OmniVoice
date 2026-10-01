@echo off
rem OmniVoice control panel - double-click to start, then open http://127.0.0.1:7870
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Run "Setup OmniVoice.bat" first.
  pause
  exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" app.py --open %*
pause
