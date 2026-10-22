@echo off
rem Serves the playground AND the MOSS-Image-SFT API on http://127.0.0.1:8000
rem (same origin, so no CORS and no API key). See backend\README.md for setup.
setlocal
cd /d "%~dp0backend"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

if "%DIFFSYNTH_MODEL_BASE_PATH%"=="" set DIFFSYNTH_MODEL_BASE_PATH=D:\models
if "%MOSS_MODE%"=="" set MOSS_MODE=disk
rem Set MOSS_FAKE=1 to try the UI with no GPU / no weights / no torch:
if "%MOSS_FAKE%"=="1" echo [start-backend] FAKE pipeline - no GPU will be used

start "" http://127.0.0.1:8000/
uvicorn app:app --host 127.0.0.1 --port 8000
pause
