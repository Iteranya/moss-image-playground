@echo off
rem ── MOSS playground: backend + UI on http://127.0.0.1:8000 ─────────────────────
rem Defaults to the Sogni Supernet (no GPU, no 27GB of weights, needs SOGNI_API_KEY).
rem   run.bat              -> Sogni engine
rem   set MOSS_BACKEND=local  then run.bat  -> your own GPU (downloads weights)
rem   set MOSS_FAKE=1     then run.bat  -> fake images, no GPU, no keys
setlocal
cd /d "%~dp0backend"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

if "%MOSS_BACKEND%"=="" set MOSS_BACKEND=sogni

rem Load the API key from .secrets\sogni.env if it is not already in the environment.
if "%SOGNI_API_KEY%"=="" if exist "%~dp0.secrets\sogni.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.secrets\sogni.env") do (
    if /i "%%A"=="SOGNI_API_KEY" set SOGNI_API_KEY=%%B
  )
)

if "%MOSS_BACKEND%"=="sogni" (
  if "%SOGNI_API_KEY%"=="" (
    echo [run] No SOGNI_API_KEY. Put it in .secrets\sogni.env or export it.
    echo [run] Or run a local GPU instead:  set MOSS_BACKEND=local ^&^& run.bat
    pause
    exit /b 1
  )
  echo [run] engine: Sogni Supernet
) else (
  echo [run] engine: %MOSS_BACKEND%
)

start "" http://127.0.0.1:8000/
uvicorn app:app --host 127.0.0.1 --port 8000
pause
