@echo off
rem Serves the MOSS Image Playground on http://127.0.0.1:8765
rem (the API rejects file:// requests with CORS 403, so a local server is required)
cd /d "%~dp0"
start "" http://127.0.0.1:8765/index.html
where python >nul 2>nul && (python -m http.server 8765 --bind 127.0.0.1) || (py -m http.server 8765 --bind 127.0.0.1)
pause
