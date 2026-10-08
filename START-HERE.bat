@echo off
rem Sets Lamina up on first use (asks two questions), then starts a chat.
rem   START-HERE.bat              chat
rem   START-HERE.bat serve        OpenAI-compatible server on http://127.0.0.1:8000
rem   START-HERE.bat setup        install, download and build only
rem   START-HERE.bat --reconfigure   choose context length and MTP again
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 or newer is required: https://www.python.org/downloads/
    exit /b 1
)
python tools\quickstart.py %*
exit /b %errorlevel%
