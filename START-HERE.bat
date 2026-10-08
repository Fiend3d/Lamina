@echo off
rem Sets Lamina up on first use (asks two questions), then starts the model server.
rem   START-HERE.bat                 server on http://127.0.0.1:8000/v1 (OpenAI API, for pi and other apps)
rem   START-HERE.bat chat            talk to the model in this window instead
rem   START-HERE.bat setup           download and set up only
rem   START-HERE.bat --build-source  compile locally (developers)
rem   START-HERE.bat --reconfigure   choose context length and MTP again
rem   START-HERE.bat --model ornith  serve Ornith-1.5-35B-A3B instead of Qwen3.6
rem   START-HERE.bat --port 9000     use another port
rem Close the window or press Ctrl+C to stop the server.
setlocal
title Lamina
cd /d "%~dp0"
set "DOUBLECLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLECLICKED=1"
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -m tools.quickstart %*
    if errorlevel 1 (
        if defined DOUBLECLICKED pause
        exit /b 1
    )
    exit /b 0
)
where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 or newer is required: https://www.python.org/downloads/
    echo Tick "Add python.exe to PATH" in the installer, then start this file again.
    if defined DOUBLECLICKED pause
    exit /b 1
)
python tools\quickstart.py %*
set "CODE=%errorlevel%"
if not "%CODE%"=="0" if defined DOUBLECLICKED pause
exit /b %CODE%
