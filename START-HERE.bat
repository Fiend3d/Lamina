@echo off
setlocal
cd /d "%~dp0"
python setup.py %*
exit /b %errorlevel%
