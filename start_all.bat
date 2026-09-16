@echo off

REM Change to script directory
cd /d "%~dp0"

REM Start backend service
start "AOD Inversion Service" /min python run_aod_service.py

REM Wait for service to start
echo Starting AOD Inversion Service...
timeout /t 3 /nobreak >nul

REM Open frontend page in browser
echo Opening frontend page...
start "" "http://localhost:8000/"

echo Service started successfully! Check the browser for the frontend page.