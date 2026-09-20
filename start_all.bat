@echo off

REM Change to script directory
cd /d "%~dp0"

REM Stop old service on port 8000 first, so every launch loads the latest code
echo Checking for existing service on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Stopping old service process %%a ...
    taskkill /F /PID %%a >nul 2>&1
)

REM Start backend service
start "AOD Inversion Service" /min python run_aod_service.py

REM Wait until the service is actually LISTENING (max 90s) before opening the page.
REM A fixed sleep was unreliable: importing numpy/pandas/matplotlib takes much
REM longer than 3s on slower machines, and the browser then hit a dead port.
echo Starting AOD Inversion Service, waiting for port 8000 to be ready...
set /a tries=0
:waitloop
set /a tries+=1
if %tries% gtr 90 (
    echo.
    echo [ERROR] Service did not start within 90 seconds.
    echo Please check the "AOD Inversion Service" window for error messages.
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if errorlevel 1 goto waitloop

REM Open frontend page in browser only after the port is live
echo Service is ready, opening frontend page...
start "" "http://localhost:8000/"

echo Service started successfully! This window can be closed.
