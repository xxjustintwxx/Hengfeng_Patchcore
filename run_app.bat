@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Prefer conda.exe over conda.bat -- calling a .bat from inside this .bat
rem without "call" blows the batch recursion stack, and "where conda" can
rem return conda.bat first depending on install layout.
set "CONDA_EXE="
for /f "delims=" %%i in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%i"
if not defined CONDA_EXE (
    for %%P in (
        "%USERPROFILE%\miniconda3\Scripts\conda.exe"
        "%USERPROFILE%\anaconda3\Scripts\conda.exe"
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    ) do if exist %%~P set "CONDA_EXE=%%~P"
)
if not defined CONDA_EXE (
    for /f "delims=" %%i in ('where conda 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%i"
)

if not defined CONDA_EXE (
    echo [ERROR] Could not find conda on this machine.
    echo Run setup_env.bat first, or install Miniconda.
    pause
    exit /b 1
)

rem No NVIDIA GPU on this machine -- force CPU inference instead of each
rem profile's configured device (normally "cuda"), same as passing
rem "python app.py --device cpu" by hand.
set "DEVICE_FLAG="
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected -- running on CPU ^(slower^).
    set "DEVICE_FLAG=--device cpu"
)

rem Open the browser a couple seconds after Flask has had time to bind,
rem without blocking the server startup below.
start "" cmd /c "timeout /t 2 >nul & start http://localhost:5000"

echo Starting Hengfeng PatchCore.
echo To stop the server, click the X to close this window.
echo (Don't press Ctrl+C -- cmd.exe will ask "Terminate batch job (Y/N)?"; closing the window skips that.)
call "%CONDA_EXE%" run -n aoi --no-capture-output python app.py %DEVICE_FLAG%

pause
