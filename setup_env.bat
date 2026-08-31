@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==========================================================
echo  Hengfeng PatchCore -- first-time environment setup
echo ==========================================================
echo.

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
    echo Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
    echo Then run this script again.
    pause
    exit /b 1
)

echo Using conda: %CONDA_EXE%
echo.

rem "conda create -n aoi" on an ALREADY-EXISTING env doesn't error or ask --
rem it silently wipes and rebuilds it, taking every installed package with
rem it. Check first and skip creation if it's already there; re-running this
rem script (e.g. after a git pull) then just re-syncs pip packages below,
rem instead of destroying whatever was already installed.
call "%CONDA_EXE%" env list | findstr /r /c:"^aoi " /c:"^aoi$" >nul
if errorlevel 1 (
    echo Creating conda environment "aoi" ^(Python 3.10^)...
    call "%CONDA_EXE%" create -n aoi python=3.10 -y
    if errorlevel 1 (
        echo [ERROR] "conda create" failed -- see the output above.
        pause
        exit /b 1
    )
) else (
    echo Environment "aoi" already exists -- skipping creation, just re-syncing packages.
)

echo.
echo Installing dependencies (this can take a while)...
call "%CONDA_EXE%" run -n aoi --no-capture-output pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] "pip install -r requirements.txt" failed -- see the output above.
    pause
    exit /b 1
)

call "%CONDA_EXE%" run -n aoi --no-capture-output pip install -e .
if errorlevel 1 (
    echo [ERROR] "pip install -e ." failed -- see the output above.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo  Setup complete. Double-click run_app.bat to start the app.
echo  Remember to also copy the models/ and configs/ folders
echo  onto this machine if they aren't here yet.
echo ==========================================================
pause
