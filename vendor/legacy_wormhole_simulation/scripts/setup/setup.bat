@echo off
REM Quantum Wormhole Simulation Framework - Windows Setup Script
REM This script provides an easy way to install the framework on Windows

echo.
echo =========================================================
echo  Quantum Wormhole Simulation Framework - Setup
echo =========================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Starting installation...
echo.

REM Set default options
set INSTALL_TYPE=standard
set EXTRA_FLAGS=

REM Parse command line arguments
:parse_args
if "%1"=="" goto :run_install
if "%1"=="--dev" (
    set INSTALL_TYPE=development
    set EXTRA_FLAGS=%EXTRA_FLAGS% --dev
)
if "%1"=="--minimal" (
    set INSTALL_TYPE=minimal
    set EXTRA_FLAGS=%EXTRA_FLAGS% --minimal
)
if "%1"=="--gpu" (
    set INSTALL_TYPE=%INSTALL_TYPE% with GPU
    set EXTRA_FLAGS=%EXTRA_FLAGS% --gpu
)
if "%1"=="--test" (
    set EXTRA_FLAGS=%EXTRA_FLAGS% --test
)
if "%1"=="--force" (
    set EXTRA_FLAGS=%EXTRA_FLAGS% --force
)
shift
goto :parse_args

:run_install
echo Installing %INSTALL_TYPE% version...
echo.

REM Run the Python installer
python install.py %EXTRA_FLAGS%

if errorlevel 1 (
    echo.
    echo Installation failed. Check installation.log for details.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo  Installation completed successfully!
echo =========================================================
echo.
echo To get started:
echo   1. Run a demo simulation: python main.py --mode demo
echo   2. Try interactive mode: python examples\03_interactive_visualization.py
echo   3. Read the user guide: docs\user_guide.md
echo.
echo For verification: python verify_installation.py
echo.
pause