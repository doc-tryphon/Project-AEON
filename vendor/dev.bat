@echo off
:: Quantum Simulation Project - Windows Development Commands

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="install-dev" goto install-dev
if "%1"=="test" goto test
if "%1"=="format" goto format
if "%1"=="lint" goto lint
if "%1"=="clean" goto clean
if "%1"=="demo" goto demo
if "%1"=="benchmark" goto benchmark
if "%1"=="setup" goto setup
if "%1"=="verify" goto verify
goto invalid

:help
echo Available commands:
echo   dev install       Install production dependencies
echo   dev install-dev   Install development dependencies
echo   dev test          Run all tests
echo   dev format        Format code with black
echo   dev lint          Lint code with flake8
echo   dev clean         Clean cache and temporary files
echo   dev demo          Run quick demo
echo   dev benchmark     Run performance benchmark
echo   dev setup         Full development environment setup
echo   dev verify        Verify installation
goto end

:install
python scripts\setup\install.py
goto end

:install-dev
python scripts\setup\install.py --dev
goto end

:test
python -m pytest tests\ -v
goto end

:format
python -m black src\ tests\ main.py scripts\
goto end

:lint
python -m flake8 src\ tests\ main.py scripts\
goto end

:clean
echo Cleaning cache and temporary files...
if exist __pycache__ rmdir /s /q __pycache__
if exist cache rmdir /s /q cache
if exist temp rmdir /s /q temp
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del .coverage
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul
echo Cleanup complete!
goto end

:demo
python main.py --mode demo
goto end

:benchmark
python main.py --mode benchmark --steps 500
goto end

:setup
call :install-dev
echo Development environment ready!
goto end

:verify
python scripts\setup\verify_installation.py
goto end

:invalid
echo Invalid command: %1
echo Use 'dev help' to see available commands
goto end

:end