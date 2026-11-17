@echo off
rem One-click setup for Windows. Double-click this file.
rem Installs Python 3.14 for your user account if no Python 3.14 or newer is found, then runs lib\setup_install.py.
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title Setup - Beatgrid and Cue Analysis

set "PYVERSION=3.14.10"
set "PYTHON="

rem 1. a suitable Python that is already installed (py launcher, PATH, or the folder this script installs to)
for %%V in (3.16 3.15 3.14) do if not defined PYTHON for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON call :check "%%P"
if not defined PYTHON for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON call :check "%%P"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" call :check "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if defined PYTHON goto run

rem 2. none found: download and install Python 3.14 (current user only, no administrator rights needed)
echo Python 3.14 or newer was not found on this computer.
echo It will now be downloaded from python.org (about 30 MB) and installed for your user account only.
echo Python is also added to your PATH. Nothing else on your computer is changed.
echo.
set "ARCH=amd64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "ARCH=arm64"
set "INSTALLER=%TEMP%\python-%PYVERSION%-%ARCH%.exe"
set "TARGET=%LOCALAPPDATA%\Programs\Python\Python314"
if exist "%INSTALLER%" del "%INSTALLER%" >nul 2>nul
echo Downloading python-%PYVERSION%-%ARCH%.exe ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/%PYVERSION%/python-%PYVERSION%-%ARCH%.exe' -OutFile '%INSTALLER%'"
if not exist "%INSTALLER%" goto download_failed
echo Installing Python %PYVERSION% (this takes a minute, no window will appear) ...
start /wait "" "%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_pip=1 Shortcuts=0 "TargetDir=%TARGET%"
del "%INSTALLER%" >nul 2>nul
if exist "%TARGET%\python.exe" call :check "%TARGET%\python.exe"
if not defined PYTHON goto install_failed
echo Python %PYVERSION% is installed.
echo.

:run
echo Using Python: %PYTHON%
echo.
"%PYTHON%" -m lib.setup_install --no-pause %*
set "EXITCODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXITCODE%

:download_failed
echo.
echo Python could not be downloaded. Check your internet connection and double-click this file again.
echo Or install Python 3.14 yourself from https://www.python.org/downloads/ (tick "Add python.exe to PATH") and double-click this file again.
echo.
pause
exit /b 1

:install_failed
echo.
echo The Python installation did not work.
echo Please install Python 3.14 yourself from https://www.python.org/downloads/ (tick "Add python.exe to PATH") and double-click this file again.
echo.
pause
exit /b 1

:check
set "CANDIDATE=%~1"
if not "%CANDIDATE:WindowsApps=%"=="%CANDIDATE%" exit /b 0
"%CANDIDATE%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON=%CANDIDATE%"
exit /b 0
