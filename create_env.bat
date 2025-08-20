@echo off
REM Example usage:
REM   create_env.bat --name myenv
REM Default environment name is "venv"

SET ENV_NAME=venv

REM Parse arguments
:parse
IF "%~1"=="" GOTO endparse
IF "%~1"=="--name" (
    IF "%~2"=="" (
        ECHO Error: --name requires a value
        EXIT /B 1
    )
    SET ENV_NAME=%~2
    SHIFT
    SHIFT
    GOTO parse
)
SHIFT
GOTO parse
:endparse

ECHO Creating virtual environment: %ENV_NAME% ...
python -m venv %ENV_NAME%
IF ERRORLEVEL 1 (
    ECHO Failed to create virtual environment!
    EXIT /B 1
)

ECHO Activating virtual environment ...
CALL %ENV_NAME%\Scripts\activate.bat
ECHO Done.
