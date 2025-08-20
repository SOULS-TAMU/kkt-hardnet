@echo off
REM Example usage:
REM   installer.bat --filename requirements.txt
REM Default requirements file is "requirements.txt"

SET REQUIREMENTS_FILE=requirements.txt

REM Parse arguments
:parse
IF "%~1"=="" GOTO endparse
IF "%~1"=="--filename" (
    IF "%~2"=="" (
        ECHO Error: --filename requires a value
        EXIT /B 1
    )
    SET REQUIREMENTS_FILE=%~2
    SHIFT
    SHIFT
    GOTO parse
)
SHIFT
GOTO parse
:endparse

IF NOT EXIST "%REQUIREMENTS_FILE%" (
    ECHO Error: Requirements file "%REQUIREMENTS_FILE%" not found!
    EXIT /B 1
)

ECHO Installing libraries from %REQUIREMENTS_FILE% ...
pip install -r %REQUIREMENTS_FILE%
IF ERRORLEVEL 1 (
    ECHO Failed to install packages!
    EXIT /B 1
)
ECHO Done.
