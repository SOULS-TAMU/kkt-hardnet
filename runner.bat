@echo off
setlocal

REM -----------------------------------------
REM Run all models for problem_pooling (Windows .bat)
REM   1. MLP
REM   2. PINN
REM   3. KKT-HardNet
REM For directory .\Op_problems\Testing
REM and plot the losses after each run.
REM -----------------------------------------

REM (Optional) enable UTF-8 so emojis/text render nicely in the console
chcp 65001 >nul

REM Set your problem directory here (Windows uses backslashes)
set PROBLEM_DIR=.\Op_problems\problem_pooling
REM "PROBLEM_DIR=.\Op_problems\Testing"
set "DO_PLOT=0"

echo.
echo Running in Problem Directory: %PROBLEM_DIR%

REM If you do not want to run a specific model, comment out the corresponding call below

call :run_model mlp  %DO_PLOT%  || goto :end
call :run_model pinn %DO_PLOT%  || goto :end

set "DO_PLOT=1"
call :run_model kkt  %DO_PLOT%  || goto :end

echo.
echo ✅ Code has been executed for %PROBLEM_DIR%
goto :eof

:run_model
REM Args: %1 = mode, %2 = do_plot
set "MODE=%~1"
set "PLOT=%~2"

echo.
REM If your Python is python3.exe, change 'python' to 'python3'
python main.py --dir_path "%PROBLEM_DIR%" --mode %MODE% --do_plot %PLOT%
if errorlevel 1 (
  echo [Error] %MODE% failed with exit code %errorlevel%.
  exit /b %errorlevel%
)
exit /b 0

:end
echo.
echo [Stopped] Due to an error in a previous step.
exit /b 1
