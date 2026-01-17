@echo off
REM Windows batch file for overnight training

echo Starting overnight training run...
echo Logs will be written to: overnight_run.log
echo To stop: Ctrl+C or close this window
echo.

REM Run from src directory with output to log file
cd src
python run_overnight.py > ..\overnight_run.log 2>&1
cd ..

echo.
echo Training complete!
echo View results: cd src && python view_overnight_results.py
pause
