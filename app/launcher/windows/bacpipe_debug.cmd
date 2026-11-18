@echo off
rem Starts bacpipe like bacpipe.exe, but with a console window showing all
rem messages (python.exe instead of pythonw.exe).

set "BACPIPE_APP_DIR=%~dp0"
"%~dp0runtime\python.exe" -E -s -X utf8 -m bacpipe_app %*

echo.
echo bacpipe has ended (exit code %ERRORLEVEL%).
pause
