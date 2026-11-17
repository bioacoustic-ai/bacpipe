@echo off
rem Starts bacpipe with a console window that shows all messages of the
rem application. Use it
rem   * to see what goes wrong if bacpipe.exe does not start,
rem   * as an alternative if your system does not allow bacpipe.exe to run.
rem It does exactly what bacpipe.exe does, with python.exe (console) instead
rem of pythonw.exe (no console).

set "BACPIPE_APP_DIR=%~dp0"
"%~dp0runtime\python.exe" -E -s -X utf8 -m bacpipe_app %*

echo.
echo bacpipe has ended (exit code %ERRORLEVEL%).
pause
