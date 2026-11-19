@echo off
setlocal
rem Double-click this launcher or run it from cmd/PowerShell.
rem Optional arguments: -Full, -DownloadWeights, -AddUvToPath, -NoPathPrompt, -PathOnly.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_test.ps1" %*
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" (
    echo [FAIL] Environment check failed. See the diagnostics above.
) else (
    echo [PASS] Environment check completed.
)
echo Press any key to close this window or return to the shell.
pause >nul
exit /b %RESULT%
