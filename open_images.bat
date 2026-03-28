@echo off
setlocal enabledelayedexpansion

REM Set root directory (current folder)
set ROOT=%cd%

echo Opening all images in %ROOT% ...
echo.

for /r "%ROOT%" %%f in (*.png *.jpg *.jpeg) do (
    echo Opening: %%f
    start "" "%%f"
)

echo.
echo Done!
pause
