@echo off
chcp 65001 >nul
echo =========================================
echo Running CS336 Assignment 2 Tests
echo =========================================
echo.

:: Fix for Windows PyTorch distributed
set USE_LIBUV=0

echo Running tests... (this may take a few minutes)
uv run pytest -v ./tests --junitxml=test_results.xml > test_output.txt 2>&1

echo.
echo =========================================
echo Tests completed!
echo =========================================
echo.
echo Results saved to: test_output.txt
echo JUnit XML saved to: test_results.xml
echo.

:: Display summary
echo Test Summary:
findstr /C:"passed" /C:"failed" /C:"error" /C:"FAILED" /C:"SKIPPED" test_output.txt 2>nul || echo Check test_output.txt for details

echo.
pause
