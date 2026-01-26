@echo off
chcp 65001 > nul

echo Запуск приложения браслета...
echo.

cd /d "%~dp0"

python app.py

echo.
echo Приложение завершено.
pause
