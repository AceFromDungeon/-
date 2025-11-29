@echo off
chcp 65001 >nul
title RZD Worker Monitor PRO
color 0a
cd /d "%~dp0"

:menu
cls
echo.
echo    ╔══════════════════════════════════════════════╗
echo    ║        RZD WORKER MONITORING SYSTEM          ║
echo    ╚══════════════════════════════════════════════╝
echo.
echo    1. Переустановить зависимости
echo    2. Нарисовать зоны
echo    3. Запустить систему
echo    4. Выход
echo.
set /p ch=Выбери пункт меню (1-4) ^> 

:: ================================
:: 1. Переустановка зависимостей
:: ================================
if "%ch%"=="1" (
    echo.
    echo Удаление старого окружения...
    rd /s /q venv 2>nul

    echo Создание нового окружения...
    python\python.exe -m venv venv

    echo Активация окружения...
    call venv\Scripts\activate.bat

    echo Обновление pip...
    python -m pip install --upgrade pip

    echo Установка зависимостей из requirements.txt...
    pip install -r requirements.txt --no-cache-dir

    echo.
    echo ===========================================
    echo      Переустановка зависимостей завершена
    echo ===========================================
    pause
    goto menu
)

:: ================================
:: 2. Рисуем зоны
:: ================================
if "%ch%"=="2" (
    if not exist venv (
        echo Виртуальное окружение не найдено. Сначала выбери пункт 1.
        pause
        goto menu
    )
    call venv\Scripts\activate.bat
    venv\Scripts\python.exe draw_zones.py
    goto menu
)

:: ================================
:: 3. Запуск системы
:: ================================
if "%ch%"=="3" (
    if not exist venv (
        echo Виртуальное окружение не найдено. Сначала выбери пункт 1.
        pause
        goto menu
    )
    call venv\Scripts\activate.bat
    venv\Scripts\python.exe main.py
    pause
    goto menu
)

:: ================================
:: 4. Выход
:: ================================
if "%ch%"=="4" exit

goto menu
