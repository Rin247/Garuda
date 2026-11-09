@echo off
REM build.bat for Garuda (CMake presets or root)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if exist "%SCRIPT_DIR%\CMakePresets.json" (
    cmake --preset Release -S "%SCRIPT_DIR%" -B "%SCRIPT_DIR%\build\Release" 2>&1
    cmake --build "%SCRIPT_DIR%\build\Release" -j 8
) else if exist "%SCRIPT_DIR%\desktop\CMakeLists.txt" (
    cmake -S "%SCRIPT_DIR%\desktop" -B "%SCRIPT_DIR%\build\Release" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/Dev/Qt/6.12.0/msvc2022_64 -DCMAKE_MAKE_PROGRAM=C:/Dev/ninja/ninja.exe 2>&1
    cmake --build "%SCRIPT_DIR%\build\Release" -j 8
) else (
    echo No CMakeLists found
    exit /b 1
)
exit /b !ERRORLEVEL!
