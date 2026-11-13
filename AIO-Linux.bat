@echo off
rem ============================================================================
rem  AIO-linux -- the Linux part of AIO.bat. Called by AIO.bat, never by hand.
rem  Standalone: stamped per project, reads nothing from the template.
rem
rem  MSVC makes Windows binaries and a Windows host cannot produce a Linux Qt
rem  binary, so the verbs here run on a Linux machine. When one is not this
rem  host, the file returns 3 and says what to run there -- a named command is
rem  returned as an actionable failure, never a silent success.
rem    build   cmake --preset (or -S/-B) + cmake --build   into build\Linux
rem    deploy  stage the binary and its Qt libraries into build\Linux\Dist
rem ============================================================================
set "AIO_APP=Garuda"
set "AIO_FAMILY=cmake"
set "AIO_EXE=Garuda.exe"
set "AIO_CMAKE_DIR=."
set "AIO_CMAKE_PRESETS=1"
set "AIO_CMAKE_QT=1"
set "AIO_CMAKE_APP="
set "AIO_GRADLE_DIR="
set "AIO_GRADLE_MODULE="
set "AIO_BUILD_DIR=build"
set "JOB_CAP=8"
set "CMAKE_EXE=C:\Dev\Qt\Tools\CMake_64\bin\cmake.exe"
set "CTEST_EXE=C:\Dev\Qt\Tools\CMake_64\bin\ctest.exe"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "QT_DIR=C:\Dev\Qt\6.12.0\msvc2022_64"
set "AIO_LINUX_SRC=%AIO_HOME%\%AIO_CMAKE_DIR%"
set "AIO_LINUX_BUILD=%AIO_HOME%\%AIO_BUILD_DIR%\Linux"

if /i "%VERB%"=="run"   goto :aio_linux_run
if /i "%VERB%"=="test"  goto :aio_linux_test
if /i "%VERB%"=="clean" goto :aio_linux_clean
if /i "%VERB%"=="build"   goto :aio_linux_build
if /i "%VERB%"=="compile" goto :aio_linux_build
if /i "%VERB%"=="debug"   goto :aio_linux_build
if /i "%VERB%"=="deploy"  goto :aio_linux_deploy
echo [linux] %VERB% is not implemented for Linux.
exit /b 3

:aio_linux_build
if "%AIO_CMAKE_DIR%"=="" (
    if "%AIO_GRADLE_DIR%"=="" (
        echo [linux] %AIO_APP% has no Linux build.
        exit /b 3
    )
    goto :aio_linux_gradle
)
if /i "%PLATFORM_NATIVE%"=="linux" goto :aio_linux_native
echo [linux] %AIO_APP% builds on a Linux host; this one is Windows.
echo [linux] there:  cmake --preset %CONFIG%   or   cmake -S . -B %AIO_BUILD_DIR%/Linux -DCMAKE_BUILD_TYPE=%CONFIG%
echo [linux] then:  cmake --build %AIO_BUILD_DIR%/Linux -j 8
exit /b 3

:aio_linux_native
call :aio_need_msvc
call :aio_need_cmake
if /i "%AIO_CMAKE_PRESETS%"=="1" (
    pushd "%AIO_LINUX_SRC%"
    "%CMAKE_EXE%" --preset %CONFIG%
    set "CFG_RC=!ERRORLEVEL!"
    popd
    if "!CFG_RC!"=="0" goto :aio_linux_build_it
)
"%CMAKE_EXE%" -S "%AIO_LINUX_SRC%" -B "%AIO_LINUX_BUILD%" -DCMAKE_BUILD_TYPE=%CONFIG%
if errorlevel 1 exit /b 1
:aio_linux_build_it
"%CMAKE_EXE%" --build "%AIO_LINUX_BUILD%" --config %CONFIG% -j %JOB_CAP%
exit /b %ERRORLEVEL%

:aio_linux_gradle
call "%AIO_HOME%\%AIO_GRADLE_DIR%\gradlew" -p "%AIO_HOME%\%AIO_GRADLE_DIR%" :%AIO_GRADLE_MODULE%:assembleRelease
exit /b %ERRORLEVEL%

:aio_linux_deploy
if not exist "%AIO_LINUX_BUILD%" (
    echo [linux] nothing built in %AIO_BUILD_DIR%\Linux yet.
    exit /b 1
)
set "DIST=%AIO_LINUX_BUILD%\Dist"
if exist "%DIST%" rmdir /s /q "%DIST%"
mkdir "%DIST%" >nul 2>&1
for %%F in ("%AIO_LINUX_BUILD%\%AIO_EXE%") do if exist "%%~F" copy /y "%%~F" "%DIST%\" >nul
echo [linux] staged %DIST%
exit /b 0

:aio_linux_run
set "TARGET=%AIO_LINUX_BUILD%\%AIO_EXE%"
if not exist "%TARGET%" (
    echo [linux] %AIO_APP% is not built in %AIO_BUILD_DIR%\Linux.
    exit /b 1
)
start "" "%TARGET%" %EXTRA%
exit /b 0

:aio_linux_test
if "%AIO_CMAKE_APP%"=="0" (
    echo [aio] %AIO_APP% registers no CTest suite; nothing to run.
    exit /b 0
)
"%CTEST_EXE%" --test-dir "%AIO_LINUX_BUILD%" -C %CONFIG% --output-on-failure --timeout 300
if errorlevel 1 exit /b 1
exit /b 0

:aio_linux_clean
if exist "%AIO_LINUX_BUILD%" rmdir /s /q "%AIO_LINUX_BUILD%"
exit /b 0
