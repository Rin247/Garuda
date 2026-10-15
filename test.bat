@echo off
REM Garuda's test entry point (root AGENTS.md 34). Runs the APEX suite; CTest
REM runs from build\Release once a build exists (root AGENTS.md 26).
setlocal
python "%~dp0tools\quantize\apex\test_apex.py"
exit /b %ERRORLEVEL%
