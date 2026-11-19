@echo off
setlocal
cd /d "%~dp0"
set "DEFAULT_COMMAND="
if "%~1"=="" set "DEFAULT_COMMAND=demo"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 -m team_search %DEFAULT_COMMAND% %*
    exit /b
)
if exist "..\cctv-search\.venv311\Scripts\python.exe" (
    "..\cctv-search\.venv311\Scripts\python.exe" -X utf8 -m team_search %DEFAULT_COMMAND% %*
    exit /b
)
py -3.11 -X utf8 -m team_search %DEFAULT_COMMAND% %*
