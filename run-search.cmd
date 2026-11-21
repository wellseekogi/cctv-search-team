@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 -m team_search %*
    exit /b
)
if exist "..\cctv-search\.venv311\Scripts\python.exe" (
    "..\cctv-search\.venv311\Scripts\python.exe" -X utf8 -m team_search %*
    exit /b
)
py -3.11 -X utf8 -m team_search %*
