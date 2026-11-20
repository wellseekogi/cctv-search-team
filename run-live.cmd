@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=..\cctv-search\.venv311\Scripts\python.exe"
echo http://127.0.0.1:8780/ui/
if exist "%PY%" (
    "%PY%" -X utf8 -m team_search.live %*
) else (
    py -3.11 -X utf8 -m team_search.live %*
)
if errorlevel 1 pause
