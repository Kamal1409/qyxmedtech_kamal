@echo off
REM QYX MedTech semen analysis website -- one-command start.
REM Builds the website the first time (or after `set REBUILD=1`), then starts the
REM server and opens http://localhost:8000.  Close this window to stop it.
setlocal
cd /d "%~dp0"

set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "frontend\dist\index.html" set REBUILD=1
if "%REBUILD%"=="1" (
  echo Building the website...
  pushd frontend
  if not exist node_modules call npm install || goto :fail
  call npm run build || goto :fail
  popd
)

echo Starting the analysis server on http://localhost:8000 ...
start "" http://localhost:8000
pushd backend
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
popd
goto :eof

:fail
echo Build failed. See the messages above.
pause
