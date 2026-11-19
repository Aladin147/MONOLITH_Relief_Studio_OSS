@echo off
title MONOLITH
cd /d "%~dp0"
"%~dp0python\python.exe" -m backend.launcher %*
if not "%errorlevel%"=="0" (
  echo.
  echo MONOLITH stopped because of an error. Take a screenshot of this window.
  pause
)
