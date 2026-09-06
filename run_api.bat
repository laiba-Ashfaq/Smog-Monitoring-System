@echo off
cd /d "%~dp0smog_sentinel_punjab"
python -m uvicorn api:app --reload --port 8000
pause
