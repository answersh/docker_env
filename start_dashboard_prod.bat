@echo off
cd /d D:\docker_env
docker-compose up -d streamlit-prod
echo Dashboard-prod service started in detached mode.
pause