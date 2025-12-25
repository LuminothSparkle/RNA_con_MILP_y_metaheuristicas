cd /D "%~dp0"
set LOCAL_PORT=1080
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_proxy.bat 2
echo Activating gurobi key ...
set HTTPS_PROXY="socks5://%LOCAL_HOST%:%LOCAL_PORT%"
grbgetkey %GUROBI_KEY%