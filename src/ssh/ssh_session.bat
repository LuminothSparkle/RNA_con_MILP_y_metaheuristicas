cd /D "%~dp0"
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_tunnel.bat 2
ssh  %LOCAL_USER%@%LOCAL_HOST% -p %LOCAL_PORT%
echo SSH session %LOCAL_USER%@%LOCAL_HOST%:%LOCAL_PORT% through tunnel closed.