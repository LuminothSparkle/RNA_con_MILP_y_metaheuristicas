cd /D "%~dp0"
set LOCAL_PORT=1080
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_proxy.bat 2
start chrome.exe --proxy-server="socks5://%LOCAL_HOST%:%LOCAL_PORT%"