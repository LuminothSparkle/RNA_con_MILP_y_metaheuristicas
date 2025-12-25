cd /D "%~dp0"
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_tunnel.bat 10
echo %LOCAL_PASS%| sshpass scp -r -p -P %LOCAL_PORT% "%~dp1/data" "%LOCAL_USER%@%LOCAL_HOST%:%~dp2"
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_tunnel.bat 5
ssh  %LOCAL_USER%@%LOCAL_HOST% -p %LOCAL_PORT% "cd \"$(dirname $(readlink -f \"%~dp2\"))\" && bash execute_gurobi.sh"
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_tunnel.bat 2
echo %LOCAL_PASS%| sshpass scp -r -p -P %LOCAL_PORT% "%LOCAL_USER%@%LOCAL_HOST%:%~dp2/results.7z" "%~dp1"
echo SSH session %LOCAL_USER%@%LOCAL_HOST%:%LOCAL_PORT% through tunnel closed.
7z x -y "%~dp1/results.7z" -o"%~dp1/results"