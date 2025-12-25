cd /D "%~dp0"
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_tunnel.bat 10
echo %LOCAL_PASS%| sshpass scp -r -p -P %LOCAL_PORT% "src/gurobi/execute_gurobi.ssh" "src/gurobi/grb_nn.cpp" "%LOCAL_USER%@%LOCAL_HOST%:%~dp1"
nc -z %LOCAL_HOST% %LOCAL_PORT% || call restart_tunnel.bat 2
echo %LOCAL_PASS%| sshpass ssh  %LOCAL_USER%@%LOCAL_HOST% -p %LOCAL_PORT% "g++ \"%~dp1/grb_nn.cpp\" -std=c++23 -lgurobi_c++ -lgurobi110 -o \"%~dp1/grb_nn\""
echo SSH session %LOCAL_USER%@%LOCAL_HOST%:%LOCAL_PORT% through tunnel closed.