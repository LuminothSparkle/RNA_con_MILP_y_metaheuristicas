#!/bin/bash
cd "$(dirname $(readlink -f \"$0\"))"
nc -z $(LOCAL_HOST) $(LOCAL_PORT) || bash restart_tunnel.sh 10
echo $(LOCAL_PASS)| sshpass scp -r -p -P $(LOCAL_PORT) "$(dirname $(readlink -f \"$1\"))/data" "$(LOCAL_USER)@$(LOCAL_HOST):$(dirname $(readlink -f \"$2\"))"
nc -z $(LOCAL_HOST) $(LOCAL_PORT) || bash restart_tunnel.sh 5
ssh  $(LOCAL_USER)@$(LOCAL_HOST) -p $(LOCAL_PORT) "cd \"$(dirname $(readlink -f \"$2\"))\" && bash execute_gurobi.sh"
nc -z $(LOCAL_HOST) $(LOCAL_PORT) || bash restart_tunnel.sh 2
echo $(LOCAL_PASS)| sshpass scp -r -p -P $(LOCAL_PORT) "$(LOCAL_USER)@$(LOCAL_HOST):$(dirname $(readlink -f \"$2\"))/results.7z" "."
echo SSH session $(LOCAL_USER)@$(LOCAL_HOST):$(LOCAL_PORT) through tunnel closed.
7z x -y "$(dirname $(readlink -f \"$1\"))/results.7z" -o"$(dirname $(readlink -f \"$1\"))/results"