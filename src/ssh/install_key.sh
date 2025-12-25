#!/bin/bash
cd "$(dirname $(readlink -f \"$0\"))"
LOCAL_PORT=108
nc -z $(LOCAL_HOST) $(LOCAL_PORT) || bash restart_proxy.sh 2
echo Activating gurobi key ...
HTTPS_PROXY="socks5://$(LOCAL_HOST):$(LOCAL_PORT)"
grbgetkey $(GUROBI_KEY)