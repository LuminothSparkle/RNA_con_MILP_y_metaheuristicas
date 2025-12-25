#!/bin/bash
cd "$(dirname $(readlink -f \"$0\"))"
nc -z $(LOCAL_HOST) $(LOCAL_PORT) || bash restart_tunnel.sh 2
ssh  $(LOCAL_USER)@$(LOCAL_HOST) -p $(LOCAL_PORT)
echo SSH session $(LOCAL_USER)@$(LOCAL_HOST):$(LOCAL_PORT) through tunnel closed.   