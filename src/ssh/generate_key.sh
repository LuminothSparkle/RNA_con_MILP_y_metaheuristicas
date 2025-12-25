#!/bin/bash
cd "$(dirname $(readlink -f \"$0\"))"
LOCAL_PORT=1080
nc -z $(LOCAL_HOST) $(LOCAL_PORT) || bash restart_proxy.sh 2
chromium-browser --proxy-server="socks5://$(LOCAL_HOST):$(LOCAL_PORT)"