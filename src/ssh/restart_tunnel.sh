#!/bin/bash
echo Establishing SSH tunnel to $(INTER_USER)@$(INTER_HOST) ...
echo $(INTER_PASS)| sshpass ssh -f -o ServerAliveInterval=60 -L $(LOCAL_PORT):$(REMOTE_HOST):$(REMOTE_PORT) $(INTER_USER)@$(INTER_HOST) sleep $1
echo SSH tunnel established on $(LOCAL_HOST):$(LOCAL_PORT) for $(REMOTE_HOST):$(REMOTE_PORT) through $(INTER_USER)@$(INTER_HOST)