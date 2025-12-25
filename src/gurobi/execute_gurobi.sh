#!/bin/bash
trap "" SIGINT
bash "grb_nn $(cat args.txt)" & PID=$!
trap "kill -s SIGINT $(PID)" SIGINT
wait $(PID)
rm -f results.7z
7z a results.7z results
trap - SIGINT