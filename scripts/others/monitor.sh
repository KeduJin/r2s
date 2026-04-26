#!/bin/bash

PID=$1
NEXT_CMD=$2

echo "Monitoring PID: $PID"
echo "Will execute next command: $NEXT_CMD"

# Loop to check if PID exists
while kill -0 $PID 2> /dev/null; do
    sleep 10
done

echo "PID $PID has finished."
echo "Executing next command..."

# Execute next command
bash $NEXT_CMD