#!/bin/bash

PID_FILE="fl_pids.txt"

echo "--- Stopping FL Processes ---"
echo "Attempt 1: Using PID file..."

if [ -f "$PID_FILE" ]; then
    PIDS_TO_KILL=$(cat $PID_FILE)
    if [ -n "$PIDS_TO_KILL" ]; then
        echo "Killing PIDs: $PIDS_TO_KILL"
        # Use 'kill' on all PIDs
        # '2>/dev/null' suppresses "No such process" errors
        kill $PIDS_TO_KILL 2>/dev/null
    else
        echo "PID file is empty."
    fi
    # Clean up the PID file
    rm $PID_FILE
else
    echo "PID file ($PID_FILE) not found."
fi

echo "Attempt 2: Forcefully killing any lingering app processes..."
# This is a more aggressive command that finds any process
# matching the "python -m server.app" or "python -m client.app" string
pkill -f "python -m server.app"
pkill -f "python -m client.app"

echo "--- Done ---"