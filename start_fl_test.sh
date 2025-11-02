#!/bin/bash

# --- Configuration ---
NUM_CLIENTS=3
LOG_DIR="fl_logs"
PID_FILE="fl_pids.txt"

# --- Setup ---
# Create log directory
mkdir -p $LOG_DIR
# Clean up old PID file
rm -f $PID_FILE

echo "--- Starting FL Test ---"

# --- Start Server ---
echo "Starting Server... Log: $LOG_DIR/server.log"
python -m server.app > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > $PID_FILE
echo "Server PID: $SERVER_PID"

# --- NEW: Wait for server to boot ---
echo "Waiting 5 seconds for server to initialize..."
sleep 5
# ---

# --- Start Clients ---
echo "Starting $NUM_CLIENTS Clients..."
CLIENT_PIDS=()
for i in $(seq 1 $NUM_CLIENTS)
do
    # Format ID as client_001, client_002, etc.
    CLIENT_ID=$(printf "client_%03d" $i)
    LOG_FILE="$LOG_DIR/$CLIENT_ID.log"
    
    echo "Starting $CLIENT_ID... Log: $LOG_FILE"
    
    # Export CLIENT_ID for the python process
    export CLIENT_ID=$CLIENT_ID
    
    # Start client in background, redirecting stdout and stderr
    python -m client.app > "$LOG_FILE" 2>&1 &
    
    # Save the client's PID
    CLIENT_PIDS+=($!)
done

# Save all client PIDs to the file
echo ${CLIENT_PIDS[@]} >> $PID_FILE

echo "---"
echo "All processes are running in the background."
echo "PIDs saved to $PID_FILE"
echo "---"
echo "To monitor the server log, run:"
echo "tail -f $LOG_DIR/server.log"
echo
echo "To monitor a client log, run:"
echo "tail -f $LOG_DIR/client_001.log"
echo
echo "To stop all processes, run: ./stop_fl_test.sh"