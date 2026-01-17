#!/bin/bash

# Start overnight training with logging

echo "Starting overnight training run..."
echo "Logs will be written to: overnight_run.log"
echo "To check progress: tail -f overnight_run.log"
echo "To stop: pkill -f run_overnight.py"
echo ""

# Run from src directory in background with output to log file
cd src
nohup python3 run_overnight.py > ../overnight_run.log 2>&1 &

PID=$!
cd ..

echo "✓ Training started in background (PID: $PID)"
echo ""
echo "Commands:"
echo "  View progress:  tail -f overnight_run.log"
echo "  Check status:   ps aux | grep run_overnight"
echo "  Stop training:  kill $PID"
echo "  View results:   cd src && python3 view_overnight_results.py"
echo ""
echo "The simulation will run until complete or until you stop it."
echo "Results are saved every 1000 steps to: results/overnight_stats.npz"
