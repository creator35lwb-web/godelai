#!/usr/bin/env python3
"""Monitor Full Shakespeare Benchmark Progress"""
import sys
import io
import json
from pathlib import Path
from datetime import datetime

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def monitor():
    print("=" * 70)
    print("FULL SHAKESPEARE BENCHMARK - Progress Monitor")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # Check dataset
    data_file = Path('data/shakespeare.txt')
    if data_file.exists():
        size = data_file.stat().st_size
        print(f"Dataset: {size:,} bytes ({size/1024/1024:.2f} MB)")
    else:
        print("Dataset: Not found - downloading...")

    # Check for latest results
    results_dir = Path('results')
    if results_dir.exists():
        files = sorted(results_dir.glob('shakespeare_benchmark_*.json'),
                      key=lambda x: x.stat().st_mtime, reverse=True)

        if files:
            latest = files[0]
            print(f"\nLatest result: {latest.name}")
            print(f"Modified: {datetime.fromtimestamp(latest.stat().st_mtime).strftime('%H:%M:%S')}")

            # Try to read partial results
            try:
                with open(latest, 'r') as f:
                    data = json.load(f)

                if 'history' in data:
                    epochs = len(data['history']['train_loss'])
                    if epochs > 0:
                        print(f"\nProgress:")
                        print(f"  Epochs completed: {epochs}")
                        print(f"  Latest train loss: {data['history']['train_loss'][-1]:.4f}")
                        print(f"  Latest val loss: {data['history']['val_loss'][-1]:.4f}")
                        print(f"  Latest T-Score: {data['history']['t_score'][-1]:.4f}")
                        print(f"  Total sleep events: {sum(data['history']['sleep_events'])}")
            except Exception as e:
                print(f"  (Results file being written...)")
    else:
        print("\nNo results yet - training initializing...")

    print()
    print("Status: RUNNING (check again in 2-3 minutes)")
    print("=" * 70)

if __name__ == "__main__":
    monitor()
