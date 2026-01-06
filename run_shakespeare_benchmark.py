#!/usr/bin/env python3
"""
Wrapper to run Shakespeare benchmark with proper encoding on Windows.
"""
import sys
import io

# Force UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now import and run the benchmark
import tests.test_shakespeare_benchmark as benchmark

if __name__ == "__main__":
    print("Starting Shakespeare benchmark...")
    results, filepath = benchmark.main()
    print(f"\n✅ Benchmark completed! Results: {filepath}")
