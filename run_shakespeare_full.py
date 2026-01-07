#!/usr/bin/env python3
"""
Run FULL Tiny Shakespeare benchmark with proper encoding on Windows.
This is the production-ready benchmark on the complete 1.1MB dataset.
"""
import sys
import io

# Force UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now import and run the benchmark
import tests.test_shakespeare_full as benchmark

if __name__ == "__main__":
    print("🚀 Starting FULL Tiny Shakespeare Benchmark...")
    print("   Dataset: 1.1MB (~1M characters)")
    print("   Model: 2-layer GRU (738,618 parameters)")
    print("   GodelAI: v1.1.0 (T-Score fix applied)")
    print("   Estimated time: 30-60 minutes")
    print()

    results, filepath = benchmark.main()

    print(f"\n✅ Benchmark completed! Results: {filepath}")
    print(f"📊 Final T-Score: {results['final_metrics']['final_t_score']:.4f}")
    print(f"📉 Best Val Loss: {results['final_metrics']['best_val_loss']:.4f}")
    print(f"💤 Total Sleep Events: {results['final_metrics']['total_sleep_events']}")
