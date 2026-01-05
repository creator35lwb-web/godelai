#!/usr/bin/env python3
"""
Wrapper to run manifesto learning test with proper encoding on Windows.
"""
import sys
import io
import subprocess

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now import and run the test
import tests.test_manifesto_learning_v2 as test_module

if __name__ == "__main__":
    # Run the test
    results, filepath = test_module.main()
    print(f"\n✅ Test completed. Results saved to: {filepath}")
