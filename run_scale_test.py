#!/usr/bin/env python3
"""
Wrapper to run scale validation test with proper encoding on Windows.
"""
import sys
import io

# Force UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Now import and run the test
import tests.test_scale_validation as test_module

if __name__ == "__main__":
    # Run the test
    results = test_module.main()
    print(f"\n✅ Scale validation completed!")
