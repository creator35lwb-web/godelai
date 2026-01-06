#!/usr/bin/env python3
"""Run T-Score fix validation with UTF-8 encoding."""
import sys
import io

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tests.test_tscore_fix as test_module

if __name__ == "__main__":
    test_module.main()
