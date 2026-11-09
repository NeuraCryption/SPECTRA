#!/usr/bin/env python3
# Copyright (c) 2026 NeuraCryption Private Limited (neuracryption.com). All Rights Reserved.
"""
SPECTRA 1-Click Turnkey Launcher
"""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectra.cli import main

if __name__ == "__main__":
    main()
