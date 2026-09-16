#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AOD Inversion Service

This script starts the AOD inversion service on http://localhost:8000

To run this script:
1. Open a command prompt
2. Navigate to the directory containing this file
3. Run: python run_aod_service.py

The service will start and run until you press Ctrl+C
"""

import uvicorn
import main

if __name__ == "__main__":
    print("Starting AOD Inversion Service...")
    print("Server will run at http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run(main.app, host="0.0.0.0", port=8000, log_level="info")
