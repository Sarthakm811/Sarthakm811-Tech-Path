#!/usr/bin/env python3
"""
Simple runner script for the Learning Path Recommender
This is an alternative to main.py for quick local testing
"""

import subprocess
import sys
import os
from pathlib import Path

def run_app():
    """Run the application using main.py"""
    try:
        # Make sure we're in the right directory
        os.chdir(Path(__file__).parent)
        
        # Run main.py
        subprocess.run([sys.executable, "main.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Application failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Learning Path Recommender...")
    run_app()