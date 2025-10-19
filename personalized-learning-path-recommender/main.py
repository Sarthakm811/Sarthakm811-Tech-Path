#!/usr/bin/env python3
"""
Main Entry Point for Personalized Learning Path Recommender
This file serves as the unified entry point for the entire application,
making it easy to run locally and deploy on Streamlit Community Cloud.
"""

import os
import sys
import subprocess
import threading
import time
import signal
from pathlib import Path

# Add backend to Python path for imports
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import flask
        import requests
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing required packages...")
        
        # Install requirements
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def start_backend_server():
    """Start the Flask backend server in a separate thread"""
    try:
        # Use the simplified backend instead
        print("🚀 Starting simplified backend server on port 5000...")
        
        # Import and run the simplified Flask app
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        from simple_app import app
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
    except Exception as e:
        print(f"❌ Backend server failed to start: {e}")
        # Try to run a minimal Flask server
        try:
            from flask import Flask, jsonify
            minimal_app = Flask(__name__)
            
            @minimal_app.route('/api/health')
            def health():
                return jsonify({'success': True, 'message': 'Minimal backend running'})
            
            @minimal_app.route('/api/recommend', methods=['POST'])
            def recommend():
                return jsonify({'success': True, 'recommendations': {'learning_path': [], 'total_estimated_hours': 0, 'timeline': [], 'recommended_career_paths': []}})
            
            minimal_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except Exception as e2:
            print(f"❌ Even minimal backend failed: {e2}")

def wait_for_backend(max_attempts=30):
    """Wait for backend server to be ready"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5000/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting for backend server... ({attempt + 1}/{max_attempts})")
        time.sleep(2)
    
    print("❌ Backend server failed to start within timeout")
    return False

def run_streamlit_app():
    """Run the Streamlit frontend application"""
    try:
        # Change to frontend directory
        os.chdir("frontend")
        
        # Run Streamlit app
        print("🚀 Starting Streamlit frontend...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])
        
    except Exception as e:
        print(f"❌ Streamlit app failed to start: {e}")

def setup_environment():
    """Setup environment variables for the application"""
    # Set default environment variables if not already set
    env_vars = {
        'DATABASE_URL': 'sqlite:///learning_recommender.db',  # Use SQLite for simplicity
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
        'SECRET_KEY': 'dev-secret-key-change-in-production',
        'FLASK_ENV': 'production',
        'DEBUG': 'False'
    }
    
    for key, value in env_vars.items():
        if not os.getenv(key):
            os.environ[key] = value
    
    print("✅ Environment variables configured")

def main():
    """Main function to run the entire application"""
    print("🚀 Starting Personalized Learning Path Recommender")
    print("=" * 60)
    
    # Check if running on Streamlit Cloud
    if os.getenv('STREAMLIT_SHARING_MODE') or '--streamlit-only' in sys.argv:
        print("🌐 Running in Streamlit Cloud mode")
        
        # Setup environment
        setup_environment()
        
        # Start backend in thread
        backend_thread = threading.Thread(target=start_backend_server, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Run Streamlit app
        run_streamlit_app()
        
    else:
        print("💻 Running in local development mode")
        
        # Check dependencies
        if not check_dependencies():
            print("❌ Dependency check failed")
            return
        
        # Setup environment
        setup_environment()
        
        # Start backend server in a separate thread
        print("🔧 Starting backend server...")
        backend_thread = threading.Thread(target=start_backend_server, daemon=True)
        backend_thread.start()
        
        # Wait for backend to be ready
        if wait_for_backend():
            # Start Streamlit frontend
            print("🎨 Starting Streamlit frontend...")
            run_streamlit_app()
        else:
            print("❌ Failed to start backend server")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)