#!/usr/bin/env python3
"""
TechPath - Deployment Script
Automated deployment for the Advanced AI-Powered Learning Platform
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print deployment banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                        TechPath                              ║
    ║            Advanced AI-Powered Learning Platform             ║
    ║                  Deployment Script v2.0                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        'python': ['python', '--version'],
        'pip': ['pip', '--version'],
        'streamlit': ['streamlit', '--version']
    }
    
    missing = []
    for tool, command in requirements.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"✅ {tool}: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
            print(f"❌ {tool}: Not found")
    
    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def install_dependencies():
    """Install project dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install main requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Main dependencies installed")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# TechPath Configuration
DEBUG=True
SECRET_KEY=techpath-development-key-2024
DATABASE_URL=sqlite:///techpath.db
OPENAI_API_KEY=your-openai-api-key-here

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Environment file created")
    else:
        print("✅ Environment file already exists")

def deploy_streamlit():
    """Deploy Streamlit application"""
    print("\n🚀 Deploying TechPath Streamlit Application...")
    
    try:
        print("Starting Streamlit server...")
        print("🌐 Application will be available at: http://localhost:8501")
        print("📝 Use Ctrl+C to stop the server")
        
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'], 
                      check=True)
    except KeyboardInterrupt:
        print("\n👋 Deployment stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")

def deploy_full_stack():
    """Deploy full stack application (backend + frontend)"""
    print("\n🚀 Deploying Full Stack TechPath Application...")
    
    # This would start both backend and frontend
    # For now, we'll just run the Streamlit version
    deploy_streamlit()

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='TechPath Deployment Script')
    parser.add_argument('--mode', choices=['streamlit', 'fullstack'], 
                       default='streamlit', help='Deployment mode')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Deploy based on mode
    if args.mode == 'streamlit':
        deploy_streamlit()
    elif args.mode == 'fullstack':
        deploy_full_stack()
    
    print("\n🎉 Deployment completed!")

if __name__ == "__main__":
    main()