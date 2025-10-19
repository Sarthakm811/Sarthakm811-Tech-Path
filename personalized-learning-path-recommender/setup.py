#!/usr/bin/env python3
"""
TechPath - Advanced AI-Powered Learning Platform Setup Script
Automated setup and initialization for the professional AI/ML learning platform
with 50+ comprehensive technical skills and advanced recommendation algorithms
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

class Colors:
    """Terminal colors for better output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_step(step, description):
    """Print a step with formatting"""
    print(f"{Colors.OKBLUE}[{step}] {Colors.BOLD}{description}{Colors.ENDC}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def check_requirements():
    """Check if required tools are installed"""
    print_step("1", "Checking system requirements...")
    
    requirements = {
        'python': 'python3',
        'node': 'node',
        'npm': 'npm',
        'postgres': 'psql'
    }
    
    missing = []
    
    for tool, command in requirements.items():
        try:
            result = subprocess.run([command, '--version'], 
                                  capture_output=True, text=True, check=True)
            print_success(f"{tool.title()} is installed: {result.stdout.split()[1]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
            print_error(f"{tool.title()} is not installed")
    
    if missing:
        print_error(f"Missing requirements: {', '.join(missing)}")
        print("Please install the missing tools and run this script again.")
        sys.exit(1)
    
    print_success("All requirements are satisfied!")

def setup_environment():
    """Setup environment variables"""
    print_step("2", "Setting up environment variables...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("Creating .env file with default values...")
        
        env_content = """# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost/learning_recommender

# AI Services
OPENAI_API_KEY=your-openai-api-key-here

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Frontend Configuration
REACT_APP_API_URL=http://localhost:5000

# Development
DEBUG=True
FLASK_ENV=development
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print_success(".env file created")
        print_warning("Please update the .env file with your actual API keys and database credentials")
    else:
        print_success(".env file already exists")

def setup_database():
    """Setup PostgreSQL database"""
    print_step("3", "Setting up PostgreSQL database...")
    
    try:
        # Create database
        subprocess.run(['createdb', 'learning_recommender'], 
                      check=True, capture_output=True)
        print_success("Database 'learning_recommender' created")
        
        # Test connection
        result = subprocess.run(['psql', '-d', 'learning_recommender', '-c', 'SELECT version();'], 
                              check=True, capture_output=True, text=True)
        print_success("Database connection successful")
        
    except subprocess.CalledProcessError as e:
        print_error(f"Database setup failed: {e}")
        print("Please ensure PostgreSQL is running and you have the necessary permissions")
        return False
    
    return True

def setup_backend():
    """Setup Python backend"""
    print_step("4", "Setting up Python backend...")
    
    backend_dir = Path('backend')
    if not backend_dir.exists():
        print_error("Backend directory not found!")
        return False
    
    # Create virtual environment
    venv_dir = backend_dir / 'venv'
    if not venv_dir.exists():
        print("Creating Python virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
        print_success("Virtual environment created")
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = venv_dir / 'Scripts' / 'activate.bat'
        pip_cmd = str(venv_dir / 'Scripts' / 'pip')
        python_cmd = str(venv_dir / 'Scripts' / 'python')
    else:  # Unix-like
        activate_script = venv_dir / 'bin' / 'activate'
        pip_cmd = str(venv_dir / 'bin' / 'pip')
        python_cmd = str(venv_dir / 'bin' / 'python')
    
    # Install dependencies
    print("Installing Python dependencies...")
    requirements_file = backend_dir / 'requirements.txt'
    if requirements_file.exists():
        subprocess.run([pip_cmd, 'install', '-r', str(requirements_file)], check=True)
        print_success("Python dependencies installed")
    else:
        print_error("requirements.txt not found!")
        return False
    
    # Initialize database models
    print("Initializing database models...")
    try:
        subprocess.run([python_cmd, '-c', 
                       'from database.seed_data import seed_database; seed_database()'], 
                      cwd=backend_dir, check=True)
        print_success("Database models initialized")
    except subprocess.CalledProcessError:
        print_warning("Database initialization failed - this is normal for first run")
    
    return True

def setup_frontend():
    """Setup React frontend"""
    print_step("5", "Setting up React frontend...")
    
    frontend_dir = Path('frontend')
    if not frontend_dir.exists():
        print_error("Frontend directory not found!")
        return False
    
    # Install Node.js dependencies
    print("Installing Node.js dependencies...")
    subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
    print_success("Node.js dependencies installed")
    
    return True

def create_startup_scripts():
    """Create startup scripts for easy development"""
    print_step("6", "Creating startup scripts...")
    
    # Backend startup script
    backend_script = """#!/bin/bash
# Advanced Learning Path Recommender - Backend Startup Script

echo "🚀 Starting Advanced Learning Path Recommender Backend..."

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.py first."
    exit 1
fi

# Activate virtual environment
source backend/venv/bin/activate

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the backend server
cd backend
python app.py
"""
    
    with open('start_backend.sh', 'w') as f:
        f.write(backend_script)
    os.chmod('start_backend.sh', 0o755)
    
    # Frontend startup script
    frontend_script = """#!/bin/bash
# Advanced Learning Path Recommender - Frontend Startup Script

echo "🚀 Starting Advanced Learning Path Recommender Frontend..."

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Node modules not found. Please run setup.py first."
    exit 1
fi

# Start the frontend development server
cd frontend
npm start
"""
    
    with open('start_frontend.sh', 'w') as f:
        f.write(frontend_script)
    os.chmod('start_frontend.sh', 0o755)
    
    # Windows batch files
    backend_bat = """@echo off
echo 🚀 Starting Advanced Learning Path Recommender Backend...

if not exist "backend\\venv" (
    echo ❌ Virtual environment not found. Please run setup.py first.
    pause
    exit /b 1
)

call backend\\venv\\Scripts\\activate.bat

cd backend
python app.py
pause
"""
    
    with open('start_backend.bat', 'w') as f:
        f.write(backend_bat)
    
    frontend_bat = """@echo off
echo 🚀 Starting Advanced Learning Path Recommender Frontend...

if not exist "frontend\\node_modules" (
    echo ❌ Node modules not found. Please run setup.py first.
    pause
    exit /b 1
)

cd frontend
npm start
pause
"""
    
    with open('start_frontend.bat', 'w') as f:
        f.write(frontend_bat)
    
    print_success("Startup scripts created")

def create_docker_setup():
    """Create Docker configuration"""
    print_step("7", "Creating Docker configuration...")
    
    # Docker Compose file
    docker_compose = """version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: learning_recommender
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/learning_recommender
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
    command: python app.py

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:5000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app

volumes:
  postgres_data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    
    # Backend Dockerfile
    backend_dockerfile = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]
"""
    
    with open('backend/Dockerfile', 'w') as f:
        f.write(backend_dockerfile)
    
    # Frontend Dockerfile
    frontend_dockerfile = """FROM node:16-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start the development server
CMD ["npm", "start"]
"""
    
    with open('frontend/Dockerfile', 'w') as f:
        f.write(frontend_dockerfile)
    
    print_success("Docker configuration created")

def print_completion_message():
    """Print completion message with next steps"""
    print_header("🎉 SETUP COMPLETED SUCCESSFULLY!")
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}Your Advanced AI-Powered Learning Path Recommender is ready!{Colors.ENDC}")
    
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("1. Update the .env file with your actual API keys")
    print("2. Start the backend server:")
    if os.name == 'nt':
        print("   - Double-click start_backend.bat")
        print("   - Or run: start_backend.bat")
    else:
        print("   - Run: ./start_backend.sh")
        print("   - Or run: bash start_backend.sh")
    
    print("3. Start the frontend server (in a new terminal):")
    if os.name == 'nt':
        print("   - Double-click start_frontend.bat")
        print("   - Or run: start_frontend.bat")
    else:
        print("   - Run: ./start_frontend.sh")
        print("   - Or run: bash start_frontend.sh")
    
    print("4. Open your browser and go to:")
    print("   - Frontend: http://localhost:3000")
    print("   - Backend API: http://localhost:5000")
    print("   - Health Check: http://localhost:5000/api/health")
    
    print(f"\n{Colors.WARNING}{Colors.BOLD}Important Notes:{Colors.ENDC}")
    print("- Make sure PostgreSQL is running")
    print("- Get your OpenAI API key from https://platform.openai.com/")
    print("- Update the .env file with your actual credentials")
    print("- For production deployment, see the README.md file")
    
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}For Docker deployment:{Colors.ENDC}")
    print("- Run: docker-compose up -d")
    print("- This will start all services automatically")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}Happy Learning! 🚀{Colors.ENDC}")

def main():
    """Main setup function"""
    print_header("🚀 ADVANCED LEARNING PATH RECOMMENDER SETUP")
    
    print(f"{Colors.OKCYAN}This script will set up your advanced AI-powered learning platform{Colors.ENDC}")
    print(f"{Colors.OKCYAN}with machine learning, user modeling, and intelligent recommendations.{Colors.ENDC}")
    
    # Check if running from correct directory
    if not Path('backend').exists() or not Path('frontend').exists():
        print_error("Please run this script from the project root directory")
        sys.exit(1)
    
    try:
        # Run setup steps
        check_requirements()
        setup_environment()
        
        if setup_database():
            setup_backend()
            setup_frontend()
            create_startup_scripts()
            create_docker_setup()
            print_completion_message()
        else:
            print_error("Setup failed due to database issues")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
