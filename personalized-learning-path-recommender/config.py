#!/usr/bin/env python3
"""
LearnAI - Advanced AI-Powered Learning Platform Configuration
Capstone-level AI/ML project configuration settings
"""

import os
from pathlib import Path

# Project Information
PROJECT_NAME = "LearnAI"
PROJECT_DESCRIPTION = "Advanced AI-Powered Learning Platform"
PROJECT_VERSION = "2.0.0"
PROJECT_TYPE = "Capstone AI/ML Project"

# Application Settings
class Config:
    """Base configuration class"""
    
    # Core Application
    SECRET_KEY = os.getenv('SECRET_KEY', 'learnai-advanced-ai-ml-platform-2024')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///learnai.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI/ML Model Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Recommendation Engine Settings
    MAX_RECOMMENDATIONS = 10
    MIN_CONFIDENCE_SCORE = 0.6
    
    # ML Model Weights
    COLLABORATIVE_FILTER_WEIGHT = 0.25
    CONTENT_BASED_WEIGHT = 0.25
    NEURAL_NETWORK_WEIGHT = 0.25
    ADAPTIVE_LEARNING_WEIGHT = 0.25
    
    # Learning Analytics
    ENABLE_ANALYTICS = True
    TRACK_USER_BEHAVIOR = True
    
    # Performance Settings
    CACHE_TIMEOUT = 300  # 5 minutes
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Security Settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET', 'learnai-jwt-secret-2024')
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Feature Flags
    ENABLE_CHATBOT = True
    ENABLE_ADVANCED_ANALYTICS = True
    ENABLE_KNOWLEDGE_GRAPH = True
    ENABLE_NEURAL_RECOMMENDATIONS = True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Technical Skills Categories
SKILL_CATEGORIES = {
    'hot_emerging': [
        'ai_ml', 'blockchain', 'quantum', 'ar_vr', 'iot', 
        'edge_computing', 'metaverse'
    ],
    'programming': [
        'python', 'javascript', 'java', 'cpp', 'rust', 
        'go', 'typescript', 'swift', 'kotlin'
    ],
    'web_mobile': [
        'web_development', 'frontend', 'backend', 'fullstack', 
        'mobile', 'react', 'vue', 'angular'
    ],
    'infrastructure': [
        'infrastructure', 'devops', 'cloud', 'kubernetes', 
        'docker', 'terraform', 'ansible'
    ],
    'data_analytics': [
        'data_science', 'data_engineering', 'analytics', 
        'visualization', 'big_data', 'database'
    ],
    'security_specialized': [
        'cybersecurity', 'ethical_hacking', 'game_development', 
        'fintech', 'healthtech', 'computer_science'
    ]
}

# Career Path Mappings
CAREER_PATHS = {
    'ai_ml_engineer': {
        'title': 'AI/ML Engineer',
        'skills': ['python', 'ai_ml', 'data_science', 'tensorflow', 'pytorch'],
        'salary_range': '$120k - $200k+',
        'growth_rate': 'Very High'
    },
    'full_stack_developer': {
        'title': 'Full Stack Developer',
        'skills': ['javascript', 'react', 'nodejs', 'database', 'cloud'],
        'salary_range': '$80k - $150k+',
        'growth_rate': 'High'
    },
    'devops_engineer': {
        'title': 'DevOps Engineer',
        'skills': ['kubernetes', 'docker', 'terraform', 'cloud', 'ansible'],
        'salary_range': '$100k - $180k+',
        'growth_rate': 'Very High'
    },
    'data_scientist': {
        'title': 'Data Scientist',
        'skills': ['python', 'data_science', 'statistics', 'ml', 'visualization'],
        'salary_range': '$110k - $190k+',
        'growth_rate': 'Very High'
    },
    'cybersecurity_specialist': {
        'title': 'Cybersecurity Specialist',
        'skills': ['cybersecurity', 'ethical_hacking', 'network_security', 'incident_response'],
        'salary_range': '$90k - $170k+',
        'growth_rate': 'High'
    }
}

# Application Metadata
APP_METADATA = {
    'name': PROJECT_NAME,
    'description': PROJECT_DESCRIPTION,
    'version': PROJECT_VERSION,
    'type': PROJECT_TYPE,
    'features': [
        '50+ Comprehensive Technical Skills',
        'Advanced AI Recommendation Engine',
        'Machine Learning Algorithms',
        'Personalized Learning Paths',
        'Career Guidance System',
        'Progress Analytics',
        'Resource Integration',
        'Project-Based Learning'
    ],
    'technologies': [
        'Streamlit', 'Flask', 'scikit-learn', 'TensorFlow',
        'NetworkX', 'Plotly', 'Pandas', 'NumPy'
    ]
}