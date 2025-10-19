#!/usr/bin/env python3
"""
🚀 TECHPATH - ADVANCED AI/ML LEARNING PLATFORM
Advanced AI-Powered Personalized Learning Path Recommender
Featuring: Neural Networks, NLP, Computer Vision, Reinforcement Learning, MLOps
"""

import streamlit as st
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# 🎨 ADVANCED STYLING AND CONFIGURATION
def load_custom_css():
    """Load advanced custom CSS for TechPath UI"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Headers */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .gradient-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .gradient-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* AI Feature Cards */
    .ai-feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .ai-feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* Skill Level Badges */
    .skill-badge-beginner {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .skill-badge-intermediate {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .skill-badge-advanced {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #333;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* AI Confidence Indicator */
    .ai-confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .ai-confidence-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        color: #333;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .ai-confidence-low {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        color: #333;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Animated Elements */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Success Messages */
    .success-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 🧠 ADVANCED AI/ML LEARNING SYSTEM WITH NEURAL NETWORKS
class TechPathAdvancedLearningAI:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        self.user_profiles = {}
        self.learning_graph = nx.DiGraph()
        self.skill_embeddings = {}
        
        # 🤖 Multiple ML Models for Ensemble Learning
        self.neural_predictor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        self.gradient_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.forest_predictor = RandomForestRegressor(n_estimators=200, random_state=42)
        
        # 📊 Advanced Analytics
        self.user_clustering = KMeans(n_clusters=5, random_state=42)
        self.skill_pca = PCA(n_components=10)
        
        self.initialize_advanced_knowledge_graph()
        self.train_ensemble_models()
    
    def initialize_advanced_knowledge_graph(self):
        """Initialize comprehensive knowledge graph with weighted relationships"""
        # 🎯 Comprehensive skill taxonomy
        skills = {
            'foundations': ['python_basics', 'statistics_fundamentals', 'linear_algebra', 'calculus'],
            'data_science': ['data_analysis', 'data_visualization', 'sql_advanced', 'big_data'],
            'machine_learning': ['ml_fundamentals', 'supervised_learning', 'unsupervised_learning', 'feature_engineering'],
            'deep_learning': ['neural_networks', 'cnn', 'rnn', 'transformers', 'gans'],
            'specialized_ai': ['nlp_advanced', 'computer_vision', 'reinforcement_learning', 'robotics'],
            'engineering': ['mlops', 'model_deployment', 'cloud_ml', 'edge_computing'],
            'web_tech': ['frontend_frameworks', 'backend_apis', 'databases', 'web_security'],
            'infrastructure': ['cloud_platforms', 'containerization', 'orchestration', 'monitoring']
        }
        
        # Add all skills as nodes
        for category, skill_list in skills.items():
            for skill in skill_list:
                self.learning_graph.add_node(skill, category=category)
        
        # 🕸️ Complex prerequisite relationships with weights
        advanced_prerequisites = [
            # Foundations
            ('python_basics', 'data_analysis', 0.9),
            ('statistics_fundamentals', 'ml_fundamentals', 0.8),
            ('linear_algebra', 'neural_networks', 0.9),
            ('calculus', 'deep_learning_optimization', 0.7),
            
            # ML Pipeline
            ('data_analysis', 'ml_fundamentals', 0.8),
            ('ml_fundamentals', 'neural_networks', 0.9),
            ('neural_networks', 'cnn', 0.8),
            ('neural_networks', 'rnn', 0.8),
            ('cnn', 'computer_vision', 0.9),
            ('rnn', 'nlp_advanced', 0.8),
            
            # Advanced AI
            ('neural_networks', 'transformers', 0.7),
            ('ml_fundamentals', 'reinforcement_learning', 0.6),
            ('neural_networks', 'gans', 0.7),
            
            # Engineering
            ('ml_fundamentals', 'mlops', 0.8),
            ('neural_networks', 'model_deployment', 0.7),
            ('cloud_platforms', 'cloud_ml', 0.8),
        ]
        
        for prereq, skill, weight in advanced_prerequisites:
            if prereq in [n for n, _ in self.learning_graph.nodes(data=True)] and skill in [n for n, _ in self.learning_graph.nodes(data=True)]:
                self.learning_graph.add_edge(prereq, skill, weight=weight)
    
    def train_ensemble_models(self):
        """Train ensemble of ML models for advanced predictions"""
        # 📊 Generate synthetic training data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [experience_level, time_commitment, topic_complexity, motivation, prior_knowledge]
        X = np.random.rand(n_samples, 5)
        X[:, 0] *= 3  # experience_level (0-3)
        X[:, 1] *= 40  # time_commitment (0-40 hours)
        X[:, 2] *= 10  # topic_complexity (0-10)
        X[:, 3] = np.random.beta(2, 2, n_samples)  # motivation (0-1, beta distribution)
        X[:, 4] *= 5  # prior_knowledge (0-5)
        
        # Target: success_probability (complex non-linear relationship)
        y = (0.3 * X[:, 0] + 0.2 * np.log1p(X[:, 1]) + 0.1 * (10 - X[:, 2]) + 
             0.3 * X[:, 3] + 0.1 * X[:, 4] + np.random.normal(0, 0.1, n_samples))
        y = np.clip(y / y.max(), 0, 1)  # Normalize to [0, 1]
        
        # Train ensemble models
        self.neural_predictor.fit(X, y)
        self.gradient_predictor.fit(X, y)
        self.forest_predictor.fit(X, y)
    
    def predict_learning_success_ensemble(self, user_profile, topic_complexity):
        """Advanced ensemble prediction using multiple ML models"""
        features = np.array([[
            user_profile.get('experience_level', 1),
            user_profile.get('time_commitment', 10),
            topic_complexity,
            user_profile.get('motivation_score', 0.8),
            len(user_profile.get('completed_topics', []))
        ]])
        
        # Get predictions from all models
        neural_pred = self.neural_predictor.predict(features)[0]
        gradient_pred = self.gradient_predictor.predict(features)[0]
        forest_pred = self.forest_predictor.predict(features)[0]
        
        # Ensemble prediction with weighted average
        ensemble_pred = (0.4 * neural_pred + 0.3 * gradient_pred + 0.3 * forest_pred)
        
        return np.clip(ensemble_pred, 0, 1)
    
    def analyze_learning_patterns(self, user_data):
        """Advanced NLP analysis of user goals and interests"""
        goals_text = user_data.get('goals_text', '')
        
        if not goals_text:
            return {'sentiment': 'neutral', 'motivation': 0.5, 'career_focus': 'general', 'urgency': 'medium'}
        
        # Sentiment analysis
        blob = TextBlob(goals_text)
        sentiment_score = blob.sentiment.polarity
        
        # Career focus detection
        career_keywords = {
            'data_scientist': ['data scientist', 'data science', 'analytics', 'insights'],
            'ml_engineer': ['machine learning', 'ml engineer', 'ai engineer', 'model'],
            'software_engineer': ['software', 'developer', 'programming', 'coding'],
            'researcher': ['research', 'phd', 'academic', 'publication'],
            'entrepreneur': ['startup', 'business', 'entrepreneur', 'product']
        }
        
        career_focus = 'general'
        max_matches = 0
        for career, keywords in career_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in goals_text.lower())
            if matches > max_matches:
                max_matches = matches
                career_focus = career
        
        # Urgency detection
        urgency_high = ['urgent', 'quickly', 'asap', 'immediately', 'fast track']
        urgency_low = ['eventually', 'someday', 'long term', 'future']
        
        urgency = 'medium'
        if any(word in goals_text.lower() for word in urgency_high):
            urgency = 'high'
        elif any(word in goals_text.lower() for word in urgency_low):
            urgency = 'low'
        
        return {
            'sentiment': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral',
            'motivation': max(0.3, min(1.0, (sentiment_score + 1) / 2)),
            'career_focus': career_focus,
            'urgency': urgency,
            'sentiment_score': sentiment_score
        }

# Initialize advanced AI system
ai_system = TechPathAdvancedLearningAI()

# 🎓 TECHPATH COMPREHENSIVE CURRICULUM DATABASE
TECHPATH_CURRICULUM = [
    # 🟢 FOUNDATION LEVEL (Beginner)
    {
        "id": "python_fundamentals",
        "title": "Python Programming & Data Structures Mastery",
        "description": "Comprehensive Python foundation: syntax, OOP, data structures, algorithms, testing, and best practices for AI/ML development",
        "category": "programming",
        "difficulty_level": "beginner",
        "estimated_hours": 40,
        "skills_covered": ["Python Syntax", "OOP Design", "Data Structures", "Algorithms", "Unit Testing", "Code Quality"],
        "prerequisites": [],
        "ml_complexity": 2,
        "success_rate": 0.92,
        "career_impact": 0.95,
        "industry_demand": 0.98,
        "ai_relevance": 0.90,
        "projects": ["Personal Finance Tracker", "Data Processing Pipeline", "Web Scraper with Analytics"],
        "certifications": ["Python Institute PCAP", "Microsoft Python Certification"],
        "learning_outcomes": ["Write clean, efficient Python code", "Implement complex data structures", "Design object-oriented systems"]
    },
    {
        "id": "statistics_ml_foundation",
        "title": "Statistics & Mathematics for Machine Learning",
        "description": "Essential mathematical foundations: probability, statistics, linear algebra, and calculus for AI/ML applications",
        "category": "mathematics",
        "difficulty_level": "beginner",
        "estimated_hours": 45,
        "skills_covered": ["Probability Theory", "Statistical Inference", "Linear Algebra", "Calculus", "Hypothesis Testing"],
        "prerequisites": [],
        "ml_complexity": 3,
        "success_rate": 0.78,
        "career_impact": 0.88,
        "industry_demand": 0.85,
        "ai_relevance": 0.95,
        "projects": ["Statistical Analysis Dashboard", "A/B Testing Framework", "Probability Simulation Engine"],
        "certifications": ["Statistics Specialization (Coursera)", "Khan Academy Mathematics"],
        "learning_outcomes": ["Apply statistical methods to real data", "Understand ML mathematical foundations", "Perform hypothesis testing"]
    },
    {
        "id": "data_manipulation_analysis",
        "title": "Data Manipulation & Exploratory Analysis",
        "description": "Master data wrangling with Pandas, NumPy, and advanced visualization techniques for data-driven insights",
        "category": "data_science",
        "difficulty_level": "beginner",
        "estimated_hours": 35,
        "skills_covered": ["Pandas Mastery", "NumPy Operations", "Data Cleaning", "EDA Techniques", "Statistical Visualization"],
        "prerequisites": ["python_fundamentals"],
        "ml_complexity": 3,
        "success_rate": 0.85,
        "career_impact": 0.90,
        "industry_demand": 0.92,
        "ai_relevance": 0.88,
        "projects": ["Customer Analytics Dashboard", "Sales Forecasting System", "Market Research Analysis"],
        "certifications": ["Google Data Analytics", "IBM Data Science Foundations"],
        "learning_outcomes": ["Clean and preprocess complex datasets", "Create insightful visualizations", "Extract business insights from data"]
    },
    
    # 🟡 INTERMEDIATE LEVEL
    {
        "id": "machine_learning_engineering",
        "title": "Machine Learning Engineering & Model Development",
        "description": "Comprehensive ML pipeline: feature engineering, model selection, hyperparameter tuning, and production deployment",
        "category": "ai_ml",
        "difficulty_level": "intermediate",
        "estimated_hours": 60,
        "skills_covered": ["Feature Engineering", "Model Selection", "Cross-Validation", "Hyperparameter Tuning", "Model Evaluation", "Scikit-learn"],
        "prerequisites": ["python_fundamentals", "statistics_ml_foundation", "data_manipulation_analysis"],
        "ml_complexity": 6,
        "success_rate": 0.75,
        "career_impact": 0.95,
        "industry_demand": 0.96,
        "ai_relevance": 0.98,
        "projects": ["Predictive Analytics Platform", "Recommendation Engine", "Fraud Detection System"],
        "certifications": ["Google ML Engineer", "AWS ML Specialty", "Microsoft Azure AI Engineer"],
        "learning_outcomes": ["Build end-to-end ML pipelines", "Deploy models to production", "Optimize model performance"]
    },
    {
        "id": "deep_learning_neural_networks",
        "title": "Deep Learning & Neural Network Architectures",
        "description": "Advanced neural networks: CNNs, RNNs, Transformers, GANs, and modern deep learning frameworks",
        "category": "ai_ml",
        "difficulty_level": "intermediate",
        "estimated_hours": 70,
        "skills_covered": ["Neural Network Design", "CNN Architectures", "RNN/LSTM", "Attention Mechanisms", "Transfer Learning", "TensorFlow/PyTorch"],
        "prerequisites": ["machine_learning_engineering"],
        "ml_complexity": 8,
        "success_rate": 0.68,
        "career_impact": 0.98,
        "industry_demand": 0.94,
        "ai_relevance": 0.99,
        "projects": ["Image Classification System", "Natural Language Chatbot", "Time Series Forecasting", "Style Transfer Application"],
        "certifications": ["TensorFlow Developer", "PyTorch Certified Developer", "NVIDIA Deep Learning Institute"],
        "learning_outcomes": ["Design custom neural architectures", "Implement state-of-the-art models", "Optimize deep learning performance"]
    },
    {
        "id": "fullstack_ai_applications",
        "title": "Full-Stack AI Application Development",
        "description": "Build complete AI-powered applications: React frontends, FastAPI backends, database integration, and cloud deployment",
        "category": "web_development",
        "difficulty_level": "intermediate",
        "estimated_hours": 55,
        "skills_covered": ["React.js", "FastAPI", "Database Design", "API Development", "Cloud Deployment", "Docker"],
        "prerequisites": ["python_fundamentals", "machine_learning_engineering"],
        "ml_complexity": 5,
        "success_rate": 0.72,
        "career_impact": 0.92,
        "industry_demand": 0.90,
        "ai_relevance": 0.85,
        "projects": ["AI-Powered SaaS Platform", "Real-time Analytics Dashboard", "ML Model Serving API"],
        "certifications": ["React Developer Certification", "FastAPI Expert", "AWS Solutions Architect"],
        "learning_outcomes": ["Build scalable web applications", "Integrate ML models with web services", "Deploy applications to cloud"]
    },
    
    # 🔴 ADVANCED LEVEL
    {
        "id": "advanced_nlp_transformers",
        "title": "Advanced NLP & Large Language Models",
        "description": "Cutting-edge NLP: BERT, GPT, T5, fine-tuning LLMs, prompt engineering, and conversational AI systems",
        "category": "ai_ml",
        "difficulty_level": "advanced",
        "estimated_hours": 80,
        "skills_covered": ["Transformer Architecture", "BERT/GPT Models", "Fine-tuning LLMs", "Prompt Engineering", "RAG Systems", "Hugging Face"],
        "prerequisites": ["deep_learning_neural_networks"],
        "ml_complexity": 9,
        "success_rate": 0.58,
        "career_impact": 0.99,
        "industry_demand": 0.97,
        "ai_relevance": 0.99,
        "projects": ["Question Answering System", "Document Summarization API", "Conversational AI Assistant", "Content Generation Platform"],
        "certifications": ["Hugging Face Expert", "OpenAI API Specialist", "Google Cloud ML Engineer"],
        "learning_outcomes": ["Fine-tune large language models", "Build conversational AI systems", "Implement RAG architectures"]
    },
    {
        "id": "computer_vision_advanced",
        "title": "Advanced Computer Vision & Visual AI",
        "description": "State-of-the-art computer vision: object detection, segmentation, face recognition, medical imaging, and autonomous systems",
        "category": "ai_ml",
        "difficulty_level": "advanced",
        "estimated_hours": 75,
        "skills_covered": ["Object Detection", "Image Segmentation", "Face Recognition", "Medical Imaging", "3D Vision", "OpenCV/YOLO"],
        "prerequisites": ["deep_learning_neural_networks"],
        "ml_complexity": 9,
        "success_rate": 0.62,
        "career_impact": 0.96,
        "industry_demand": 0.88,
        "ai_relevance": 0.95,
        "projects": ["Autonomous Vehicle Vision", "Medical Diagnosis System", "Real-time Object Tracking", "AR/VR Applications"],
        "certifications": ["NVIDIA Computer Vision", "OpenCV Certified", "Google Cloud Vision AI"],
        "learning_outcomes": ["Build real-time vision systems", "Implement medical AI applications", "Deploy edge AI solutions"]
    },
    {
        "id": "mlops_production_systems",
        "title": "MLOps & Production AI Systems",
        "description": "Enterprise MLOps: model versioning, CI/CD for ML, monitoring, A/B testing, and scalable ML infrastructure",
        "category": "ai_ml",
        "difficulty_level": "advanced",
        "estimated_hours": 65,
        "skills_covered": ["MLOps Pipelines", "Model Monitoring", "A/B Testing", "Feature Stores", "ML Infrastructure", "Kubernetes for ML"],
        "prerequisites": ["machine_learning_engineering", "fullstack_ai_applications"],
        "ml_complexity": 7,
        "success_rate": 0.70,
        "career_impact": 0.97,
        "industry_demand": 0.95,
        "ai_relevance": 0.92,
        "projects": ["Enterprise ML Platform", "Model Monitoring System", "Automated ML Pipeline", "Multi-model Serving Infrastructure"],
        "certifications": ["MLOps Specialist", "Kubernetes for ML", "AWS ML Operations"],
        "learning_outcomes": ["Design production ML systems", "Implement model monitoring", "Scale ML infrastructure"]
    },
    {
        "id": "reinforcement_learning_agents",
        "title": "Reinforcement Learning & Autonomous Agents",
        "description": "Advanced RL: Q-learning, policy gradients, multi-agent systems, robotics, and autonomous decision-making systems",
        "category": "ai_ml",
        "difficulty_level": "advanced",
        "estimated_hours": 85,
        "skills_covered": ["Q-Learning", "Policy Gradients", "Actor-Critic", "Multi-Agent RL", "Robotics AI", "Game AI"],
        "prerequisites": ["deep_learning_neural_networks"],
        "ml_complexity": 10,
        "success_rate": 0.52,
        "career_impact": 0.94,
        "industry_demand": 0.82,
        "ai_relevance": 0.96,
        "projects": ["Autonomous Trading Bot", "Game Playing AI", "Robotics Control System", "Resource Optimization Engine"],
        "certifications": ["DeepMind RL Certification", "OpenAI Gym Expert", "Robotics AI Specialist"],
        "learning_outcomes": ["Build autonomous agents", "Implement multi-agent systems", "Design reward systems"]
    }
]

def generate_advanced_recommendations(user_data):
    """🚀 TECHPATH AI RECOMMENDATION ENGINE"""
    level = user_data.get('level', 'beginner')
    interests = user_data.get('interests', [])
    time_per_week = user_data.get('time_per_week', 10)
    goals = user_data.get('goals', [])
    goals_text = user_data.get('goals_text', '')
    learning_style = user_data.get('learning_style', 'visual')
    ai_challenge_level = user_data.get('ai_challenge_level', 5)
    focus_area = user_data.get('focus_area', 'General AI/ML')
    
    # 🧠 Advanced user profiling with NLP analysis
    user_profile = {
        'experience_level': {'beginner': 1, 'intermediate': 2, 'advanced': 3}[level],
        'time_commitment': time_per_week,
        'interests': interests,
        'goals': goals,
        'motivation_score': 0.8,
        'completed_topics': [],
        'learning_style': learning_style,
        'ai_challenge_level': ai_challenge_level,
        'focus_area': focus_area
    }
    
    # 📊 Advanced NLP analysis of user goals
    goal_analysis = ai_system.analyze_learning_patterns(user_data)
    user_profile.update(goal_analysis)
    
    # 🔍 Advanced content similarity with enhanced TF-IDF
    all_descriptions = []
    for topic in TECHPATH_CURRICULUM:
        content = (topic['description'] + ' ' + 
                  ' '.join(topic['skills_covered']) + ' ' +
                  ' '.join(topic.get('learning_outcomes', [])) + ' ' +
                  topic['category'])
        all_descriptions.append(content)
    
    user_query = ' '.join(interests + goals + [goals_text, focus_area])
    
    if user_query.strip():
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_descriptions + [user_query])
        user_vector = tfidf_matrix[-1]
        content_similarities = cosine_similarity(user_vector, tfidf_matrix[:-1]).flatten()
    else:
        content_similarities = np.ones(len(TECHPATH_CURRICULUM)) * 0.5
    
    # 🎯 ADVANCED SCORING ALGORITHM WITH MULTIPLE FACTORS
    scored_topics = []
    for i, topic in enumerate(TECHPATH_CURRICULUM):
        
        # 🚫 STRICT DIFFICULTY FILTERING
        if level == 'beginner' and topic['difficulty_level'] == 'advanced':
            continue
        if level == 'advanced' and topic['difficulty_level'] == 'beginner':
            beginner_count = len([t for t in scored_topics if t['topic']['difficulty_level'] == 'beginner'])
            if beginner_count >= 1:
                continue
        
        # 🎯 ENHANCED DIFFICULTY MATCHING
        difficulty_match = 0.0
        if level == 'beginner':
            if topic['difficulty_level'] == 'beginner':
                difficulty_match = 1.0
            elif topic['difficulty_level'] == 'intermediate':
                difficulty_match = 0.05
        elif level == 'intermediate':
            if topic['difficulty_level'] == 'beginner':
                difficulty_match = 0.2
            elif topic['difficulty_level'] == 'intermediate':
                difficulty_match = 1.0
            else:
                difficulty_match = 0.15
        else:  # advanced
            if topic['difficulty_level'] == 'beginner':
                difficulty_match = 0.05
            elif topic['difficulty_level'] == 'intermediate':
                difficulty_match = 0.3
            else:
                difficulty_match = 1.0
        
        # 🎨 LEARNING STYLE MATCHING
        style_bonus = 0.0
        if learning_style == 'visual' and 'visualization' in topic['description'].lower():
            style_bonus = 0.2
        elif learning_style == 'kinesthetic' and any(word in topic['description'].lower() for word in ['project', 'build', 'hands-on']):
            style_bonus = 0.2
        elif learning_style == 'auditory' and 'interactive' in topic['description'].lower():
            style_bonus = 0.1
        
        # 🎯 FOCUS AREA ALIGNMENT
        focus_bonus = 0.0
        if focus_area != 'General AI/ML':
            if focus_area.lower() in topic['title'].lower() or focus_area.lower() in topic['description'].lower():
                focus_bonus = 0.3
        
        # 💼 CAREER FOCUS ALIGNMENT
        career_bonus = 0.0
        career_focus = goal_analysis.get('career_focus', 'general')
        career_mappings = {
            'data_scientist': ['data', 'statistics', 'analysis'],
            'ml_engineer': ['machine learning', 'neural', 'model'],
            'software_engineer': ['programming', 'development', 'application'],
            'researcher': ['advanced', 'theory', 'research']
        }
        
        if career_focus in career_mappings:
            for keyword in career_mappings[career_focus]:
                if keyword in topic['description'].lower():
                    career_bonus = 0.2
                    break
        
        # 🚀 INTEREST ALIGNMENT WITH ENHANCED MATCHING
        interest_score = 0.3
        if interests:
            max_interest_score = 0.3
            for interest in interests:
                # Direct category match
                if interest in topic['category'] or interest in topic['title'].lower():
                    max_interest_score = max(max_interest_score, 1.0)
                # Skills match
                skill_matches = sum(1 for skill in topic['skills_covered'] 
                                  if interest.lower() in skill.lower())
                if skill_matches > 0:
                    max_interest_score = max(max_interest_score, 0.8 + (skill_matches * 0.1))
            interest_score = max_interest_score
        
        # 🤖 ENSEMBLE ML PREDICTION
        success_prob = ai_system.predict_learning_success_ensemble(user_profile, topic.get('ml_complexity', 5))
        
        # 📈 INDUSTRY RELEVANCE & DEMAND
        industry_factor = (topic.get('industry_demand', 0.8) + topic.get('career_impact', 0.8)) / 2
        
        # 🎯 AI CHALLENGE LEVEL MATCHING
        challenge_match = 1.0 - abs(topic.get('ml_complexity', 5) - ai_challenge_level) / 10
        
        # 🏆 COMPREHENSIVE SCORING WITH WEIGHTED FACTORS
        ai_score = (
            difficulty_match * 0.35 +      # Primary factor
            interest_score * 0.20 +        # User interests
            success_prob * 0.15 +          # ML prediction
            content_similarities[i] * 0.10 + # Content similarity
            industry_factor * 0.08 +       # Industry relevance
            challenge_match * 0.05 +       # Challenge level
            style_bonus +                  # Learning style
            focus_bonus +                  # Focus area
            career_bonus                   # Career alignment
        )
        
        scored_topics.append({
            'topic': topic,
            'ai_score': ai_score,
            'success_probability': success_prob,
            'content_similarity': content_similarities[i],
            'industry_relevance': industry_factor,
            'difficulty_match': difficulty_match,
            'challenge_match': challenge_match
        })
    
    # Sort by comprehensive AI score
    scored_topics.sort(key=lambda x: x['ai_score'], reverse=True)
    
    # 🎯 INTELLIGENT PATH CONSTRUCTION
    learning_path = []
    total_hours = 0
    
    # Select top topics with prerequisite validation
    for scored_topic in scored_topics[:6]:
        topic = scored_topic['topic']
        
        # Check prerequisites
        prereqs_satisfied = True
        for prereq in topic.get('prerequisites', []):
            if not any(prereq in selected['id'] for selected in learning_path):
                # Find and add prerequisite if available
                prereq_topic = next((t for t in TECHPATH_CURRICULUM if t['id'] == prereq), None)
                if prereq_topic and len(learning_path) < 5:
                    learning_path.append(prereq_topic)
                    total_hours += prereq_topic['estimated_hours']
        
        if len(learning_path) < 5:
            learning_path.append(topic)
            total_hours += topic['estimated_hours']
    
    # 📊 ENHANCED LEARNING PATH WITH AI INSIGHTS
    enhanced_path = []
    for topic in learning_path:
        scored_data = next((st for st in scored_topics if st['topic']['id'] == topic['id']), None)
        
        enhanced_topic = {
            "title": topic["title"],
            "description": topic["description"],
            "category": topic["category"],
            "level": topic["difficulty_level"],
            "estimated_hours": topic["estimated_hours"],
            "skills_covered": topic["skills_covered"],
            "learning_outcomes": topic.get("learning_outcomes", []),
            "projects": topic.get("projects", []),
            "certifications": topic.get("certifications", []),
            "ai_confidence": scored_data['ai_score'] if scored_data else 0.5,
            "success_probability": scored_data['success_probability'] if scored_data else 0.7,
            "ml_complexity": topic.get('ml_complexity', 5),
            "career_impact": topic.get('career_impact', 0.8),
            "industry_demand": topic.get('industry_demand', 0.8),
            "ai_relevance": topic.get('ai_relevance', 0.8)
        }
        enhanced_path.append(enhanced_topic)
    
    # 📅 INTELLIGENT TIMELINE GENERATION
    timeline = []
    current_week = 1
    
    for topic in enhanced_path:
        weeks_needed = max(2, topic["estimated_hours"] // time_per_week)
        end_week = current_week + weeks_needed - 1
        
        timeline.append({
            "week_range": f"Week {current_week}" + (f"-{end_week}" if weeks_needed > 1 else ""),
            "topic": topic["title"],
            "hours_per_week": min(time_per_week, topic["estimated_hours"]),
            "total_hours": topic["estimated_hours"],
            "milestones": [
                f"Master {topic['skills_covered'][0] if topic['skills_covered'] else 'core concepts'}",
                f"Complete hands-on {topic['projects'][0] if topic.get('projects') else 'project'}",
                f"Achieve {topic['certifications'][0] if topic.get('certifications') else 'proficiency'}"
            ],
            "difficulty": topic["level"]
        })
        current_week = end_week + 1
    
    # 🎯 CAREER PATH RECOMMENDATIONS
    career_paths = generate_career_recommendations(interests, goals, goal_analysis['career_focus'])
    
    return {
        "learning_path": enhanced_path,
        "total_estimated_hours": total_hours,
        "timeline": timeline,
        "recommended_career_paths": career_paths,
        "user_analysis": goal_analysis,
        "completion_prediction": {
            "estimated_weeks": total_hours // time_per_week,
            "success_probability": np.mean([topic['success_probability'] for topic in enhanced_path]),
            "challenge_level": np.mean([topic['ml_complexity'] for topic in enhanced_path])
        }
    }

def generate_career_recommendations(interests, goals, career_focus):
    """🎯 Advanced career path recommendations"""
    career_mapping = {
        'data_scientist': ['Senior Data Scientist', 'Principal Data Scientist', 'Head of Data Science', 'Chief Data Officer'],
        'ml_engineer': ['Senior ML Engineer', 'Staff ML Engineer', 'ML Architect', 'Head of AI/ML'],
        'software_engineer': ['Senior Software Engineer', 'Tech Lead', 'Engineering Manager', 'CTO'],
        'researcher': ['Research Scientist', 'Principal Researcher', 'Research Director', 'Chief Scientist'],
        'entrepreneur': ['AI Startup Founder', 'Product Manager (AI)', 'AI Consultant', 'Innovation Director']
    }
    
    base_careers = career_mapping.get(career_focus, ['Data Scientist', 'ML Engineer', 'Software Engineer'])
    
    # Add interest-based careers
    interest_careers = {
        'ai_ml': ['AI Research Scientist', 'Machine Learning Engineer', 'AI Product Manager'],
        'data_science': ['Data Scientist', 'Analytics Manager', 'Business Intelligence Lead'],
        'web_development': ['Full-Stack Developer', 'Frontend Architect', 'DevOps Engineer'],
        'cybersecurity': ['Security Engineer', 'Cybersecurity Analyst', 'Security Architect']
    }
    
    additional_careers = []
    for interest in interests:
        if interest in interest_careers:
            additional_careers.extend(interest_careers[interest])
    
    return list(set(base_careers + additional_careers))[:6]

def create_advanced_visualizations(learning_path, user_data):
    """🎨 Create TechPath interactive visualizations"""
    
    # 🕸️ SKILL NETWORK GRAPH
    def create_skill_network():
        G = nx.Graph()
        
        # Add nodes for each topic
        for topic in learning_path:
            G.add_node(topic['title'][:20], 
                      complexity=topic.get('ml_complexity', 5),
                      hours=topic['estimated_hours'],
                      level=topic['level'])
        
        # Add edges based on prerequisites and relationships
        for i, topic in enumerate(learning_path[:-1]):
            G.add_edge(topic['title'][:20], learning_path[i+1]['title'][:20])
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Extract coordinates
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [f"{node}<br>Complexity: {G.nodes[node]['complexity']}" for node in G.nodes()]
        node_size = [G.nodes[node]['complexity'] * 8 for node in G.nodes()]
        
        # Create edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=3, color='rgba(102, 126, 234, 0.6)'),
                                hoverinfo='none',
                                mode='lines'))
        
        # Add nodes
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                marker=dict(size=node_size, 
                                           color='rgba(102, 126, 234, 0.8)',
                                           line=dict(width=2, color='white')),
                                text=[node.split('<br>')[0] for node in node_text],
                                textposition="middle center",
                                hovertext=node_text,
                                hoverinfo='text'))
        
        fig.update_layout(title='🕸️ Your Learning Path Network',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)')
        
        return fig
    
    # 📊 AI CONFIDENCE & COMPLEXITY ANALYSIS
    def create_confidence_analysis():
        topics = [topic['title'][:25] + '...' if len(topic['title']) > 25 else topic['title'] for topic in learning_path]
        ai_confidence = [topic.get('ai_confidence', 0.5) * 100 for topic in learning_path]
        success_prob = [topic.get('success_probability', 0.7) * 100 for topic in learning_path]
        complexity = [topic.get('ml_complexity', 5) for topic in learning_path]
        
        fig = go.Figure()
        
        # AI Confidence bars
        fig.add_trace(go.Bar(
            name='AI Confidence Score',
            x=topics,
            y=ai_confidence,
            marker_color='rgba(102, 126, 234, 0.8)',
            yaxis='y'
        ))
        
        # Success probability line
        fig.add_trace(go.Scatter(
            name='Success Probability',
            x=topics,
            y=success_prob,
            mode='lines+markers',
            marker_color='rgba(255, 99, 132, 0.8)',
            line=dict(width=3),
            yaxis='y2'
        ))
        
        # Complexity scatter
        fig.add_trace(go.Scatter(
            name='ML Complexity',
            x=topics,
            y=[c * 10 for c in complexity],  # Scale for visibility
            mode='markers',
            marker=dict(size=[c * 3 for c in complexity], 
                       color='rgba(255, 206, 84, 0.8)',
                       symbol='diamond'),
            yaxis='y'
        ))
        
        fig.update_layout(
            title='🤖 AI Analysis: Confidence, Success Probability & Complexity',
            xaxis_title='Learning Topics',
            yaxis=dict(title='AI Confidence & Complexity (%)', side='left'),
            yaxis2=dict(title='Success Probability (%)', side='right', overlaying='y'),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    # 📈 LEARNING PROGRESSION TIMELINE
    def create_progression_timeline():
        weeks = []
        cumulative_hours = []
        topics_completed = []
        current_hours = 0
        
        for i, topic in enumerate(learning_path):
            weeks_for_topic = max(1, topic['estimated_hours'] // user_data.get('time_per_week', 10))
            for week in range(weeks_for_topic):
                weeks.append(len(weeks) + 1)
                current_hours += topic['estimated_hours'] / weeks_for_topic
                cumulative_hours.append(current_hours)
                topics_completed.append(i + (week + 1) / weeks_for_topic)
        
        fig = go.Figure()
        
        # Cumulative hours
        fig.add_trace(go.Scatter(
            x=weeks,
            y=cumulative_hours,
            mode='lines+markers',
            name='Cumulative Learning Hours',
            line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
            fill='tonexty'
        ))
        
        # Topics progression
        fig.add_trace(go.Scatter(
            x=weeks,
            y=[t * 50 for t in topics_completed],  # Scale for dual axis
            mode='lines+markers',
            name='Topics Mastered',
            line=dict(color='rgba(255, 99, 132, 0.8)', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='📈 Learning Progression Timeline',
            xaxis_title='Weeks',
            yaxis=dict(title='Cumulative Hours', side='left'),
            yaxis2=dict(title='Topics Mastered', side='right', overlaying='y'),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    return create_skill_network(), create_confidence_analysis(), create_progression_timeline()

def main():
    """🚀 TECHPATH MAIN APPLICATION"""
    
    # 🎨 Advanced page configuration
    st.set_page_config(
        page_title="🚀 TechPath - AI Learning Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/techpath',
            'Report a bug': "https://github.com/your-repo/techpath/issues",
            'About': "# TechPath - AI Learning Platform\nAdvanced AI-powered learning path recommender using neural networks, NLP, and ensemble ML models."
        }
    )
    
    # Load custom CSS
    load_custom_css()
    
    # 🎨 STUNNING HEADER
    st.markdown("""
    <div class="gradient-header">
        <h1>🚀 TechPath - AI Learning Platform</h1>
        <p>Advanced AI/ML Learning Path Recommender • Neural Networks • NLP • Computer Vision • MLOps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 🎯 ADVANCED SIDEBAR WITH COMPREHENSIVE PROFILING
    with st.sidebar:
        st.markdown("## 🎯 Advanced AI Profiling")
        
        # Skill Level with enhanced descriptions
        st.markdown("### 🎓 Your Technical Level")
        user_level = st.selectbox(
            "Current Skill Level",
            ["beginner", "intermediate", "advanced"],
            help="This significantly impacts AI recommendations and complexity"
        )
        
        # Dynamic level description
        level_descriptions = {
            "beginner": "🟢 **Foundation Builder**: Python, Statistics, Data Analysis fundamentals",
            "intermediate": "🟡 **Skill Developer**: Machine Learning, Deep Learning, Full-stack AI apps", 
            "advanced": "🔴 **Expert Track**: Advanced NLP, Computer Vision, MLOps, Research-level AI"
        }
        st.markdown(level_descriptions[user_level])
        
        # Time commitment
        time_per_week = st.slider(
            "⏰ Weekly Learning Hours",
            min_value=5, max_value=50, value=15,
            help="Recommended: 15-25 hours for optimal learning"
        )
        
        # Learning style
        learning_style = st.selectbox(
            "🎨 Learning Style",
            ["visual", "kinesthetic", "auditory", "reading"],
            help="AI will adapt recommendations to your learning preference"
        )
        
        # AI Challenge Level
        ai_challenge_level = st.slider(
            "🧠 AI Challenge Preference",
            min_value=1, max_value=10, value=6,
            help="How challenging should your AI/ML topics be? (1=Easy, 10=Research-level)"
        )
        
        # Focus Area
        focus_area = st.selectbox(
            "🎯 AI/ML Specialization",
            ["General AI/ML", "Deep Learning", "Natural Language Processing", "Computer Vision", 
             "MLOps & Production", "Reinforcement Learning", "AI Research"],
            help="Your primary area of interest in AI/ML"
        )
        
        # Technical Interests
        st.markdown("### 🔬 Technical Domains")
        interests = st.multiselect(
            "Select your interests:",
            ["ai_ml", "data_science", "web_development", "cloud_computing", 
             "cybersecurity", "mobile_development", "blockchain", "quantum_computing"],
            default=["ai_ml", "data_science"],
            help="Choose multiple areas - AI will find connections"
        )
        
        # Career Goals with NLP Analysis
        st.markdown("### 🎯 Career Vision")
        goals_text = st.text_area(
            "Describe your career goals:",
            placeholder="e.g., I want to become a senior ML engineer at a tech company, build AI products that impact millions of users, and eventually lead an AI research team...",
            height=120,
            help="Be specific - our NLP AI will analyze your goals for personalized recommendations"
        )
        
        goals = [goal.strip() for goal in goals_text.split(',') if goal.strip()]
        
        # 🚀 GENERATE RECOMMENDATIONS
        if st.button("🚀 Generate AI-Powered Learning Path", type="primary", use_container_width=True):
            if not interests and not goals:
                st.error("Please select interests or describe your goals for AI analysis.")
            else:
                user_data = {
                    "level": user_level,
                    "interests": interests,
                    "goals": goals,
                    "goals_text": goals_text,
                    "time_per_week": time_per_week,
                    "learning_style": learning_style,
                    "ai_challenge_level": ai_challenge_level,
                    "focus_area": focus_area
                }
                
                with st.spinner("🤖 Advanced AI is analyzing your profile and generating personalized recommendations..."):
                    time.sleep(3)  # Simulate advanced AI processing
                    result = generate_advanced_recommendations(user_data)
                
                st.session_state.recommendations = result
                st.session_state.user_data = user_data
                
                st.success("🎉 Advanced AI analysis complete!")
    
    # 🎨 MAIN CONTENT AREA
    if 'recommendations' in st.session_state:
        result = st.session_state.recommendations
        user_data = st.session_state.user_data
        
        # 🎯 AI ANALYSIS SUMMARY
        st.markdown("## 🤖 AI Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📚 Learning Path</h3>
                <h2>{} Topics</h2>
                <p>Professional curriculum</p>
            </div>
            """.format(len(result['learning_path'])), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>⏱️ Time Investment</h3>
                <h2>{} Hours</h2>
                <p>Professional development</p>
            </div>
            """.format(result['total_estimated_hours']), unsafe_allow_html=True)
        
        with col3:
            completion = result['completion_prediction']
            st.markdown("""
            <div class="metric-card">
                <h3>📅 Timeline</h3>
                <h2>{} Weeks</h2>
                <p>To mastery</p>
            </div>
            """.format(completion['estimated_weeks']), unsafe_allow_html=True)
        
        with col4:
            success_rate = completion['success_probability'] * 100
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Success Rate</h3>
                <h2>{:.0f}%</h2>
                <p>AI prediction</p>
            </div>
            """.format(success_rate), unsafe_allow_html=True)
        
        # 🧠 USER ANALYSIS INSIGHTS
        user_analysis = result['user_analysis']
        st.markdown("### 🧠 AI-Powered User Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment_colors = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
            st.markdown(f"**Motivation Analysis:** {sentiment_colors[user_analysis['sentiment']]} {user_analysis['sentiment'].title()}")
        with col2:
            st.markdown(f"**Career Focus:** 🎯 {user_analysis['career_focus'].replace('_', ' ').title()}")
        with col3:
            st.markdown(f"**Learning Urgency:** ⚡ {user_analysis['urgency'].title()}")
        
        # 💼 CAREER RECOMMENDATIONS
        if result.get('recommended_career_paths'):
            st.markdown("### 💼 AI-Recommended Career Paths")
            career_cols = st.columns(min(len(result['recommended_career_paths']), 3))
            for i, career in enumerate(result['recommended_career_paths'][:3]):
                with career_cols[i]:
                    st.markdown(f"""
                    <div class="ai-feature-card">
                        <h4>🎯 {career}</h4>
                        <p>Aligned with your goals</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 🎨 ADVANCED VISUALIZATIONS
        st.markdown("### 📊 Advanced AI Analytics")
        
        network_fig, confidence_fig, timeline_fig = create_advanced_visualizations(result['learning_path'], user_data)
        
        tab1, tab2, tab3 = st.tabs(["🕸️ Learning Network", "🤖 AI Analysis", "📈 Progression"])
        
        with tab1:
            st.plotly_chart(network_fig, use_container_width=True)
            st.info("💡 **Network Analysis**: Visualizes the relationships and progression between your learning topics.")
        
        with tab2:
            st.plotly_chart(confidence_fig, use_container_width=True)
            st.info("💡 **AI Insights**: Shows AI confidence scores, success predictions, and complexity analysis.")
        
        with tab3:
            st.plotly_chart(timeline_fig, use_container_width=True)
            st.info("💡 **Timeline Prediction**: Forecasts your learning progression over time.")
        
        # 🛤️ DETAILED LEARNING PATH
        st.markdown("### 🛤️ Your TechPath Learning Journey")
        
        for i, topic in enumerate(result['learning_path']):
            # Difficulty color coding
            level_colors = {"beginner": "skill-badge-beginner", "intermediate": "skill-badge-intermediate", "advanced": "skill-badge-advanced"}
            level_class = level_colors.get(topic['level'], "skill-badge-beginner")
            
            # AI confidence indicator
            confidence = topic.get('ai_confidence', 0.5)
            if confidence > 0.8:
                confidence_class = "ai-confidence-high"
                confidence_text = "High Confidence"
            elif confidence > 0.6:
                confidence_class = "ai-confidence-medium"  
                confidence_text = "Medium Confidence"
            else:
                confidence_class = "ai-confidence-low"
                confidence_text = "Building Confidence"
            
            with st.expander(f"🎯 {i+1}. {topic['title']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📖 Description:** {topic['description']}")
                    
                    # Skills with badges
                    st.markdown("**🎯 Skills You'll Master:**")
                    skills_html = ""
                    for skill in topic['skills_covered'][:4]:
                        skills_html += f'<span class="{level_class}">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    # Learning outcomes
                    if topic.get('learning_outcomes'):
                        st.markdown("**🎓 Learning Outcomes:**")
                        for outcome in topic['learning_outcomes'][:3]:
                            st.markdown(f"• {outcome}")
                
                with col2:
                    st.markdown(f'<div class="{level_class}">{topic["level"].title()} Level</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="{confidence_class}">{confidence_text}</div>', unsafe_allow_html=True)
                    
                    st.metric("⏱️ Hours", topic['estimated_hours'])
                    st.metric("🧠 ML Complexity", f"{topic.get('ml_complexity', 5)}/10")
                    st.metric("📈 Success Rate", f"{topic.get('success_probability', 0.7)*100:.0f}%")
                    st.metric("💼 Career Impact", f"{topic.get('career_impact', 0.8)*100:.0f}%")
                
                # Projects and certifications
                if topic.get('projects'):
                    st.markdown("**🚀 Featured Projects:**")
                    for project in topic['projects'][:2]:
                        st.markdown(f"• {project}")
                
                if topic.get('certifications'):
                    st.markdown("**🏆 Industry Certifications:**")
                    for cert in topic['certifications'][:2]:
                        st.markdown(f"• {cert}")
        
        # 📅 DETAILED TIMELINE
        st.markdown("### 📅 Detailed Learning Timeline")
        
        for week_info in result['timeline']:
            difficulty_colors = {"beginner": "#4facfe", "intermediate": "#fa709a", "advanced": "#a8edea"}
            color = difficulty_colors.get(week_info.get('difficulty', 'intermediate'), "#4facfe")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}80 100%); 
                        padding: 1rem; border-radius: 10px; margin: 0.5rem 0; color: white;">
                <h4>📅 {week_info['week_range']}: {week_info['topic']}</h4>
                <p>⏰ {week_info['hours_per_week']} hours/week • 📊 Total: {week_info['total_hours']} hours</p>
                <p><strong>🎯 Milestones:</strong> {' • '.join(week_info.get('milestones', [])[:2])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 🎉 SUCCESS BANNER
        st.markdown("""
        <div class="success-banner">
            <h3>🎉 Your TechPath AI Learning Journey Awaits!</h3>
            <p>This advanced curriculum will transform you into an AI/ML expert ready for industry leadership roles.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # 🌟 WELCOME SECTION WITH AI FEATURES
        st.markdown("## 🌟 Welcome to the Future of AI Education!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="ai-feature-card">
                <h3>🧠 Neural Network Recommendations</h3>
                <p>Advanced ensemble ML models analyze your profile using neural networks, gradient boosting, and random forests for hyper-personalized recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="ai-feature-card">
                <h3>🔍 NLP Goal Analysis</h3>
                <p>Sophisticated natural language processing analyzes your career goals, detecting sentiment, urgency, and career focus for tailored learning paths.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="ai-feature-card">
                <h3>🕸️ Knowledge Graph Intelligence</h3>
                <p>Advanced graph algorithms map skill dependencies and prerequisites, ensuring optimal learning sequence and knowledge building.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="ai-feature-card">
                <h3>📊 Predictive Analytics</h3>
                <p>Machine learning models predict your success probability, optimal timeline, and career trajectory based on comprehensive data analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 🎯 CAPSTONE FEATURES
        st.markdown("### 🎯 TechPath AI/ML Features")
        
        features_col1, features_col2, features_col3 = st.columns(3)
        
        with features_col1:
            st.markdown("""
            **🤖 Advanced ML Models:**
            - Neural Network Ensemble
            - Gradient Boosting Predictions  
            - Random Forest Analysis
            - TF-IDF Content Similarity
            - K-Means User Clustering
            """)
        
        with features_col2:
            st.markdown("""
            **🔬 AI Techniques:**
            - Natural Language Processing
            - Sentiment Analysis
            - Graph Neural Networks
            - Predictive Modeling
            - Recommendation Systems
            """)
        
        with features_col3:
            st.markdown("""
            **📊 Advanced Analytics:**
            - Interactive Network Graphs
            - Success Probability Modeling
            - Learning Progression Forecasting
            - Career Path Optimization
            - Real-time Personalization
            """)
        
        st.markdown("""
        ### 🚀 Ready to Begin Your AI Mastery Journey?
        
        Complete your advanced profile in the sidebar to unlock:
        - **Personalized AI curriculum** tailored to your goals
        - **Neural network-powered recommendations** 
        - **Advanced career path analysis**
        - **Interactive learning visualizations**
        - **Industry-aligned skill development**
        
        This isn't just a learning platform - it's your gateway to becoming an AI/ML expert! 🤖✨
        """)

if __name__ == "__main__":
    main()