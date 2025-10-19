from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json
import os
from datetime import datetime
import logging

# Import our advanced models
from models.advanced_recommender import AdvancedRecommender
from models.user_model import UserModel, LearningEvent
from models.knowledge_graph import KnowledgeGraph
from ai_chatbot.chatbot import AIChatbot
from database.database import setup_database, get_database_manager
from database.models import User, Topic, LearningSession, TopicProgress

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LearnAI - Advanced AI-Powered Learning Platform Backend
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'learnai-capstone-project-2024')
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost/learning_recommender')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)
db_manager = setup_database()

# Initialize advanced components
advanced_recommender = AdvancedRecommender()
ai_chatbot = AIChatbot(openai_api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User ID is required'
            }), 400
        
        # Get or create user profile
        user_profile = get_or_create_user_profile(user_id, data)
        
        # Get advanced recommendations
        recommendations = advanced_recommender.get_recommendations(
            user_id=user_id,
            n_recommendations=10,
            context=data
        )
        
        # Get user insights
        user_insights = advanced_recommender.user_model.get_user_insights(user_id)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'user_insights': user_insights,
            'learning_path': generate_learning_path(recommendations),
            'career_suggestions': get_career_suggestions(user_profile)
        })
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/topics', methods=['GET'])
def get_available_topics():
    """Get list of available learning topics"""
    try:
        topics = list(advanced_recommender.knowledge_graph.topics.values())
        topics_data = []
        
        for topic in topics:
            topics_data.append({
                'id': topic.topic_id,
                'title': topic.title,
                'description': topic.description,
                'category': topic.category,
                'difficulty_level': topic.difficulty_level,
                'estimated_hours': topic.estimated_hours,
                'skills_covered': topic.skills_covered,
                'prerequisites': topic.prerequisites,
                'popularity_score': topic.popularity_score,
                'industry_relevance': topic.industry_relevance
            })
        
        return jsonify({
            'success': True,
            'topics': topics_data
        })
    except Exception as e:
        logger.error(f"Error getting topics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_bot():
    """Chat with AI learning assistant"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        message = data.get('message')
        
        if not user_id or not message:
            return jsonify({
                'success': False,
                'error': 'User ID and message are required'
            }), 400
        
        # Process message with AI chatbot
        import asyncio
        response = asyncio.run(ai_chatbot.process_message(user_id, message))
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user/<user_id>/progress', methods=['GET'])
def get_user_progress(user_id):
    """Get detailed user progress and analytics"""
    try:
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'User not found'
                }), 404
            
            # Get progress data
            progress_data = get_user_progress_data(user_id)
            
            return jsonify({
                'success': True,
                'progress': progress_data
            })
    
    except Exception as e:
        logger.error(f"Error getting user progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user/<user_id>/feedback', methods=['POST'])
def submit_feedback(user_id):
    """Submit user feedback for topics"""
    try:
        data = request.get_json()
        topic_id = data.get('topic_id')
        rating = data.get('rating')
        feedback_text = data.get('feedback', '')
        
        if not topic_id or not rating:
            return jsonify({
                'success': False,
                'error': 'Topic ID and rating are required'
            }), 400
        
        # Update feedback in advanced recommender
        advanced_recommender.update_feedback(user_id, topic_id, rating, feedback_text)
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully'
        })
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_system_analytics():
    """Get system-wide analytics and insights"""
    try:
        analytics_data = {
            'total_users': get_total_users(),
            'active_users': get_active_users(),
            'popular_topics': get_popular_topics(),
            'completion_rates': get_completion_rates(),
            'model_performance': advanced_recommender.get_model_performance()
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics_data
        })
    
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = db_manager.test_connection()
        
        # Check model status
        model_status = {
            'user_model': advanced_recommender.user_model is not None,
            'knowledge_graph': advanced_recommender.knowledge_graph is not None,
            'collaborative_filter': advanced_recommender.collaborative_filter is not None,
            'neural_recommender': advanced_recommender.neural_recommender is not None
        }
        
        return jsonify({
            'success': True,
            'message': 'Advanced Learning Path Recommender API is running',
            'database': 'connected' if db_status else 'disconnected',
            'models': model_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Helper functions
def get_or_create_user_profile(user_id, data):
    """Get or create user profile"""
    with db_manager.get_session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        
        if not user:
            # Create new user
            user = User(
                id=user_id,
                email=data.get('email', f'{user_id}@example.com'),
                username=data.get('username', user_id),
                learning_goals=data.get('goals', []),
                interests=data.get('interests', []),
                preferred_learning_style=data.get('learning_style', 'visual'),
                difficulty_preference=data.get('difficulty_preference', 'medium'),
                available_time_per_week=data.get('time_per_week', 10)
            )
            session.add(user)
            session.commit()
            
            # Create user profile in advanced recommender
            advanced_recommender.user_model.create_user_profile(user_id, data)
        
        return user

def generate_learning_path(recommendations):
    """Generate structured learning path from recommendations"""
    learning_path = []
    total_hours = 0
    
    for i, rec in enumerate(recommendations):
        learning_path.append({
            'step': i + 1,
            'topic_id': rec['topic_id'],
            'title': rec['title'],
            'description': rec['description'],
            'estimated_hours': rec['estimated_hours'],
            'difficulty_level': rec['difficulty_level'],
            'score': rec['score'],
            'prerequisites': rec.get('prerequisites', [])
        })
        total_hours += rec['estimated_hours']
    
    return {
        'path': learning_path,
        'total_hours': total_hours,
        'estimated_weeks': total_hours // 10  # Assuming 10 hours per week
    }

def get_career_suggestions(user_profile):
    """Get career suggestions based on user profile"""
    career_mapping = {
        'data_science': ['Data Scientist', 'Data Analyst', 'ML Engineer'],
        'web_development': ['Frontend Developer', 'Backend Developer', 'Full Stack Developer'],
        'ai_ml': ['AI Engineer', 'ML Engineer', 'Data Scientist'],
        'programming': ['Software Engineer', 'Developer', 'Programmer']
    }
    
    suggestions = []
    for interest in user_profile.interests:
        if interest in career_mapping:
            suggestions.extend(career_mapping[interest])
    
    return list(set(suggestions))  # Remove duplicates

def get_user_progress_data(user_id):
    """Get comprehensive user progress data"""
    with db_manager.get_session() as session:
        # Get topic progress
        topic_progress = session.query(TopicProgress).filter(
            TopicProgress.user_id == user_id
        ).all()
        
        # Get learning sessions
        learning_sessions = session.query(LearningSession).filter(
            LearningSession.user_id == user_id
        ).order_by(LearningSession.start_time.desc()).limit(20).all()
        
        progress_data = {
            'completed_topics': len([tp for tp in topic_progress if tp.status == 'completed']),
            'in_progress_topics': len([tp for tp in topic_progress if tp.status == 'in_progress']),
            'total_hours': sum(tp.time_spent_hours for tp in topic_progress),
            'recent_sessions': [
                {
                    'topic_id': ls.topic_id,
                    'start_time': ls.start_time.isoformat(),
                    'duration_minutes': ls.duration_minutes,
                    'completion_percentage': ls.completion_percentage,
                    'engagement_score': ls.engagement_score
                }
                for ls in learning_sessions
            ]
        }
        
        return progress_data

def get_total_users():
    """Get total number of users"""
    with db_manager.get_session() as session:
        return session.query(User).count()

def get_active_users():
    """Get number of active users (last 30 days)"""
    with db_manager.get_session() as session:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=30)
        return session.query(User).filter(User.last_login >= cutoff_date).count()

def get_popular_topics():
    """Get most popular topics"""
    with db_manager.get_session() as session:
        topics = session.query(Topic).order_by(Topic.total_enrollments.desc()).limit(10).all()
        return [
            {
                'topic_id': topic.id,
                'title': topic.title,
                'enrollments': topic.total_enrollments,
                'popularity_score': topic.popularity_score
            }
            for topic in topics
        ]

def get_completion_rates():
    """Get topic completion rates"""
    with db_manager.get_session() as session:
        topics = session.query(Topic).all()
        completion_rates = []
        
        for topic in topics:
            total_enrollments = topic.total_enrollments
            completed = session.query(TopicProgress).filter(
                TopicProgress.topic_id == topic.id,
                TopicProgress.status == 'completed'
            ).count()
            
            completion_rate = completed / max(total_enrollments, 1)
            completion_rates.append({
                'topic_id': topic.id,
                'title': topic.title,
                'completion_rate': completion_rate
            })
        
        return sorted(completion_rates, key=lambda x: x['completion_rate'], reverse=True)

if __name__ == '__main__':
    # Seed database on startup
    try:
        from database.seed_data import seed_database
        seed_database()
        logger.info("Database seeded successfully")
    except Exception as e:
        logger.warning(f"Database seeding failed: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
