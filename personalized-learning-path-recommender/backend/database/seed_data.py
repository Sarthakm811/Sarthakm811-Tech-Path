"""
Seed Data for Advanced Learning Path Recommender
Populates the database with comprehensive learning topics, skills, and sample data
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

from .models import *
from .database import get_database_manager

def create_skills() -> List[Skill]:
    """Create comprehensive skill definitions"""
    skills_data = [
        # Programming Fundamentals
        {
            'id': 'python_basics',
            'name': 'Python Programming Basics',
            'description': 'Fundamental Python programming concepts including variables, data types, control structures, and functions',
            'category': 'programming',
            'difficulty_level': 1,
            'estimated_hours': 20,
            'prerequisites': [],
            'learning_objectives': [
                'Write basic Python programs',
                'Use variables and data types effectively',
                'Implement control structures',
                'Create and use functions'
            ],
            'assessment_criteria': [
                'Complete coding exercises',
                'Build a simple calculator',
                'Pass syntax and logic tests'
            ]
        },
        {
            'id': 'python_advanced',
            'name': 'Advanced Python Programming',
            'description': 'Advanced Python concepts including OOP, decorators, generators, and design patterns',
            'category': 'programming',
            'difficulty_level': 3,
            'estimated_hours': 30,
            'prerequisites': ['python_basics'],
            'learning_objectives': [
                'Implement object-oriented programming',
                'Use decorators and generators',
                'Apply design patterns',
                'Handle exceptions effectively'
            ],
            'assessment_criteria': [
                'Build OOP-based applications',
                'Implement advanced Python features',
                'Pass comprehensive coding challenges'
            ]
        },
        {
            'id': 'data_structures',
            'name': 'Data Structures and Algorithms',
            'description': 'Essential data structures and algorithm design patterns for efficient programming',
            'category': 'computer_science',
            'difficulty_level': 3,
            'estimated_hours': 40,
            'prerequisites': ['python_basics'],
            'learning_objectives': [
                'Implement common data structures',
                'Analyze algorithm complexity',
                'Solve algorithmic problems',
                'Apply optimization techniques'
            ],
            'assessment_criteria': [
                'Implement data structures from scratch',
                'Solve coding interview problems',
                'Analyze time and space complexity'
            ]
        },
        # Data Science
        {
            'id': 'data_analysis',
            'name': 'Data Analysis with Python',
            'description': 'Data manipulation, analysis, and visualization using pandas, NumPy, and Matplotlib',
            'category': 'data_science',
            'difficulty_level': 2,
            'estimated_hours': 25,
            'prerequisites': ['python_basics'],
            'learning_objectives': [
                'Manipulate data with pandas',
                'Perform statistical analysis',
                'Create data visualizations',
                'Clean and preprocess data'
            ],
            'assessment_criteria': [
                'Complete data analysis projects',
                'Create meaningful visualizations',
                'Present data insights effectively'
            ]
        },
        {
            'id': 'machine_learning',
            'name': 'Machine Learning Fundamentals',
            'description': 'Introduction to machine learning concepts, algorithms, and implementation',
            'category': 'ai_ml',
            'difficulty_level': 3,
            'estimated_hours': 35,
            'prerequisites': ['python_advanced', 'data_analysis'],
            'learning_objectives': [
                'Understand ML concepts and terminology',
                'Implement supervised learning algorithms',
                'Apply unsupervised learning techniques',
                'Evaluate model performance'
            ],
            'assessment_criteria': [
                'Build ML models from scratch',
                'Compare different algorithms',
                'Optimize model performance'
            ]
        },
        {
            'id': 'deep_learning',
            'name': 'Deep Learning and Neural Networks',
            'description': 'Advanced machine learning with neural networks, CNNs, RNNs, and modern architectures',
            'category': 'ai_ml',
            'difficulty_level': 4,
            'estimated_hours': 50,
            'prerequisites': ['machine_learning'],
            'learning_objectives': [
                'Build neural networks',
                'Implement CNNs and RNNs',
                'Use deep learning frameworks',
                'Apply transfer learning'
            ],
            'assessment_criteria': [
                'Build deep learning models',
                'Train networks effectively',
                'Apply to real-world problems'
            ]
        },
        # Web Development
        {
            'id': 'html_css',
            'name': 'HTML and CSS Fundamentals',
            'description': 'Web markup and styling fundamentals for creating responsive web pages',
            'category': 'web_development',
            'difficulty_level': 1,
            'estimated_hours': 15,
            'prerequisites': [],
            'learning_objectives': [
                'Write semantic HTML',
                'Style pages with CSS',
                'Create responsive layouts',
                'Use CSS frameworks'
            ],
            'assessment_criteria': [
                'Build responsive web pages',
                'Implement CSS animations',
                'Pass accessibility tests'
            ]
        },
        {
            'id': 'javascript',
            'name': 'JavaScript Programming',
            'description': 'Client-side programming with JavaScript, DOM manipulation, and modern ES6+ features',
            'category': 'web_development',
            'difficulty_level': 2,
            'estimated_hours': 25,
            'prerequisites': ['html_css'],
            'learning_objectives': [
                'Write JavaScript programs',
                'Manipulate the DOM',
                'Handle events and async operations',
                'Use modern JavaScript features'
            ],
            'assessment_criteria': [
                'Build interactive web applications',
                'Implement complex functionality',
                'Debug JavaScript code effectively'
            ]
        },
        {
            'id': 'react',
            'name': 'React.js Development',
            'description': 'Modern frontend development with React.js, hooks, and component architecture',
            'category': 'web_development',
            'difficulty_level': 3,
            'estimated_hours': 30,
            'prerequisites': ['javascript'],
            'learning_objectives': [
                'Build React components',
                'Use React hooks',
                'Manage application state',
                'Implement routing and navigation'
            ],
            'assessment_criteria': [
                'Build complete React applications',
                'Implement state management',
                'Optimize component performance'
            ]
        },
        # Database
        {
            'id': 'sql',
            'name': 'SQL and Database Management',
            'description': 'Database design, SQL queries, and database management systems',
            'category': 'database',
            'difficulty_level': 2,
            'estimated_hours': 20,
            'prerequisites': [],
            'learning_objectives': [
                'Write complex SQL queries',
                'Design database schemas',
                'Optimize database performance',
                'Manage database security'
            ],
            'assessment_criteria': [
                'Design normalized databases',
                'Write efficient queries',
                'Optimize database performance'
            ]
        },
        # Cloud Computing
        {
            'id': 'aws_cloud',
            'name': 'AWS Cloud Computing',
            'description': 'Cloud computing fundamentals and AWS services for scalable applications',
            'category': 'cloud_computing',
            'difficulty_level': 3,
            'estimated_hours': 30,
            'prerequisites': ['python_basics'],
            'learning_objectives': [
                'Deploy applications to AWS',
                'Use AWS services effectively',
                'Implement cloud security',
                'Optimize cloud costs'
            ],
            'assessment_criteria': [
                'Deploy applications to cloud',
                'Implement cloud architectures',
                'Pass AWS certification exams'
            ]
        },
        # Cybersecurity
        {
            'id': 'cybersecurity_basics',
            'name': 'Cybersecurity Fundamentals',
            'description': 'Information security principles, threat analysis, and security best practices',
            'category': 'security',
            'difficulty_level': 2,
            'estimated_hours': 25,
            'prerequisites': [],
            'learning_objectives': [
                'Identify security threats',
                'Implement security measures',
                'Conduct security assessments',
                'Follow security best practices'
            ],
            'assessment_criteria': [
                'Complete security assessments',
                'Implement security solutions',
                'Pass security certifications'
            ]
        }
    ]
    
    skills = []
    for skill_data in skills_data:
        skill = Skill(**skill_data)
        skills.append(skill)
    
    return skills

def create_topics() -> List[Topic]:
    """Create comprehensive learning topics"""
    topics_data = [
        # Programming Track
        {
            'id': 'python_fundamentals',
            'title': 'Python Programming Fundamentals',
            'description': 'Master the basics of Python programming with hands-on projects and real-world applications',
            'category': 'programming',
            'subcategory': 'python',
            'difficulty_level': 1,
            'estimated_hours': 25,
            'skills_covered': ['python_basics'],
            'prerequisites': [],
            'learning_outcomes': [
                'Write clean, readable Python code',
                'Understand Python data types and structures',
                'Implement control flow and functions',
                'Handle errors and exceptions'
            ],
            'assessment_methods': [
                'Coding exercises',
                'Project-based assessments',
                'Code review sessions'
            ],
            'resources': [
                {'type': 'video', 'title': 'Python Basics Tutorial', 'url': 'https://example.com/python-basics'},
                {'type': 'documentation', 'title': 'Python Official Docs', 'url': 'https://docs.python.org/'},
                {'type': 'book', 'title': 'Python Crash Course', 'author': 'Eric Matthes'}
            ],
            'tags': ['python', 'programming', 'beginner', 'fundamentals'],
            'popularity_score': 0.9,
            'industry_relevance': 0.95
        },
        {
            'id': 'python_oop',
            'title': 'Object-Oriented Programming in Python',
            'description': 'Learn advanced Python concepts including classes, inheritance, and design patterns',
            'category': 'programming',
            'subcategory': 'python',
            'difficulty_level': 3,
            'estimated_hours': 30,
            'skills_covered': ['python_advanced'],
            'prerequisites': ['python_fundamentals'],
            'learning_outcomes': [
                'Design and implement classes',
                'Apply inheritance and polymorphism',
                'Use advanced Python features',
                'Implement design patterns'
            ],
            'assessment_methods': [
                'OOP design projects',
                'Code architecture reviews',
                'Pattern implementation exercises'
            ],
            'resources': [
                {'type': 'video', 'title': 'Python OOP Masterclass', 'url': 'https://example.com/python-oop'},
                {'type': 'documentation', 'title': 'Python OOP Guide', 'url': 'https://docs.python.org/3/tutorial/classes.html'}
            ],
            'tags': ['python', 'oop', 'advanced', 'design-patterns'],
            'popularity_score': 0.8,
            'industry_relevance': 0.9
        },
        # Data Science Track
        {
            'id': 'data_science_intro',
            'title': 'Introduction to Data Science',
            'description': 'Comprehensive introduction to data science with Python, covering analysis, visualization, and statistics',
            'category': 'data_science',
            'subcategory': 'analytics',
            'difficulty_level': 2,
            'estimated_hours': 35,
            'skills_covered': ['data_analysis', 'python_advanced'],
            'prerequisites': ['python_fundamentals'],
            'learning_outcomes': [
                'Analyze data with pandas and NumPy',
                'Create meaningful visualizations',
                'Apply statistical concepts',
                'Present data insights'
            ],
            'assessment_methods': [
                'Data analysis projects',
                'Visualization portfolios',
                'Statistical analysis reports'
            ],
            'resources': [
                {'type': 'video', 'title': 'Data Science with Python', 'url': 'https://example.com/data-science'},
                {'type': 'dataset', 'title': 'Sample Datasets', 'url': 'https://example.com/datasets'}
            ],
            'tags': ['data-science', 'pandas', 'numpy', 'visualization'],
            'popularity_score': 0.85,
            'industry_relevance': 0.9
        },
        {
            'id': 'machine_learning_basics',
            'title': 'Machine Learning Fundamentals',
            'description': 'Learn machine learning algorithms and implementation with scikit-learn and real-world projects',
            'category': 'ai_ml',
            'subcategory': 'machine_learning',
            'difficulty_level': 3,
            'estimated_hours': 40,
            'skills_covered': ['machine_learning'],
            'prerequisites': ['data_science_intro'],
            'learning_outcomes': [
                'Implement supervised learning algorithms',
                'Apply unsupervised learning techniques',
                'Evaluate model performance',
                'Handle real-world ML problems'
            ],
            'assessment_methods': [
                'ML model implementations',
                'Performance evaluation reports',
                'Kaggle-style competitions'
            ],
            'resources': [
                {'type': 'video', 'title': 'ML with Scikit-learn', 'url': 'https://example.com/ml-scikit'},
                {'type': 'book', 'title': 'Hands-On Machine Learning', 'author': 'Aurélien Géron'}
            ],
            'tags': ['machine-learning', 'scikit-learn', 'algorithms', 'supervised-learning'],
            'popularity_score': 0.9,
            'industry_relevance': 0.95
        },
        # Web Development Track
        {
            'id': 'web_development_basics',
            'title': 'Web Development Fundamentals',
            'description': 'Build modern web applications with HTML, CSS, and JavaScript',
            'category': 'web_development',
            'subcategory': 'frontend',
            'difficulty_level': 1,
            'estimated_hours': 30,
            'skills_covered': ['html_css', 'javascript'],
            'prerequisites': [],
            'learning_outcomes': [
                'Create responsive web pages',
                'Implement interactive features',
                'Use modern web standards',
                'Optimize web performance'
            ],
            'assessment_methods': [
                'Website development projects',
                'Code quality reviews',
                'Performance optimization tasks'
            ],
            'resources': [
                {'type': 'video', 'title': 'Web Development Bootcamp', 'url': 'https://example.com/web-dev'},
                {'type': 'documentation', 'title': 'MDN Web Docs', 'url': 'https://developer.mozilla.org/'}
            ],
            'tags': ['web-development', 'html', 'css', 'javascript', 'frontend'],
            'popularity_score': 0.9,
            'industry_relevance': 0.9
        },
        {
            'id': 'react_development',
            'title': 'React.js Development',
            'description': 'Master modern frontend development with React.js, hooks, and state management',
            'category': 'web_development',
            'subcategory': 'frontend',
            'difficulty_level': 3,
            'estimated_hours': 35,
            'skills_covered': ['react'],
            'prerequisites': ['web_development_basics'],
            'learning_outcomes': [
                'Build React applications',
                'Manage component state',
                'Implement routing',
                'Optimize performance'
            ],
            'assessment_methods': [
                'React application projects',
                'Component architecture reviews',
                'Performance optimization challenges'
            ],
            'resources': [
                {'type': 'video', 'title': 'React Complete Guide', 'url': 'https://example.com/react-guide'},
                {'type': 'documentation', 'title': 'React Official Docs', 'url': 'https://reactjs.org/docs/'}
            ],
            'tags': ['react', 'javascript', 'frontend', 'components', 'state-management'],
            'popularity_score': 0.85,
            'industry_relevance': 0.9
        }
    ]
    
    topics = []
    for topic_data in topics_data:
        topic = Topic(**topic_data)
        topics.append(topic)
    
    return topics

def create_sample_users() -> List[User]:
    """Create sample users for testing"""
    users_data = [
        {
            'email': 'john.doe@example.com',
            'username': 'john_doe',
            'full_name': 'John Doe',
            'preferred_learning_style': 'visual',
            'optimal_session_duration': 45,
            'preferred_time_of_day': 'morning',
            'difficulty_preference': 'medium',
            'available_time_per_week': 15,
            'preferred_pace': 'medium',
            'learning_goals': ['Become a data scientist', 'Learn machine learning'],
            'interests': ['data_science', 'programming', 'ai_ml'],
            'target_career_paths': ['data_scientist', 'ai_engineer']
        },
        {
            'email': 'jane.smith@example.com',
            'username': 'jane_smith',
            'full_name': 'Jane Smith',
            'preferred_learning_style': 'auditory',
            'optimal_session_duration': 60,
            'preferred_time_of_day': 'evening',
            'difficulty_preference': 'hard',
            'available_time_per_week': 20,
            'preferred_pace': 'fast',
            'learning_goals': ['Become a full-stack developer', 'Master React'],
            'interests': ['web_development', 'programming'],
            'target_career_paths': ['web_developer', 'full_stack_developer']
        },
        {
            'email': 'bob.wilson@example.com',
            'username': 'bob_wilson',
            'full_name': 'Bob Wilson',
            'preferred_learning_style': 'kinesthetic',
            'optimal_session_duration': 30,
            'preferred_time_of_day': 'afternoon',
            'difficulty_preference': 'easy',
            'available_time_per_week': 8,
            'preferred_pace': 'slow',
            'learning_goals': ['Learn programming basics', 'Build simple websites'],
            'interests': ['programming', 'web_development'],
            'target_career_paths': ['web_developer']
        }
    ]
    
    users = []
    for user_data in users_data:
        user = User(**user_data)
        users.append(user)
    
    return users

def seed_database():
    """Seed the database with initial data"""
    try:
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Create skills
            print("Creating skills...")
            skills = create_skills()
            for skill in skills:
                session.add(skill)
            
            # Create topics
            print("Creating topics...")
            topics = create_topics()
            for topic in topics:
                session.add(topic)
            
            # Create sample users
            print("Creating sample users...")
            users = create_sample_users()
            for user in users:
                session.add(user)
            
            session.commit()
            print("✅ Database seeded successfully!")
            
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        raise

def clear_database():
    """Clear all data from the database (use with caution!)"""
    try:
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Delete all data (in correct order to respect foreign keys)
            session.query(LearningSession).delete()
            session.query(TopicProgress).delete()
            session.query(Feedback).delete()
            session.query(Recommendation).delete()
            session.query(LearningAnalytics).delete()
            session.query(SystemMetrics).delete()
            session.query(StudyGroupMember).delete()
            session.query(StudyGroup).delete()
            session.query(SocialConnection).delete()
            session.query(User).delete()
            session.query(Topic).delete()
            session.query(Skill).delete()
            
            session.commit()
            print("✅ Database cleared successfully!")
            
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Clear and seed database
        clear_database()
        seed_database()
        
        print("🎉 Database setup completed!")
        
    except Exception as e:
        print(f"💥 Database setup failed: {e}")
