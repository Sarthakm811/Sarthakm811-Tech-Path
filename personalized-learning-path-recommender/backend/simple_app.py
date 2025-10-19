
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    import json
    from datetime import datetime
    
    app = Flask(__name__)
    CORS(app)
    
    # Sample data for demonstration
    SAMPLE_TOPICS = [
        {
            "id": "python_basics",
            "title": "Python Programming Fundamentals",
            "description": "Learn the basics of Python programming including variables, functions, and control structures",
            "category": "programming",
            "difficulty_level": "beginner",
            "estimated_hours": 20,
            "skills_covered": ["Variables", "Functions", "Loops", "Data Types"],
            "prerequisites": [],
            "popularity_score": 0.9,
            "industry_relevance": 0.95
        },
        {
            "id": "data_analysis",
            "title": "Data Analysis with Pandas",
            "description": "Master data manipulation and analysis using Python's Pandas library",
            "category": "data_science",
            "difficulty_level": "intermediate",
            "estimated_hours": 25,
            "skills_covered": ["Pandas", "Data Cleaning", "Data Visualization"],
            "prerequisites": ["python_basics"],
            "popularity_score": 0.85,
            "industry_relevance": 0.9
        },
        {
            "id": "machine_learning",
            "title": "Introduction to Machine Learning",
            "description": "Learn fundamental machine learning concepts and algorithms",
            "category": "ai_ml",
            "difficulty_level": "intermediate",
            "estimated_hours": 40,
            "skills_covered": ["Supervised Learning", "Unsupervised Learning", "Model Evaluation"],
            "prerequisites": ["python_basics", "data_analysis"],
            "popularity_score": 0.95,
            "industry_relevance": 0.98
        },
        {
            "id": "web_development",
            "title": "Web Development with Flask",
            "description": "Build web applications using Python Flask framework",
            "category": "web_development",
            "difficulty_level": "intermediate",
            "estimated_hours": 30,
            "skills_covered": ["Flask", "HTML", "CSS", "REST APIs"],
            "prerequisites": ["python_basics"],
            "popularity_score": 0.8,
            "industry_relevance": 0.85
        },
        {
            "id": "database_design",
            "title": "Database Design and SQL",
            "description": "Learn database design principles and SQL query optimization",
            "category": "database",
            "difficulty_level": "intermediate",
            "estimated_hours": 35,
            "skills_covered": ["SQL", "Database Design", "Query Optimization"],
            "prerequisites": [],
            "popularity_score": 0.75,
            "industry_relevance": 0.9
        }
    ]
    
    def generate_recommendations(user_data):
        """Generate personalized recommendations based on user data"""
        level = user_data.get('level', 'beginner')
        interests = user_data.get('interests', [])
        time_per_week = user_data.get('time_per_week', 10)
        
        # Filter topics based on user preferences
        suitable_topics = []
        for topic in SAMPLE_TOPICS:
            # Check difficulty level
            if level == 'beginner' and topic['difficulty_level'] in ['beginner', 'intermediate']:
                suitable_topics.append(topic)
            elif level == 'intermediate' and topic['difficulty_level'] in ['beginner', 'intermediate', 'advanced']:
                suitable_topics.append(topic)
            elif level == 'advanced':
                suitable_topics.append(topic)
        
        # Filter by interests if provided
        if interests:
            filtered_topics = []
            for topic in suitable_topics:
                for interest in interests:
                    if interest in topic['category'] or any(interest in skill.lower() for skill in topic['skills_covered']):
                        filtered_topics.append(topic)
                        break
            suitable_topics = filtered_topics if filtered_topics else suitable_topics
        
        # Sort by relevance and popularity
        suitable_topics.sort(key=lambda x: (x['industry_relevance'] + x['popularity_score']) / 2, reverse=True)
        
        # Create learning path
        learning_path = []
        total_hours = 0
        
        for topic in suitable_topics[:5]:  # Limit to top 5 recommendations
            learning_path.append({
                "title": topic["title"],
                "description": topic["description"],
                "category": topic["category"],
                "level": topic["difficulty_level"],
                "estimated_hours": topic["estimated_hours"],
                "skills_covered": topic["skills_covered"],
                "prerequisites": topic["prerequisites"]
            })
            total_hours += topic["estimated_hours"]
        
        # Generate timeline
        timeline = []
        current_week = 1
        for topic in learning_path:
            weeks_needed = max(1, topic["estimated_hours"] // time_per_week)
            end_week = current_week + weeks_needed - 1
            
            timeline.append({
                "week_range": f"Week {current_week}" + (f"-{end_week}" if weeks_needed > 1 else ""),
                "topic": topic["title"],
                "hours_per_week": min(time_per_week, topic["estimated_hours"]),
                "total_hours": topic["estimated_hours"],
                "milestones": [
                    f"Complete {topic['title']} fundamentals",
                    f"Practice {topic['skills_covered'][0] if topic['skills_covered'] else 'core concepts'}",
                    f"Build a project using {topic['title']} skills"
                ]
            })
            current_week = end_week + 1
        
        return {
            "learning_path": learning_path,
            "total_estimated_hours": total_hours,
            "timeline": timeline,
            "recommended_career_paths": get_career_suggestions(interests)
        }
    
    def get_career_suggestions(interests):
        """Get career suggestions based on interests"""
        career_mapping = {
            'programming': ['Software Developer', 'Full Stack Developer'],
            'data': ['Data Analyst', 'Data Scientist'],
            'ai_ml': ['Machine Learning Engineer', 'AI Researcher'],
            'web': ['Frontend Developer', 'Backend Developer'],
            'database': ['Database Administrator', 'Data Engineer']
        }
        
        suggestions = []
        for interest in interests:
            for key, careers in career_mapping.items():
                if key in interest:
                    suggestions.extend(careers)
        
        return list(set(suggestions)) if suggestions else ['Software Developer', 'Data Analyst']
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'success': True,
            'message': 'Learning Path Recommender API is running',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/recommend', methods=['POST'])
    def get_recommendations():
        try:
            data = request.get_json() or {}
            recommendations = generate_recommendations(data)
            
            return jsonify({
                'success': True,
                'recommendations': recommendations
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/topics', methods=['GET'])
    def get_topics():
        return jsonify({
            'success': True,
            'topics': SAMPLE_TOPICS
        })
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=False)

except ImportError as e:
    print(f"Import error in simplified backend: {e}")
    # Create minimal Flask app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/api/health')
    def health():
        return jsonify({'success': True, 'message': 'Minimal backend running'})
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=False)
