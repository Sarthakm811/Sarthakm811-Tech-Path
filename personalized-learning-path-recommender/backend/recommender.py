import random
from typing import List, Dict, Any

class LearningPathRecommender:
    def __init__(self):
        # Define learning topics with their metadata using grade-based progression
        self.topics = {
            'python_basics': {
                'id': 'python_basics',
                'title': 'Basic Programming with Python',
                'grade': 9,  # Beginner level
                'topic': 'Programming',
                'level': 'beginner',
                'category': 'programming',
                'estimated_hours': 20,
                'description': 'Learn fundamental Python syntax, variables, loops, and functions',
                'prerequisites': [],
                'skills_covered': ['Variables', 'Data Types', 'Control Flow', 'Functions', 'Error Handling']
            },
            'python_advanced': {
                'id': 'python_advanced',
                'title': 'Intermediate Python',
                'grade': 10,  # Intermediate level
                'topic': 'Programming',
                'level': 'intermediate',
                'category': 'programming',
                'estimated_hours': 30,
                'description': 'Object-oriented programming, decorators, generators, and advanced concepts',
                'prerequisites': ['python_basics'],
                'skills_covered': ['OOP', 'Decorators', 'Generators', 'Context Managers', 'Metaclasses']
            },
            'data_analysis': {
                'id': 'data_analysis',
                'title': 'Introduction to Data Science',
                'grade': 11,  # Intermediate level
                'topic': 'Data Science',
                'level': 'intermediate',
                'category': 'data',
                'estimated_hours': 30,
                'description': 'Pandas, NumPy, Matplotlib, and statistical analysis',
                'prerequisites': ['python_basics'],
                'skills_covered': ['Pandas', 'NumPy', 'Matplotlib', 'Statistical Analysis', 'Data Visualization']
            },
            'machine_learning': {
                'id': 'machine_learning',
                'title': 'Machine Learning Basics',
                'grade': 12,  # Advanced level
                'topic': 'AI/ML',
                'level': 'advanced',
                'category': 'ai_ml',
                'estimated_hours': 35,
                'description': 'Introduction to ML concepts, supervised and unsupervised learning',
                'prerequisites': ['python_advanced', 'data_analysis'],
                'skills_covered': ['Linear Regression', 'Classification', 'Clustering', 'Feature Engineering']
            },
            'deep_learning': {
                'id': 'deep_learning',
                'title': 'Advanced AI Techniques',
                'grade': 12,  # Advanced level
                'topic': 'AI/ML',
                'level': 'advanced',
                'category': 'ai_ml',
                'estimated_hours': 50,
                'description': 'Neural networks, CNNs, RNNs, and modern architectures',
                'prerequisites': ['machine_learning'],
                'skills_covered': ['Neural Networks', 'CNNs', 'RNNs', 'Transformers', 'TensorFlow/PyTorch']
            },
            'web_development': {
                'id': 'web_development',
                'title': 'Web Development Fundamentals',
                'grade': 10,  # Intermediate level
                'topic': 'Web Dev',
                'level': 'intermediate',
                'category': 'web',
                'estimated_hours': 25,
                'description': 'HTML, CSS, JavaScript, and basic web frameworks',
                'prerequisites': [],
                'skills_covered': ['HTML', 'CSS', 'JavaScript', 'React/Vue', 'Node.js']
            },
            'frontend_frameworks': {
                'id': 'frontend_frameworks',
                'title': 'Frontend Frameworks',
                'grade': 11,  # Advanced level
                'topic': 'Web Dev',
                'level': 'advanced',
                'category': 'web',
                'estimated_hours': 30,
                'description': 'Advanced frontend development with modern frameworks',
                'prerequisites': ['web_development'],
                'skills_covered': ['React', 'Vue.js', 'Angular', 'State Management', 'Component Architecture']
            },
            'databases': {
                'id': 'databases',
                'title': 'Databases and SQL',
                'grade': 11,  # Intermediate level
                'topic': 'Databases',
                'level': 'intermediate',
                'category': 'data',
                'estimated_hours': 20,
                'description': 'SQL, NoSQL, and database design principles',
                'prerequisites': ['python_basics'],
                'skills_covered': ['SQL', 'PostgreSQL', 'MongoDB', 'Database Design', 'Query Optimization']
            },
            'advanced_databases': {
                'id': 'advanced_databases',
                'title': 'Advanced Database Systems',
                'grade': 12,  # Advanced level
                'topic': 'Databases',
                'level': 'advanced',
                'category': 'data',
                'estimated_hours': 25,
                'description': 'Advanced database concepts, optimization, and distributed systems',
                'prerequisites': ['databases'],
                'skills_covered': ['Database Optimization', 'Distributed Systems', 'Data Warehousing', 'Big Data']
            },
            'data_structures': {
                'id': 'data_structures',
                'title': 'Data Structures and Algorithms',
                'grade': 11,  # Intermediate level
                'topic': 'Computer Science',
                'level': 'intermediate',
                'category': 'computer_science',
                'estimated_hours': 40,
                'description': 'Essential data structures and algorithm design patterns',
                'prerequisites': ['python_basics'],
                'skills_covered': ['Arrays', 'Linked Lists', 'Trees', 'Graphs', 'Sorting', 'Searching']
            },
            'cloud_computing': {
                'id': 'cloud_computing',
                'title': 'Cloud Computing',
                'grade': 11,  # Intermediate level
                'topic': 'Cloud/DevOps',
                'level': 'intermediate',
                'category': 'infrastructure',
                'estimated_hours': 25,
                'description': 'AWS, Azure, Docker, and cloud deployment strategies',
                'prerequisites': ['web_development'],
                'skills_covered': ['AWS/Azure', 'Docker', 'Kubernetes', 'CI/CD', 'Cloud Security']
            },
            'cybersecurity': {
                'id': 'cybersecurity',
                'title': 'Cybersecurity Fundamentals',
                'grade': 10,  # Intermediate level
                'topic': 'Security',
                'level': 'intermediate',
                'category': 'security',
                'estimated_hours': 30,
                'description': 'Network security, encryption, and ethical hacking basics',
                'prerequisites': ['python_basics'],
                'skills_covered': ['Network Security', 'Encryption', 'Penetration Testing', 'Security Policies']
            }
        }
        
        # Define learning paths for different career tracks
        self.career_paths = {
            'data_scientist': ['python_basics', 'data_analysis', 'machine_learning', 'deep_learning', 'databases'],
            'web_developer': ['python_basics', 'web_development', 'databases', 'cloud_computing'],
            'ai_researcher': ['python_basics', 'python_advanced', 'data_structures', 'machine_learning', 'deep_learning'],
            'cybersecurity_analyst': ['python_basics', 'cybersecurity', 'data_analysis', 'cloud_computing'],
            'full_stack_developer': ['python_basics', 'web_development', 'databases', 'cloud_computing', 'data_analysis']
        }

    def get_available_topics(self) -> List[Dict[str, Any]]:
        """Return list of all available topics"""
        return [
            {
                'id': topic_id,
                'name': topic.get('title', topic.get('name', 'Unknown Course')),
                'level': topic['level'],
                'category': topic['category'],
                'estimated_hours': topic['estimated_hours'],
                'description': topic['description']
            }
            for topic_id, topic in self.topics.items()
        ]

    def get_recommendations(self, user_level: str, interests: List[str], 
                          goals: List[str], time_per_week: int) -> Dict[str, Any]:
        """
        Generate personalized learning path recommendations
        """
        # Filter topics based on user level
        suitable_topics = self._filter_by_level(user_level)
        
        # Score topics based on interests and goals
        scored_topics = self._score_topics(suitable_topics, interests, goals)
        
        # Generate learning path
        learning_path = self._generate_path(scored_topics, time_per_week)
        
        # Calculate timeline
        timeline = self._calculate_timeline(learning_path, time_per_week)
        
        return {
            'learning_path': learning_path,
            'timeline': timeline,
            'total_estimated_hours': sum(topic['estimated_hours'] for topic in learning_path),
            'recommended_career_paths': self._suggest_career_paths(goals, interests)
        }

    def _filter_by_level(self, user_level: str) -> List[Dict[str, Any]]:
        """Filter topics based on user's current level"""
        level_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        user_level_num = level_order.get(user_level, 0)
        
        suitable_topics = []
        for topic_id, topic in self.topics.items():
            topic_level_num = level_order.get(topic['level'], 0)
            # Include topics at or below user's level
            if topic_level_num <= user_level_num:
                topic_copy = topic.copy()
                topic_copy['id'] = topic_id
                suitable_topics.append(topic_copy)
        
        return suitable_topics

    def _score_topics(self, topics: List[Dict], interests: List[str], goals: List[str]) -> List[Dict]:
        """Score topics based on user interests and goals with improved matching"""
        
        # Create interest mapping for better matching
        interest_mapping = {
            'programming': ['programming', 'python', 'code', 'software'],
            'data': ['data', 'data science', 'analysis', 'database', 'sql'],
            'ai_ml': ['ai', 'machine learning', 'deep learning', 'artificial intelligence', 'ml'],
            'web': ['web', 'frontend', 'backend', 'html', 'css', 'javascript'],
            'computer_science': ['algorithm', 'data structure', 'computer science', 'cs'],
            'security': ['security', 'cybersecurity', 'hacking', 'encryption'],
            'infrastructure': ['cloud', 'devops', 'aws', 'azure', 'docker'],
            'mobile': ['mobile', 'app', 'ios', 'android'],
            'game_development': ['game', 'gaming', 'unity', 'unreal']
        }
        
        # Create goal keywords for better matching
        goal_keywords = {
            'data_scientist': ['data scientist', 'data science', 'analyst', 'statistics', 'machine learning'],
            'web_developer': ['web developer', 'frontend', 'backend', 'full stack', 'website'],
            'ai_researcher': ['ai researcher', 'artificial intelligence', 'machine learning', 'deep learning'],
            'cybersecurity': ['cybersecurity', 'security analyst', 'penetration testing', 'ethical hacking'],
            'software_engineer': ['software engineer', 'developer', 'programming', 'coding']
        }
        
        for topic in topics:
            score = 0
            
            # Score based on interests (improved matching)
            for interest in interests:
                interest_lower = interest.lower()
                
                # Direct category match
                if interest_lower in topic['category'].lower():
                    score += 5
                
                # Topic field match
                if interest_lower in topic.get('topic', '').lower():
                    score += 4
                
                # Title match
                if interest_lower in topic['title'].lower():
                    score += 3
                
                # Description match
                if interest_lower in topic['description'].lower():
                    score += 2
                
                # Skills match
                for skill in topic.get('skills_covered', []):
                    if interest_lower in skill.lower():
                        score += 1
                
                # Use interest mapping for broader matching
                if interest_lower in interest_mapping:
                    for keyword in interest_mapping[interest_lower]:
                        if keyword in topic['title'].lower() or keyword in topic['description'].lower():
                            score += 2
            
            # Score based on goals (improved matching)
            for goal in goals:
                goal_lower = goal.lower()
                
                # Direct goal match
                if goal_lower in topic['description'].lower():
                    score += 4
                
                if goal_lower in topic['title'].lower():
                    score += 3
                
                # Check against goal keywords
                for career, keywords in goal_keywords.items():
                    for keyword in keywords:
                        if keyword in goal_lower:
                            # If this topic is relevant to this career path
                            if topic['id'] in self.career_paths.get(career, []):
                                score += 3
                
                # Skills match with goals
                for skill in topic.get('skills_covered', []):
                    if goal_lower in skill.lower() or any(word in goal_lower for word in skill.lower().split()):
                        score += 2
            
            # Bonus for beginner-friendly topics if user is beginner
            if topic.get('level') == 'beginner' and topic.get('grade', 0) <= 9:
                score += 2
            
            topic['score'] = score
        
        # Sort by score (highest first), then by grade (lowest first for progression)
        return sorted(topics, key=lambda x: (x['score'], -x.get('grade', 0)), reverse=True)

    def _generate_path(self, scored_topics: List[Dict], time_per_week: int) -> List[Dict]:
        """Generate a learning path based on scored topics with proper progression"""
        path = []
        total_hours = 0
        max_hours = time_per_week * 16  # 4 months of learning
        
        # Track added topics and their grades for progression
        added_topics = set()
        max_grade = 9  # Start with beginner level
        
        # Sort topics by score and grade for better progression
        sorted_topics = sorted(scored_topics, key=lambda x: (x['score'], x.get('grade', 9)), reverse=True)
        
        for topic in sorted_topics:
            if topic['id'] in added_topics:
                continue
            
            # Ensure proper grade progression (don't jump too far ahead)
            topic_grade = topic.get('grade', 9)
            if topic_grade > max_grade + 1:  # Don't skip more than one grade level
                continue
                
            # Check if we have time for this topic
            if total_hours + topic['estimated_hours'] > max_hours:
                continue
            
            # Add prerequisites first
            for prereq_id in topic.get('prerequisites', []):
                if prereq_id not in added_topics and prereq_id in self.topics:
                    prereq_topic = self.topics[prereq_id].copy()
                    prereq_topic['id'] = prereq_id
                    
                    # Check if prerequisite is at appropriate level
                    prereq_grade = prereq_topic.get('grade', 9)
                    if prereq_grade <= max_grade + 1 and total_hours + prereq_topic['estimated_hours'] <= max_hours:
                        path.append(prereq_topic)
                        added_topics.add(prereq_id)
                        total_hours += prereq_topic['estimated_hours']
                        max_grade = max(max_grade, prereq_grade)
            
            # Add the topic itself
            if total_hours + topic['estimated_hours'] <= max_hours:
                path.append(topic)
                added_topics.add(topic['id'])
                total_hours += topic['estimated_hours']
                max_grade = max(max_grade, topic_grade)
        
        # Sort final path by grade for logical progression
        return sorted(path, key=lambda x: x.get('grade', 9))

    def _calculate_timeline(self, learning_path: List[Dict], time_per_week: int) -> List[Dict]:
        """Calculate timeline for the learning path"""
        timeline = []
        current_week = 1
        
        for topic in learning_path:
            weeks_needed = max(1, topic['estimated_hours'] // time_per_week)
            
            timeline.append({
                'week_range': f"Week {current_week} - Week {current_week + weeks_needed - 1}",
                'topic': topic.get('title', topic.get('name', 'Unknown Course')),
                'hours_per_week': min(time_per_week, topic['estimated_hours']),
                'total_hours': topic['estimated_hours'],
                'milestones': self._generate_milestones(topic)
            })
            
            current_week += weeks_needed
        
        return timeline

    def _generate_milestones(self, topic: Dict) -> List[str]:
        """Generate learning milestones for a topic"""
        milestones = []
        skills = topic.get('skills_covered', [])
        
        for i, skill in enumerate(skills, 1):
            milestones.append(f"Complete {skill} module")
        
        return milestones

    def _suggest_career_paths(self, goals: List[str], interests: List[str]) -> List[str]:
        """Suggest career paths based on goals and interests"""
        suggested_paths = []
        
        for career, topics in self.career_paths.items():
            score = 0
            for goal in goals:
                if goal.lower() in career.lower():
                    score += 2
            
            for interest in interests:
                if interest.lower() in career.lower():
                    score += 1
            
            if score > 0:
                suggested_paths.append(career.replace('_', ' ').title())
        
        return suggested_paths[:3]  # Return top 3 suggestions
