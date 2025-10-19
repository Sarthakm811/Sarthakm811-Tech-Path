"""
Advanced User Modeling System for Personalized Learning Path Recommender
Implements sophisticated user profiling, learning pattern analysis, and behavioral modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import os

@dataclass
class LearningEvent:
    """Represents a learning event in the user's journey"""
    topic_id: str
    start_time: datetime
    end_time: datetime
    completion_percentage: float
    difficulty_rating: int  # 1-5 scale
    engagement_score: float  # 0-1 scale
    learning_style: str  # visual, auditory, kinesthetic, reading
    platform: str  # web, mobile, desktop
    session_quality: float  # 0-1 scale

@dataclass
class UserPreferences:
    """User learning preferences and characteristics"""
    preferred_learning_style: str
    optimal_session_duration: int  # minutes
    preferred_time_of_day: str  # morning, afternoon, evening
    difficulty_preference: str  # easy, medium, hard
    learning_goals: List[str]
    interests: List[str]
    available_time_per_week: int
    preferred_pace: str  # slow, medium, fast

class UserModel:
    """
    Advanced user modeling system that creates comprehensive user profiles
    and predicts learning behavior using machine learning techniques
    """
    
    def __init__(self):
        self.users = {}
        self.learning_events = {}
        self.user_clusters = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def create_user_profile(self, user_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive user profile with initial data"""
        
        user_profile = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'preferences': UserPreferences(
                preferred_learning_style=initial_data.get('learning_style', 'visual'),
                optimal_session_duration=initial_data.get('session_duration', 45),
                preferred_time_of_day=initial_data.get('time_preference', 'afternoon'),
                difficulty_preference=initial_data.get('difficulty_preference', 'medium'),
                learning_goals=initial_data.get('goals', []),
                interests=initial_data.get('interests', []),
                available_time_per_week=initial_data.get('time_per_week', 10),
                preferred_pace=initial_data.get('pace', 'medium')
            ),
            'learning_history': [],
            'skill_progress': {},
            'learning_patterns': {
                'completion_rate': 0.0,
                'average_session_duration': 0.0,
                'preferred_topics': [],
                'difficulty_progression': [],
                'learning_velocity': 0.0,
                'engagement_trend': [],
                'optimal_learning_times': []
            },
            'behavioral_metrics': {
                'consistency_score': 0.0,
                'persistence_score': 0.0,
                'curiosity_score': 0.0,
                'collaboration_score': 0.0,
                'adaptability_score': 0.0
            },
            'recommendation_history': [],
            'feedback_history': []
        }
        
        self.users[user_id] = user_profile
        self.learning_events[user_id] = []
        
        return user_profile
    
    def add_learning_event(self, user_id: str, event: LearningEvent):
        """Add a learning event and update user model"""
        if user_id not in self.learning_events:
            self.learning_events[user_id] = []
        
        self.learning_events[user_id].append(event)
        self._update_user_patterns(user_id)
        self._calculate_behavioral_metrics(user_id)
    
    def _update_user_patterns(self, user_id: str):
        """Update learning patterns based on recent events"""
        if user_id not in self.learning_events:
            return
        
        events = self.learning_events[user_id]
        if not events:
            return
        
        # Calculate completion rate
        completed_events = [e for e in events if e.completion_percentage >= 0.8]
        completion_rate = len(completed_events) / len(events) if events else 0.0
        
        # Calculate average session duration
        avg_duration = np.mean([(e.end_time - e.start_time).total_seconds() / 60 
                               for e in events]) if events else 0.0
        
        # Find preferred topics
        topic_counts = {}
        for event in events:
            topic_counts[event.topic_id] = topic_counts.get(event.topic_id, 0) + 1
        preferred_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate learning velocity (topics completed per week)
        if len(events) >= 2:
            time_span = (events[-1].start_time - events[0].start_time).days / 7
            velocity = len(completed_events) / max(time_span, 1)
        else:
            velocity = 0.0
        
        # Update patterns
        self.users[user_id]['learning_patterns'].update({
            'completion_rate': completion_rate,
            'average_session_duration': avg_duration,
            'preferred_topics': [t[0] for t in preferred_topics],
            'learning_velocity': velocity,
            'last_updated': datetime.now()
        })
    
    def _calculate_behavioral_metrics(self, user_id: str):
        """Calculate advanced behavioral metrics"""
        if user_id not in self.learning_events:
            return
        
        events = self.learning_events[user_id]
        if not events:
            return
        
        # Consistency Score: Regularity of learning sessions
        if len(events) >= 3:
            session_intervals = []
            for i in range(1, len(events)):
                interval = (events[i].start_time - events[i-1].start_time).days
                session_intervals.append(interval)
            
            if session_intervals:
                consistency_score = 1.0 - (np.std(session_intervals) / np.mean(session_intervals))
                consistency_score = max(0.0, min(1.0, consistency_score))
            else:
                consistency_score = 0.0
        else:
            consistency_score = 0.5  # Default for new users
        
        # Persistence Score: Tendency to complete difficult topics
        difficult_events = [e for e in events if e.difficulty_rating >= 4]
        if difficult_events:
            persistence_score = np.mean([e.completion_percentage for e in difficult_events])
        else:
            persistence_score = 0.5
        
        # Curiosity Score: Diversity of topics explored
        unique_topics = len(set(e.topic_id for e in events))
        total_events = len(events)
        curiosity_score = min(1.0, unique_topics / max(total_events * 0.3, 1))
        
        # Engagement Score: Overall engagement with learning
        engagement_score = np.mean([e.engagement_score for e in events]) if events else 0.5
        
        # Adaptability Score: How well user adjusts to different learning styles
        learning_styles = [e.learning_style for e in events]
        style_diversity = len(set(learning_styles)) / len(learning_styles) if learning_styles else 0.5
        adaptability_score = (style_diversity + engagement_score) / 2
        
        self.users[user_id]['behavioral_metrics'].update({
            'consistency_score': consistency_score,
            'persistence_score': persistence_score,
            'curiosity_score': curiosity_score,
            'engagement_score': engagement_score,
            'adaptability_score': adaptability_score
        })
    
    def get_user_features(self, user_id: str) -> np.ndarray:
        """Extract feature vector for machine learning models"""
        if user_id not in self.users:
            return np.zeros(20)  # Default feature vector
        
        user = self.users[user_id]
        patterns = user['learning_patterns']
        metrics = user['behavioral_metrics']
        prefs = user['preferences']
        
        features = [
            patterns['completion_rate'],
            patterns['average_session_duration'] / 60,  # Normalize to hours
            patterns['learning_velocity'],
            metrics['consistency_score'],
            metrics['persistence_score'],
            metrics['curiosity_score'],
            metrics['engagement_score'],
            metrics['adaptability_score'],
            len(prefs.interests),
            len(prefs.learning_goals),
            prefs.available_time_per_week,
            1.0 if prefs.preferred_learning_style == 'visual' else 0.0,
            1.0 if prefs.preferred_learning_style == 'auditory' else 0.0,
            1.0 if prefs.preferred_learning_style == 'kinesthetic' else 0.0,
            1.0 if prefs.difficulty_preference == 'easy' else 0.0,
            1.0 if prefs.difficulty_preference == 'medium' else 0.0,
            1.0 if prefs.difficulty_preference == 'hard' else 0.0,
            1.0 if prefs.preferred_pace == 'slow' else 0.0,
            1.0 if prefs.preferred_pace == 'medium' else 0.0,
            1.0 if prefs.preferred_pace == 'fast' else 0.0
        ]
        
        return np.array(features)
    
    def create_user_clusters(self, n_clusters: int = 5):
        """Create user clusters for collaborative filtering"""
        if len(self.users) < n_clusters:
            return
        
        # Extract features for all users
        user_features = []
        user_ids = []
        
        for user_id in self.users.keys():
            features = self.get_user_features(user_id)
            user_features.append(features)
            user_ids.append(user_id)
        
        if not user_features:
            return
        
        # Normalize features
        user_features = np.array(user_features)
        normalized_features = self.scaler.fit_transform(user_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        # Store cluster information
        self.user_clusters = {
            'labels': cluster_labels,
            'user_ids': user_ids,
            'centers': kmeans.cluster_centers_,
            'model': kmeans
        }
    
    def find_similar_users(self, user_id: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Find similar users using collaborative filtering"""
        if user_id not in self.users or self.user_clusters is None:
            return []
        
        user_features = self.get_user_features(user_id).reshape(1, -1)
        normalized_features = self.scaler.transform(user_features)
        
        # Find user's cluster
        user_cluster = self.user_clusters['model'].predict(normalized_features)[0]
        
        # Find users in the same cluster
        cluster_users = []
        for i, (label, uid) in enumerate(zip(self.user_clusters['labels'], self.user_clusters['user_ids'])):
            if label == user_cluster and uid != user_id:
                # Calculate similarity within cluster
                other_features = self.get_user_features(uid).reshape(1, -1)
                similarity = cosine_similarity(normalized_features, other_features)[0][0]
                cluster_users.append((uid, similarity))
        
        # Sort by similarity and return top N
        cluster_users.sort(key=lambda x: x[1], reverse=True)
        return cluster_users[:n_similar]
    
    def predict_user_preference(self, user_id: str, topic_features: np.ndarray) -> float:
        """Predict user preference for a topic using learned patterns"""
        if user_id not in self.users:
            return 0.5  # Default preference
        
        user_features = self.get_user_features(user_id)
        
        # Simple linear combination (can be replaced with more sophisticated ML models)
        preference_weights = np.array([
            0.2,  # completion_rate
            0.1,  # avg_session_duration
            0.15, # learning_velocity
            0.1,  # consistency_score
            0.1,  # persistence_score
            0.15, # curiosity_score
            0.1,  # engagement_score
            0.1   # adaptability_score
        ])
        
        # Calculate weighted preference score
        preference_score = np.dot(user_features[:8], preference_weights)
        
        # Normalize to 0-1 range
        preference_score = max(0.0, min(1.0, preference_score))
        
        return preference_score
    
    def update_user_feedback(self, user_id: str, topic_id: str, rating: float, feedback: str = ""):
        """Update user model based on explicit feedback"""
        if user_id not in self.users:
            return
        
        feedback_entry = {
            'topic_id': topic_id,
            'rating': rating,
            'feedback': feedback,
            'timestamp': datetime.now()
        }
        
        self.users[user_id]['feedback_history'].append(feedback_entry)
        self._update_user_patterns(user_id)
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive user insights"""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        patterns = user['learning_patterns']
        metrics = user['behavioral_metrics']
        
        insights = {
            'learning_strengths': [],
            'learning_challenges': [],
            'recommendations': [],
            'progress_summary': {
                'total_sessions': len(self.learning_events.get(user_id, [])),
                'completion_rate': patterns['completion_rate'],
                'average_session_duration': patterns['average_session_duration'],
                'learning_velocity': patterns['learning_velocity']
            },
            'behavioral_analysis': {
                'learning_style': user['preferences'].preferred_learning_style,
                'consistency_level': self._get_consistency_level(metrics['consistency_score']),
                'difficulty_preference': user['preferences'].difficulty_preference,
                'engagement_level': self._get_engagement_level(metrics['engagement_score'])
            }
        }
        
        # Generate strengths and challenges
        if metrics['persistence_score'] > 0.7:
            insights['learning_strengths'].append("High persistence with challenging topics")
        if metrics['curiosity_score'] > 0.7:
            insights['learning_strengths'].append("Explores diverse topics")
        if metrics['consistency_score'] > 0.7:
            insights['learning_strengths'].append("Consistent learning schedule")
        
        if patterns['completion_rate'] < 0.6:
            insights['learning_challenges'].append("Low completion rate - consider shorter sessions")
        if patterns['average_session_duration'] > 120:
            insights['learning_challenges'].append("Long sessions may lead to fatigue")
        
        # Generate recommendations
        if metrics['adaptability_score'] < 0.5:
            insights['recommendations'].append("Try different learning styles to improve adaptability")
        if patterns['learning_velocity'] < 0.5:
            insights['recommendations'].append("Consider increasing learning frequency")
        
        return insights
    
    def _get_consistency_level(self, score: float) -> str:
        """Convert consistency score to descriptive level"""
        if score > 0.8:
            return "Very Consistent"
        elif score > 0.6:
            return "Consistent"
        elif score > 0.4:
            return "Moderately Consistent"
        else:
            return "Inconsistent"
    
    def _get_engagement_level(self, score: float) -> str:
        """Convert engagement score to descriptive level"""
        if score > 0.8:
            return "Highly Engaged"
        elif score > 0.6:
            return "Engaged"
        elif score > 0.4:
            return "Moderately Engaged"
        else:
            return "Low Engagement"
    
    def save_model(self, filepath: str):
        """Save the user model to disk"""
        model_data = {
            'users': self.users,
            'learning_events': self.learning_events,
            'user_clusters': self.user_clusters,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the user model from disk"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.users = model_data.get('users', {})
        self.learning_events = model_data.get('learning_events', {})
        self.user_clusters = model_data.get('user_clusters', None)
        
        # Restore scaler
        if model_data.get('scaler_mean') and model_data.get('scaler_scale'):
            self.scaler.mean_ = np.array(model_data['scaler_mean'])
            self.scaler.scale_ = np.array(model_data['scaler_scale'])
