"""
Advanced Recommendation Engine for Personalized Learning Path Recommender
Implements sophisticated ML algorithms including collaborative filtering, content-based filtering,
neural networks, and adaptive learning systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import pickle
import os
from datetime import datetime, timedelta

from .user_model import UserModel, LearningEvent
from .knowledge_graph import KnowledgeGraph, Topic, Skill

class AdvancedRecommender:
    """
    Advanced recommendation engine that combines multiple ML approaches
    for personalized learning path recommendations
    """
    
    def __init__(self):
        self.user_model = UserModel()
        self.knowledge_graph = KnowledgeGraph()
        
        # ML Models
        self.collaborative_filter = None
        self.content_based_filter = None
        self.neural_recommender = None
        self.adaptive_learning_model = None
        
        # Feature extractors
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Model parameters
        self.model_weights = {
            'collaborative': 0.3,
            'content_based': 0.25,
            'neural': 0.25,
            'adaptive': 0.2
        }
        
        # Learning from feedback
        self.feedback_history = []
        self.model_performance = {}
        
    def train_collaborative_filter(self, user_interactions: List[Dict[str, Any]]):
        """Train collaborative filtering model using user-item interactions"""
        
        if not user_interactions:
            return
        
        # Create user-item matrix
        df = pd.DataFrame(user_interactions)
        user_item_matrix = df.pivot_table(
            index='user_id', 
            columns='topic_id', 
            values='rating', 
            fill_value=0
        )
        
        # Calculate user-user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Store the model
        self.collaborative_filter = {
            'user_item_matrix': user_item_matrix,
            'user_similarity': user_similarity,
            'user_ids': user_item_matrix.index.tolist()
        }
        
        print(f"Collaborative filter trained on {len(user_interactions)} interactions")
    
    def train_content_based_filter(self, topics: Dict[str, Topic]):
        """Train content-based filtering model using topic features"""
        
        if not topics:
            return
        
        # Prepare content features
        topic_descriptions = []
        topic_ids = []
        
        for topic_id, topic in topics.items():
            # Combine title, description, and tags for content analysis
            content = f"{topic.title} {topic.description} {' '.join(topic.tags)}"
            topic_descriptions.append(content)
            topic_ids.append(topic_id)
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(topic_descriptions)
        
        # Calculate topic-topic similarity
        topic_similarity = cosine_similarity(tfidf_matrix)
        
        # Store the model
        self.content_based_filter = {
            'tfidf_matrix': tfidf_matrix,
            'topic_similarity': topic_similarity,
            'topic_ids': topic_ids,
            'vectorizer': self.tfidf_vectorizer
        }
        
        print(f"Content-based filter trained on {len(topics)} topics")
    
    def train_neural_recommender(self, training_data: List[Dict[str, Any]]):
        """Train neural network for recommendation scoring"""
        
        if not training_data:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for data_point in training_data:
            # Extract features
            user_features = self.user_model.get_user_features(data_point['user_id'])
            topic_features = self._extract_topic_features(data_point['topic_id'])
            
            # Combine features
            combined_features = np.concatenate([user_features, topic_features])
            X.append(combined_features)
            y.append(data_point['rating'])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            return
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Build neural network
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Store model
        self.neural_recommender = model
        
        # Calculate performance
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        self.model_performance['neural'] = test_loss
        
        print(f"Neural recommender trained with MSE: {test_loss:.4f}")
    
    def train_adaptive_learning_model(self, learning_events: List[LearningEvent]):
        """Train adaptive learning model that adjusts to user behavior"""
        
        if not learning_events:
            return
        
        # Group events by user
        user_events = {}
        for event in learning_events:
            # We need to get user_id from somewhere - this is a simplified approach
            # In a real system, events would be properly linked to users
            if hasattr(event, 'user_id'):
                if event.user_id not in user_events:
                    user_events[event.user_id] = []
                user_events[event.user_id].append(event)
        
        # Train Random Forest for adaptive learning
        X = []
        y = []
        
        for user_id, events in user_events.items():
            for event in events:
                # Extract context features
                features = self._extract_learning_context_features(event)
                if features is not None:
                    X.append(features)
                    y.append(event.engagement_score)
        
        if len(X) == 0:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        rf_model.fit(X, y)
        
        # Store model
        self.adaptive_learning_model = rf_model
        
        # Calculate performance
        score = rf_model.score(X, y)
        self.model_performance['adaptive'] = score
        
        print(f"Adaptive learning model trained with R²: {score:.4f}")
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10,
                          context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get personalized recommendations using ensemble of models"""
        
        if context is None:
            context = {}
        
        # Get user profile
        user_profile = self.user_model.users.get(user_id)
        if not user_profile:
            return []
        
        # Get candidate topics
        candidate_topics = self._get_candidate_topics(user_id, context)
        
        if not candidate_topics:
            return []
        
        # Score topics using different models
        recommendations = []
        
        for topic_id in candidate_topics:
            scores = {}
            
            # Collaborative filtering score
            if self.collaborative_filter:
                scores['collaborative'] = self._get_collaborative_score(user_id, topic_id)
            
            # Content-based score
            if self.content_based_filter:
                scores['content_based'] = self._get_content_based_score(user_id, topic_id)
            
            # Neural network score
            if self.neural_recommender:
                scores['neural'] = self._get_neural_score(user_id, topic_id)
            
            # Adaptive learning score
            if self.adaptive_learning_model:
                scores['adaptive'] = self._get_adaptive_score(user_id, topic_id, context)
            
            # Ensemble score
            ensemble_score = 0.0
            total_weight = 0.0
            
            for model_name, score in scores.items():
                weight = self.model_weights.get(model_name, 0.0)
                ensemble_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
            
            # Get topic details
            topic = self.knowledge_graph.topics.get(topic_id)
            if topic:
                recommendations.append({
                    'topic_id': topic_id,
                    'title': topic.title,
                    'description': topic.description,
                    'category': topic.category,
                    'difficulty_level': topic.difficulty_level,
                    'estimated_hours': topic.estimated_hours,
                    'score': ensemble_score,
                    'model_scores': scores,
                    'skills_covered': topic.skills_covered,
                    'prerequisites': topic.prerequisites
                })
        
        # Sort by ensemble score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _get_collaborative_score(self, user_id: str, topic_id: str) -> float:
        """Get collaborative filtering score for a topic"""
        if not self.collaborative_filter:
            return 0.5
        
        user_item_matrix = self.collaborative_filter['user_item_matrix']
        user_similarity = self.collaborative_filter['user_similarity']
        user_ids = self.collaborative_filter['user_ids']
        
        if user_id not in user_ids or topic_id not in user_item_matrix.columns:
            return 0.5
        
        user_idx = user_ids.index(user_id)
        topic_ratings = user_item_matrix[topic_id].values
        
        # Find similar users who rated this topic
        similar_users = np.argsort(user_similarity[user_idx])[-10:]  # Top 10 similar users
        similar_ratings = topic_ratings[similar_users]
        similar_similarities = user_similarity[user_idx][similar_users]
        
        # Calculate weighted average rating
        valid_ratings = similar_ratings > 0
        if np.any(valid_ratings):
            weighted_rating = np.average(
                similar_ratings[valid_ratings],
                weights=similar_similarities[valid_ratings]
            )
            return max(0.0, min(1.0, weighted_rating))
        
        return 0.5
    
    def _get_content_based_score(self, user_id: str, topic_id: str) -> float:
        """Get content-based filtering score for a topic"""
        if not self.content_based_filter:
            return 0.5
        
        user_profile = self.user_model.users.get(user_id)
        if not user_profile:
            return 0.5
        
        # Get user's preferred topics
        preferred_topics = user_profile['learning_patterns'].get('preferred_topics', [])
        if not preferred_topics:
            return 0.5
        
        topic_ids = self.content_based_filter['topic_ids']
        topic_similarity = self.content_based_filter['topic_similarity']
        
        if topic_id not in topic_ids:
            return 0.5
        
        topic_idx = topic_ids.index(topic_id)
        
        # Calculate similarity to user's preferred topics
        similarities = []
        for pref_topic in preferred_topics:
            if pref_topic in topic_ids:
                pref_idx = topic_ids.index(pref_topic)
                similarity = topic_similarity[topic_idx][pref_idx]
                similarities.append(similarity)
        
        if similarities:
            return max(0.0, min(1.0, np.mean(similarities)))
        
        return 0.5
    
    def _get_neural_score(self, user_id: str, topic_id: str) -> float:
        """Get neural network score for a topic"""
        if not self.neural_recommender:
            return 0.5
        
        # Get features
        user_features = self.user_model.get_user_features(user_id)
        topic_features = self._extract_topic_features(topic_id)
        
        if topic_features is None:
            return 0.5
        
        # Combine features
        combined_features = np.concatenate([user_features, topic_features])
        combined_features = combined_features.reshape(1, -1)
        
        # Normalize features
        combined_features_scaled = self.scaler.transform(combined_features)
        
        # Predict
        score = self.neural_recommender.predict(combined_features_scaled)[0][0]
        return max(0.0, min(1.0, score))
    
    def _get_adaptive_score(self, user_id: str, topic_id: str, context: Dict[str, Any]) -> float:
        """Get adaptive learning score for a topic"""
        if not self.adaptive_learning_model:
            return 0.5
        
        # Create learning context
        context_features = self._create_learning_context(user_id, topic_id, context)
        
        if context_features is None:
            return 0.5
        
        # Predict engagement
        engagement_score = self.adaptive_learning_model.predict([context_features])[0]
        return max(0.0, min(1.0, engagement_score))
    
    def _extract_topic_features(self, topic_id: str) -> Optional[np.ndarray]:
        """Extract numerical features for a topic"""
        topic = self.knowledge_graph.topics.get(topic_id)
        if not topic:
            return None
        
        features = [
            topic.difficulty_level,
            topic.estimated_hours,
            len(topic.skills_covered),
            len(topic.prerequisites),
            topic.popularity_score,
            topic.industry_relevance,
            len(topic.tags),
            len(topic.learning_outcomes)
        ]
        
        # Add category encoding
        categories = ['programming', 'data', 'ai_ml', 'web', 'computer_science', 'security', 'infrastructure']
        category_vector = [1.0 if topic.category.lower() == cat else 0.0 for cat in categories]
        
        features.extend(category_vector)
        return np.array(features)
    
    def _extract_learning_context_features(self, event: LearningEvent) -> Optional[np.ndarray]:
        """Extract features for learning context analysis"""
        try:
            features = [
                event.difficulty_rating,
                (event.end_time - event.start_time).total_seconds() / 3600,  # Duration in hours
                event.completion_percentage,
                event.engagement_score,
                1.0 if event.learning_style == 'visual' else 0.0,
                1.0 if event.learning_style == 'auditory' else 0.0,
                1.0 if event.learning_style == 'kinesthetic' else 0.0,
                event.session_quality
            ]
            
            # Add time-based features
            hour = event.start_time.hour
            day_of_week = event.start_time.weekday()
            
            features.extend([
                hour / 24.0,  # Normalized hour
                day_of_week / 7.0,  # Normalized day
                1.0 if hour < 12 else 0.0,  # Morning
                1.0 if 12 <= hour < 18 else 0.0,  # Afternoon
                1.0 if hour >= 18 else 0.0  # Evening
            ])
            
            return np.array(features)
        except:
            return None
    
    def _create_learning_context(self, user_id: str, topic_id: str, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create learning context features for adaptive scoring"""
        user_profile = self.user_model.users.get(user_id)
        topic = self.knowledge_graph.topics.get(topic_id)
        
        if not user_profile or not topic:
            return None
        
        # User features
        patterns = user_profile['learning_patterns']
        metrics = user_profile['behavioral_metrics']
        
        features = [
            patterns['completion_rate'],
            patterns['average_session_duration'] / 60.0,
            patterns['learning_velocity'],
            metrics['consistency_score'],
            metrics['engagement_score'],
            metrics['adaptability_score']
        ]
        
        # Topic features
        features.extend([
            topic.difficulty_level,
            topic.estimated_hours,
            len(topic.skills_covered),
            topic.popularity_score
        ])
        
        # Context features
        current_hour = datetime.now().hour
        features.extend([
            current_hour / 24.0,
            1.0 if current_hour < 12 else 0.0,  # Morning
            1.0 if 12 <= current_hour < 18 else 0.0,  # Afternoon
            1.0 if current_hour >= 18 else 0.0  # Evening
        ])
        
        return np.array(features)
    
    def _get_candidate_topics(self, user_id: str, context: Dict[str, Any]) -> List[str]:
        """Get candidate topics for recommendation"""
        user_profile = self.user_model.users.get(user_id)
        if not user_profile:
            return []
        
        # Get user's current skills and interests
        user_skills = list(user_profile.get('skill_progress', {}).keys())
        interests = user_profile['preferences'].interests
        
        # Use knowledge graph to find suitable topics
        recommendations = self.knowledge_graph.get_topic_recommendations(
            user_skills, interests, user_profile['preferences'].difficulty_preference
        )
        
        # Return topic IDs
        return [rec[0] for rec in recommendations]
    
    def update_feedback(self, user_id: str, topic_id: str, rating: float, feedback: str = ""):
        """Update models based on user feedback"""
        
        # Store feedback
        feedback_entry = {
            'user_id': user_id,
            'topic_id': topic_id,
            'rating': rating,
            'feedback': feedback,
            'timestamp': datetime.now()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update user model
        self.user_model.update_user_feedback(user_id, topic_id, rating, feedback)
        
        # Retrain models periodically
        if len(self.feedback_history) % 100 == 0:  # Retrain every 100 feedback entries
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain models with new feedback data"""
        print("Retraining models with new feedback...")
        
        # Prepare training data from feedback history
        training_data = []
        for feedback in self.feedback_history:
            training_data.append({
                'user_id': feedback['user_id'],
                'topic_id': feedback['topic_id'],
                'rating': feedback['rating']
            })
        
        # Retrain neural model
        if len(training_data) > 50:  # Need minimum data for training
            self.train_neural_recommender(training_data)
        
        # Update collaborative filter
        self.train_collaborative_filter(training_data)
        
        print("Models retrained successfully")
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics for all models"""
        return self.model_performance.copy()
    
    def save_models(self, directory: str):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        # Save user model
        self.user_model.save_model(os.path.join(directory, 'user_model.pkl'))
        
        # Save knowledge graph
        self.knowledge_graph.save_graph(os.path.join(directory, 'knowledge_graph.pkl'))
        
        # Save ML models
        if self.neural_recommender:
            self.neural_recommender.save(os.path.join(directory, 'neural_recommender.h5'))
        
        if self.adaptive_learning_model:
            joblib.dump(self.adaptive_learning_model, os.path.join(directory, 'adaptive_model.pkl'))
        
        # Save other models
        model_data = {
            'collaborative_filter': self.collaborative_filter,
            'content_based_filter': self.content_based_filter,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'feedback_history': self.feedback_history
        }
        
        with open(os.path.join(directory, 'recommender_models.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load all trained models"""
        
        # Load user model
        self.user_model.load_model(os.path.join(directory, 'user_model.pkl'))
        
        # Load knowledge graph
        self.knowledge_graph.load_graph(os.path.join(directory, 'knowledge_graph.pkl'))
        
        # Load ML models
        neural_path = os.path.join(directory, 'neural_recommender.h5')
        if os.path.exists(neural_path):
            self.neural_recommender = keras.models.load_model(neural_path)
        
        adaptive_path = os.path.join(directory, 'adaptive_model.pkl')
        if os.path.exists(adaptive_path):
            self.adaptive_learning_model = joblib.load(adaptive_path)
        
        # Load other models
        models_path = os.path.join(directory, 'recommender_models.pkl')
        if os.path.exists(models_path):
            with open(models_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.collaborative_filter = model_data.get('collaborative_filter')
            self.content_based_filter = model_data.get('content_based_filter')
            self.model_weights = model_data.get('model_weights', self.model_weights)
            self.model_performance = model_data.get('model_performance', {})
            self.feedback_history = model_data.get('feedback_history', [])
        
        print(f"Models loaded from {directory}")
