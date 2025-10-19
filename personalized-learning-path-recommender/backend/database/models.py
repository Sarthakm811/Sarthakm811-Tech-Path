"""
Database Models for Advanced Learning Path Recommender
Implements comprehensive data models for user profiles, learning history, and analytics
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """User profile model with comprehensive learning data"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Learning preferences
    preferred_learning_style = Column(String(50), default='visual')  # visual, auditory, kinesthetic, reading
    optimal_session_duration = Column(Integer, default=45)  # minutes
    preferred_time_of_day = Column(String(20), default='afternoon')  # morning, afternoon, evening
    difficulty_preference = Column(String(20), default='medium')  # easy, medium, hard
    available_time_per_week = Column(Integer, default=10)  # hours
    preferred_pace = Column(String(20), default='medium')  # slow, medium, fast
    
    # Learning goals and interests
    learning_goals = Column(JSON, default=list)
    interests = Column(JSON, default=list)
    target_career_paths = Column(JSON, default=list)
    
    # Progress tracking
    total_learning_hours = Column(Float, default=0.0)
    completed_topics_count = Column(Integer, default=0)
    current_streak_days = Column(Integer, default=0)
    longest_streak_days = Column(Integer, default=0)
    
    # Behavioral metrics
    consistency_score = Column(Float, default=0.0)
    persistence_score = Column(Float, default=0.0)
    curiosity_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    adaptability_score = Column(Float, default=0.0)
    
    # Relationships
    learning_sessions = relationship("LearningSession", back_populates="user")
    topic_progress = relationship("TopicProgress", back_populates="user")
    feedback_entries = relationship("Feedback", back_populates="user")
    social_connections = relationship("SocialConnection", foreign_keys="SocialConnection.user_id", back_populates="user")
    social_connections_received = relationship("SocialConnection", foreign_keys="SocialConnection.connected_user_id", back_populates="connected_user")

class Topic(Base):
    """Learning topic model with enhanced metadata"""
    __tablename__ = 'topics'
    
    id = Column(String(100), primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    subcategory = Column(String(100), nullable=True)
    difficulty_level = Column(Integer, nullable=False)  # 1-5 scale
    estimated_hours = Column(Integer, nullable=False)
    
    # Content metadata
    skills_covered = Column(JSON, default=list)
    prerequisites = Column(JSON, default=list)
    learning_outcomes = Column(JSON, default=list)
    assessment_methods = Column(JSON, default=list)
    resources = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    
    # Metrics
    popularity_score = Column(Float, default=0.0)
    industry_relevance = Column(Float, default=0.0)
    completion_rate = Column(Float, default=0.0)
    average_rating = Column(Float, default=0.0)
    total_enrollments = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    learning_sessions = relationship("LearningSession", back_populates="topic")
    topic_progress = relationship("TopicProgress", back_populates="topic")
    feedback_entries = relationship("Feedback", back_populates="topic")

class Skill(Base):
    """Skill model for knowledge graph"""
    __tablename__ = 'skills'
    
    id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    difficulty_level = Column(Integer, nullable=False)  # 1-5 scale
    estimated_hours = Column(Integer, nullable=False)
    
    # Relationships
    prerequisites = Column(JSON, default=list)
    learning_objectives = Column(JSON, default=list)
    assessment_criteria = Column(JSON, default=list)
    
    # Metrics
    mastery_threshold = Column(Float, default=0.8)
    industry_demand = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LearningSession(Base):
    """Learning session tracking with detailed analytics"""
    __tablename__ = 'learning_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    topic_id = Column(String(100), ForeignKey('topics.id'), nullable=False)
    
    # Session details
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    completion_percentage = Column(Float, default=0.0)
    
    # Learning analytics
    engagement_score = Column(Float, default=0.0)  # 0-1 scale
    difficulty_rating = Column(Integer, nullable=True)  # 1-5 scale (user reported)
    session_quality = Column(Float, default=0.0)  # 0-1 scale
    learning_style_used = Column(String(50), nullable=True)
    platform = Column(String(50), default='web')  # web, mobile, desktop
    
    # Progress tracking
    concepts_learned = Column(JSON, default=list)
    skills_practiced = Column(JSON, default=list)
    exercises_completed = Column(Integer, default=0)
    quiz_scores = Column(JSON, default=list)
    
    # Context
    session_notes = Column(Text, nullable=True)
    mood_before = Column(String(50), nullable=True)
    mood_after = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="learning_sessions")
    topic = relationship("Topic", back_populates="learning_sessions")

class TopicProgress(Base):
    """Detailed progress tracking for each topic"""
    __tablename__ = 'topic_progress'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    topic_id = Column(String(100), ForeignKey('topics.id'), nullable=False)
    
    # Progress metrics
    progress_percentage = Column(Float, default=0.0)
    time_spent_hours = Column(Float, default=0.0)
    sessions_completed = Column(Integer, default=0)
    
    # Status tracking
    status = Column(String(50), default='not_started')  # not_started, in_progress, completed, paused, abandoned
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, nullable=True)
    
    # Performance metrics
    average_session_rating = Column(Float, default=0.0)
    best_quiz_score = Column(Float, default=0.0)
    average_engagement = Column(Float, default=0.0)
    
    # Learning analytics
    difficulty_progression = Column(JSON, default=list)
    skill_mastery = Column(JSON, default=dict)
    learning_velocity = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="topic_progress")
    topic = relationship("Topic", back_populates="topic_progress")

class Feedback(Base):
    """User feedback and ratings system"""
    __tablename__ = 'feedback'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    topic_id = Column(String(100), ForeignKey('topics.id'), nullable=False)
    
    # Feedback content
    rating = Column(Float, nullable=False)  # 1-5 scale
    feedback_text = Column(Text, nullable=True)
    feedback_type = Column(String(50), default='general')  # general, difficulty, content, instructor
    
    # Detailed ratings
    content_quality = Column(Float, nullable=True)
    difficulty_appropriateness = Column(Float, nullable=True)
    engagement_level = Column(Float, nullable=True)
    practical_applicability = Column(Float, nullable=True)
    
    # Recommendations
    would_recommend = Column(Boolean, nullable=True)
    helpful_for_career = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="feedback_entries")
    topic = relationship("Topic", back_populates="feedback_entries")

class SocialConnection(Base):
    """Social learning connections and study groups"""
    __tablename__ = 'social_connections'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    connected_user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    connection_type = Column(String(50), default='peer')  # peer, mentor, mentee, study_buddy
    
    # Connection details
    status = Column(String(50), default='pending')  # pending, accepted, blocked
    created_at = Column(DateTime, default=datetime.utcnow)
    accepted_at = Column(DateTime, nullable=True)
    
    # Learning collaboration
    shared_topics = Column(JSON, default=list)
    collaborative_sessions = Column(Integer, default=0)
    mutual_goals = Column(JSON, default=list)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="social_connections")
    connected_user = relationship("User", foreign_keys=[connected_user_id], back_populates="social_connections_received")

class StudyGroup(Base):
    """Study groups and collaborative learning"""
    __tablename__ = 'study_groups'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    topic_id = Column(String(100), ForeignKey('topics.id'), nullable=True)
    
    # Group settings
    max_members = Column(Integer, default=10)
    is_public = Column(Boolean, default=True)
    requires_approval = Column(Boolean, default=False)
    
    # Group metrics
    member_count = Column(Integer, default=0)
    activity_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    topic = relationship("Topic")

class StudyGroupMember(Base):
    """Study group membership"""
    __tablename__ = 'study_group_members'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(UUID(as_uuid=True), ForeignKey('study_groups.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Membership details
    role = Column(String(50), default='member')  # member, moderator, admin
    joined_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Participation metrics
    sessions_attended = Column(Integer, default=0)
    contributions_count = Column(Integer, default=0)
    last_activity = Column(DateTime, nullable=True)

class Recommendation(Base):
    """AI-generated recommendations with tracking"""
    __tablename__ = 'recommendations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    topic_id = Column(String(100), ForeignKey('topics.id'), nullable=False)
    
    # Recommendation details
    recommendation_score = Column(Float, nullable=False)
    recommendation_type = Column(String(50), nullable=False)  # collaborative, content_based, neural, adaptive
    algorithm_version = Column(String(50), default='1.0')
    
    # User interaction
    was_viewed = Column(Boolean, default=False)
    was_clicked = Column(Boolean, default=False)
    was_enrolled = Column(Boolean, default=False)
    user_rating = Column(Float, nullable=True)
    
    # Context
    recommendation_context = Column(JSON, default=dict)
    generated_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")
    topic = relationship("Topic")

class LearningAnalytics(Base):
    """Aggregated learning analytics and insights"""
    __tablename__ = 'learning_analytics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(50), default='weekly')  # daily, weekly, monthly
    
    # Learning metrics
    total_learning_hours = Column(Float, default=0.0)
    sessions_completed = Column(Integer, default=0)
    topics_started = Column(Integer, default=0)
    topics_completed = Column(Integer, default=0)
    skills_learned = Column(Integer, default=0)
    
    # Performance metrics
    average_engagement = Column(Float, default=0.0)
    average_rating = Column(Float, default=0.0)
    completion_rate = Column(Float, default=0.0)
    learning_velocity = Column(Float, default=0.0)
    
    # Behavioral insights
    consistency_score = Column(Float, default=0.0)
    curiosity_index = Column(Float, default=0.0)
    difficulty_preference = Column(String(20), nullable=True)
    optimal_learning_times = Column(JSON, default=list)
    
    # Goal tracking
    goals_progress = Column(JSON, default=dict)
    milestones_achieved = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetrics(Base):
    """System-wide metrics and performance tracking"""
    __tablename__ = 'system_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # System performance
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # performance, usage, quality
    
    # Context
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    additional_data = Column(JSON, default=dict)
    
    recorded_at = Column(DateTime, default=datetime.utcnow)
