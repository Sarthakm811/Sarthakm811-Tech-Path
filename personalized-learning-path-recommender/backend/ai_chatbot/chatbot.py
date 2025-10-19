"""
Advanced AI Chatbot for Learning Path Recommender
Provides personalized learning guidance, assistance, and intelligent tutoring
"""

import openai
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..database.models import User, Topic, LearningSession
from ..database.database import get_database_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message in the conversation"""
    message_id: str
    user_id: str
    message_type: str  # 'user', 'bot', 'system'
    content: str
    timestamp: datetime
    context: Dict[str, Any] = None
    intent: str = None
    entities: Dict[str, Any] = None
    confidence: float = 0.0

@dataclass
class LearningContext:
    """Context about user's learning journey"""
    current_topics: List[str]
    completed_topics: List[str]
    learning_goals: List[str]
    interests: List[str]
    current_difficulty: str
    learning_style: str
    progress_percentage: float
    recent_sessions: List[Dict[str, Any]]

class AIChatbot:
    """
    Advanced AI-powered chatbot for personalized learning assistance
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the AI chatbot
        
        Args:
            openai_api_key: OpenAI API key for GPT integration
        """
        self.openai_api_key = openai_api_key or "your-openai-api-key"
        openai.api_key = self.openai_api_key
        
        # Chatbot personality and capabilities
        self.personality = {
            'name': 'EduBot',
            'role': 'Personal Learning Assistant',
            'tone': 'encouraging and supportive',
            'expertise': 'learning guidance, study tips, motivation'
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'help': ['help', 'assistance', 'support', 'guidance'],
            'progress': ['progress', 'how am i doing', 'my progress', 'learning progress'],
            'recommendation': ['recommend', 'suggest', 'what should i learn', 'next topic'],
            'difficulty': ['too hard', 'too easy', 'difficult', 'challenging', 'boring'],
            'motivation': ['motivate', 'encourage', 'inspired', 'stuck', 'frustrated'],
            'schedule': ['schedule', 'time', 'when', 'how long', 'timeline'],
            'resources': ['resources', 'materials', 'books', 'videos', 'tutorials'],
            'career': ['career', 'job', 'profession', 'future', 'goals'],
            'technical': ['code', 'programming', 'error', 'bug', 'implementation']
        }
        
        # Response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm EduBot, your personal learning assistant. How can I help you on your learning journey today?",
                "Hi there! Ready to continue your learning adventure? What would you like to work on?",
                "Welcome back! I'm here to support your learning goals. What can I help you with?"
            ],
            'help': [
                "I'm here to help you with your learning journey! I can assist with recommendations, progress tracking, study tips, and motivation. What would you like to know?",
                "I can help you with learning recommendations, track your progress, provide study guidance, and keep you motivated. What specific area would you like assistance with?"
            ],
            'encouragement': [
                "You're doing great! Learning is a journey, and every step forward is progress. Keep up the excellent work!",
                "Remember, every expert was once a beginner. You're making fantastic progress on your learning path!",
                "I believe in you! You have the ability to master any skill with dedication and practice."
            ]
        }
        
        # Learning guidance knowledge base
        self.learning_guidance = {
            'study_tips': {
                'visual': [
                    "Use diagrams and flowcharts to visualize concepts",
                    "Create mind maps to organize information",
                    "Watch video tutorials and demonstrations",
                    "Use color coding in your notes"
                ],
                'auditory': [
                    "Listen to lectures and podcasts",
                    "Explain concepts out loud to yourself",
                    "Join study groups for discussions",
                    "Use mnemonic devices and rhymes"
                ],
                'kinesthetic': [
                    "Practice with hands-on projects",
                    "Take breaks to move around while studying",
                    "Build physical models or prototypes",
                    "Use interactive coding exercises"
                ],
                'reading': [
                    "Read comprehensive documentation",
                    "Take detailed written notes",
                    "Create summaries and outlines",
                    "Read multiple sources on the same topic"
                ]
            },
            'motivation_quotes': [
                "The expert in anything was once a beginner. - Helen Hayes",
                "Learning never exhausts the mind. - Leonardo da Vinci",
                "Education is the most powerful weapon which you can use to change the world. - Nelson Mandela",
                "The capacity to learn is a gift; the ability to learn is a skill; the willingness to learn is a choice. - Brian Herbert"
            ]
        }
        
        # Initialize TF-IDF for intent classification
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.intent_classifier = None
        
    async def process_message(self, user_id: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user message and generate an intelligent response
        
        Args:
            user_id: ID of the user sending the message
            message: The user's message content
            context: Additional context about the conversation
            
        Returns:
            Dictionary containing the bot's response and metadata
        """
        try:
            # Get user context
            user_context = await self._get_user_context(user_id)
            
            # Classify intent
            intent = self._classify_intent(message)
            
            # Extract entities
            entities = self._extract_entities(message, user_context)
            
            # Generate response based on intent and context
            response = await self._generate_response(
                message, intent, entities, user_context, context
            )
            
            # Store conversation in database
            await self._store_conversation(user_id, message, response, intent, entities)
            
            return {
                'response': response['content'],
                'intent': intent,
                'entities': entities,
                'confidence': response['confidence'],
                'suggestions': response.get('suggestions', []),
                'context': response.get('context', {})
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'response': "I apologize, but I'm having trouble processing your message right now. Please try again later.",
                'intent': 'error',
                'entities': {},
                'confidence': 0.0,
                'suggestions': [],
                'context': {}
            }
    
    async def _get_user_context(self, user_id: str) -> LearningContext:
        """Get comprehensive context about the user's learning journey"""
        try:
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                # Get user data
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return LearningContext([], [], [], [], 'medium', 'visual', 0.0, [])
                
                # Get current topic progress
                topic_progress = session.query(TopicProgress).filter(
                    TopicProgress.user_id == user_id,
                    TopicProgress.status == 'in_progress'
                ).all()
                current_topics = [tp.topic_id for tp in topic_progress]
                
                # Get completed topics
                completed_progress = session.query(TopicProgress).filter(
                    TopicProgress.user_id == user_id,
                    TopicProgress.status == 'completed'
                ).all()
                completed_topics = [tp.topic_id for tp in completed_progress]
                
                # Get recent learning sessions
                recent_sessions = session.query(LearningSession).filter(
                    LearningSession.user_id == user_id
                ).order_by(LearningSession.start_time.desc()).limit(5).all()
                
                recent_sessions_data = []
                for session_obj in recent_sessions:
                    recent_sessions_data.append({
                        'topic_id': session_obj.topic_id,
                        'start_time': session_obj.start_time,
                        'duration_minutes': session_obj.duration_minutes,
                        'completion_percentage': session_obj.completion_percentage,
                        'engagement_score': session_obj.engagement_score
                    })
                
                return LearningContext(
                    current_topics=current_topics,
                    completed_topics=completed_topics,
                    learning_goals=user.learning_goals,
                    interests=user.interests,
                    current_difficulty=user.difficulty_preference,
                    learning_style=user.preferred_learning_style,
                    progress_percentage=len(completed_topics) / max(len(completed_topics) + len(current_topics), 1),
                    recent_sessions=recent_sessions_data
                )
                
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return LearningContext([], [], [], [], 'medium', 'visual', 0.0, [])
    
    def _classify_intent(self, message: str) -> str:
        """Classify the user's intent from their message"""
        message_lower = message.lower()
        
        # Simple keyword-based classification
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _extract_entities(self, message: str, context: LearningContext) -> Dict[str, Any]:
        """Extract relevant entities from the user's message"""
        entities = {}
        message_lower = message.lower()
        
        # Extract topic mentions
        topic_mentions = []
        for topic in context.current_topics + context.completed_topics:
            if topic.lower() in message_lower:
                topic_mentions.append(topic)
        
        if topic_mentions:
            entities['topics'] = topic_mentions
        
        # Extract difficulty mentions
        difficulty_words = ['easy', 'hard', 'difficult', 'challenging', 'simple', 'complex']
        for word in difficulty_words:
            if word in message_lower:
                entities['difficulty'] = word
                break
        
        # Extract time references
        time_words = ['today', 'yesterday', 'tomorrow', 'week', 'month', 'hour', 'minute']
        for word in time_words:
            if word in message_lower:
                entities['time_reference'] = word
                break
        
        return entities
    
    async def _generate_response(self, message: str, intent: str, entities: Dict[str, Any], 
                               user_context: LearningContext, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an intelligent response based on intent and context"""
        
        if intent == 'greeting':
            return {
                'content': np.random.choice(self.response_templates['greeting']),
                'confidence': 0.9,
                'suggestions': ['Show my progress', 'Recommend next topic', 'Help with current topic']
            }
        
        elif intent == 'help':
            return {
                'content': np.random.choice(self.response_templates['help']),
                'confidence': 0.8,
                'suggestions': ['Learning recommendations', 'Progress tracking', 'Study tips', 'Motivation']
            }
        
        elif intent == 'progress':
            return await self._handle_progress_inquiry(user_context)
        
        elif intent == 'recommendation':
            return await self._handle_recommendation_request(user_context, entities)
        
        elif intent == 'difficulty':
            return await self._handle_difficulty_feedback(user_context, entities)
        
        elif intent == 'motivation':
            return await self._handle_motivation_request(user_context)
        
        elif intent == 'technical':
            return await self._handle_technical_question(message, user_context)
        
        else:
            # Use OpenAI for general responses
            return await self._generate_openai_response(message, user_context, intent)
    
    async def _handle_progress_inquiry(self, user_context: LearningContext) -> Dict[str, Any]:
        """Handle progress-related inquiries"""
        completed_count = len(user_context.completed_topics)
        current_count = len(user_context.current_topics)
        
        if completed_count > 0:
            response = f"Great progress! You've completed {completed_count} topics and are currently working on {current_count} topics. "
            response += f"Your overall progress is {user_context.progress_percentage:.1%}. "
            
            if user_context.recent_sessions:
                recent_session = user_context.recent_sessions[0]
                response += f"Your most recent session was {recent_session['duration_minutes']} minutes with {recent_session['engagement_score']:.1%} engagement."
        else:
            response = "You're just getting started on your learning journey! I'm excited to help you progress through your first topics."
        
        return {
            'content': response,
            'confidence': 0.8,
            'suggestions': ['Show detailed progress', 'Recommend next topic', 'Set learning goals']
        }
    
    async def _handle_recommendation_request(self, user_context: LearningContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recommendation requests"""
        if not user_context.current_topics and not user_context.completed_topics:
            # New user - recommend based on interests
            if 'data_science' in user_context.interests:
                response = "Since you're interested in data science, I recommend starting with 'Python Programming Fundamentals' and then 'Introduction to Data Science'. These will give you a solid foundation!"
            elif 'web_development' in user_context.interests:
                response = "For web development, I suggest starting with 'Web Development Fundamentals' to learn HTML, CSS, and JavaScript basics."
            else:
                response = "Let's start with 'Python Programming Fundamentals' - it's a great foundation for many career paths!"
        else:
            # Continuing user - recommend next logical step
            if user_context.current_topics:
                response = f"You're currently working on {len(user_context.current_topics)} topics. "
                response += "Focus on completing your current topics before starting new ones. Would you like help with any specific topic?"
            else:
                response = "You've completed some great topics! Based on your progress, I recommend advancing to more challenging topics in your area of interest."
        
        return {
            'content': response,
            'confidence': 0.7,
            'suggestions': ['Show learning path', 'Start specific topic', 'Adjust difficulty level']
        }
    
    async def _handle_difficulty_feedback(self, user_context: LearningContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle difficulty-related feedback"""
        if 'difficulty' in entities:
            difficulty = entities['difficulty']
            
            if difficulty in ['hard', 'difficult', 'challenging']:
                response = "I understand you're finding the material challenging. That's actually a good sign - it means you're pushing yourself to grow! "
                response += "Here are some strategies that might help:\n"
                response += "• Break the topic into smaller chunks\n"
                response += "• Practice with simpler examples first\n"
                response += "• Take breaks to let concepts sink in\n"
                response += "• Ask for help when you need it"
                
                return {
                    'content': response,
                    'confidence': 0.8,
                    'suggestions': ['Adjust difficulty', 'Get study tips', 'Find easier topics']
                }
            
            elif difficulty in ['easy', 'simple']:
                response = "If you're finding the material too easy, that's great! It means you're ready for more challenging content. "
                response += "I can recommend more advanced topics or help you find additional challenges in your current topic."
                
                return {
                    'content': response,
                    'confidence': 0.8,
                    'suggestions': ['Show advanced topics', 'Increase difficulty', 'Find challenges']
                }
        
        # General difficulty feedback
        response = "I'm here to help you find the right difficulty level for your learning. "
        response += "Remember, the best learning happens when you're slightly challenged but not overwhelmed."
        
        return {
            'content': response,
            'confidence': 0.6,
            'suggestions': ['Adjust difficulty', 'Get personalized recommendations', 'Track progress']
        }
    
    async def _handle_motivation_request(self, user_context: LearningContext) -> Dict[str, Any]:
        """Handle motivation and encouragement requests"""
        quote = np.random.choice(self.learning_guidance['motivation_quotes'])
        
        if user_context.progress_percentage > 0.5:
            response = f"You're doing amazing! {quote} "
            response += "You've already completed over half of your learning journey. Keep up the fantastic work!"
        elif user_context.progress_percentage > 0.2:
            response = f"You're making great progress! {quote} "
            response += "Every expert was once a beginner, and you're well on your way to becoming an expert yourself!"
        else:
            response = f"Starting is often the hardest part, and you've already taken that step! {quote} "
            response += "Remember, every small step forward is progress. You've got this!"
        
        return {
            'content': response,
            'confidence': 0.9,
            'suggestions': ['Show progress', 'Set goals', 'Find study buddy']
        }
    
    async def _handle_technical_question(self, message: str, user_context: LearningContext) -> Dict[str, Any]:
        """Handle technical questions using OpenAI"""
        # Create a focused prompt for technical assistance
        prompt = f"""You are EduBot, a helpful learning assistant. A student is asking a technical question about programming or learning.

Student's question: {message}

Student's context:
- Current topics: {user_context.current_topics}
- Learning style: {user_context.learning_style}
- Difficulty preference: {user_context.current_difficulty}

Please provide a helpful, encouraging response that:
1. Directly answers their question
2. Explains concepts clearly for their level
3. Provides practical examples if appropriate
4. Encourages continued learning
5. Keeps the response concise but comprehensive

Response:"""
        
        try:
            response = await self._call_openai(prompt)
            return {
                'content': response,
                'confidence': 0.8,
                'suggestions': ['Ask follow-up question', 'Show related topics', 'Get more examples']
            }
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return {
                'content': "I'd love to help with your technical question! While I'm having some technical difficulties right now, I encourage you to check the documentation or ask in our community forums. Don't give up - you're learning valuable skills!",
                'confidence': 0.5,
                'suggestions': ['Try again later', 'Check documentation', 'Ask community']
            }
    
    async def _generate_openai_response(self, message: str, user_context: LearningContext, intent: str) -> Dict[str, Any]:
        """Generate response using OpenAI GPT"""
        prompt = f"""You are EduBot, a friendly and encouraging personal learning assistant. A student is asking you something.

Student's message: {message}

Student's learning context:
- Current topics: {user_context.current_topics}
- Completed topics: {user_context.completed_topics}
- Learning goals: {user_context.learning_goals}
- Interests: {user_context.interests}
- Learning style: {user_context.learning_style}
- Progress: {user_context.progress_percentage:.1%}

Please provide a helpful, encouraging response that:
1. Directly addresses their question or concern
2. Is personalized to their learning context
3. Maintains an encouraging and supportive tone
4. Offers practical suggestions when appropriate
5. Keeps the response conversational and not too long

Response:"""
        
        try:
            response = await self._call_openai(prompt)
            return {
                'content': response,
                'confidence': 0.7,
                'suggestions': ['Ask follow-up question', 'Show progress', 'Get recommendations']
            }
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return {
                'content': "I'm here to help, but I'm experiencing some technical difficulties right now. Please try rephrasing your question or ask me about something else I can assist with!",
                'confidence': 0.3,
                'suggestions': ['Try again', 'Ask about progress', 'Get help']
            }
    
    async def _call_openai(self, prompt: str) -> str:
        """Make API call to OpenAI"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are EduBot, a helpful and encouraging learning assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _store_conversation(self, user_id: str, user_message: str, bot_response: Dict[str, Any], 
                                intent: str, entities: Dict[str, Any]):
        """Store conversation in database for learning and analytics"""
        try:
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                # Create conversation entries (you might want to create a Conversation model)
                # For now, we'll just log the interaction
                logger.info(f"Conversation stored - User: {user_id}, Intent: {intent}, Entities: {entities}")
                
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
    
    def get_study_tips(self, learning_style: str, topic: str = None) -> List[str]:
        """Get personalized study tips based on learning style"""
        tips = self.learning_guidance['study_tips'].get(learning_style, [])
        
        if topic:
            # Add topic-specific tips
            topic_tips = {
                'python': ['Practice coding daily', 'Use Python REPL for experimentation', 'Read other people\'s code'],
                'data_science': ['Work with real datasets', 'Practice statistical thinking', 'Visualize your data'],
                'web_development': ['Build projects regularly', 'Study existing websites', 'Practice responsive design']
            }
            tips.extend(topic_tips.get(topic.lower(), []))
        
        return tips
    
    async def generate_learning_plan(self, user_id: str, goals: List[str], time_available: int) -> Dict[str, Any]:
        """Generate a personalized learning plan"""
        user_context = await self._get_user_context(user_id)
        
        plan = {
            'weekly_schedule': self._create_weekly_schedule(time_available),
            'recommended_topics': await self._recommend_topics_for_goals(goals, user_context),
            'milestones': self._create_milestones(goals, time_available),
            'study_tips': self.get_study_tips(user_context.learning_style)
        }
        
        return plan
    
    def _create_weekly_schedule(self, time_available: int) -> Dict[str, Any]:
        """Create a weekly learning schedule"""
        sessions_per_week = max(1, time_available // 2)  # 2-hour sessions
        session_duration = min(120, time_available // sessions_per_week * 60)
        
        return {
            'sessions_per_week': sessions_per_week,
            'session_duration_minutes': session_duration,
            'total_weekly_hours': time_available,
            'recommended_days': ['Monday', 'Wednesday', 'Friday', 'Sunday'][:sessions_per_week]
        }
    
    async def _recommend_topics_for_goals(self, goals: List[str], user_context: LearningContext) -> List[str]:
        """Recommend topics based on learning goals"""
        # This would integrate with your recommendation engine
        recommendations = []
        
        for goal in goals:
            if 'data' in goal.lower() or 'analyst' in goal.lower():
                recommendations.extend(['python_fundamentals', 'data_science_intro', 'machine_learning_basics'])
            elif 'web' in goal.lower() or 'developer' in goal.lower():
                recommendations.extend(['web_development_basics', 'javascript', 'react_development'])
            elif 'ai' in goal.lower() or 'machine learning' in goal.lower():
                recommendations.extend(['python_fundamentals', 'data_science_intro', 'machine_learning_basics'])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_milestones(self, goals: List[str], time_available: int) -> List[Dict[str, Any]]:
        """Create learning milestones"""
        milestones = []
        
        for i, goal in enumerate(goals):
            milestones.append({
                'goal': goal,
                'target_date': (datetime.now() + timedelta(weeks=(i + 1) * 4)).strftime('%Y-%m-%d'),
                'estimated_hours': time_available * 4,  # 4 weeks
                'status': 'pending'
            })
        
        return milestones
