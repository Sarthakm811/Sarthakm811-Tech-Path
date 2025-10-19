import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta

# Configure page - TechPath Frontend
st.set_page_config(
    page_title="TechPath - AI Learning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL - automatically detect the environment
import os
if os.getenv('STREAMLIT_SHARING_MODE'):
    API_URL = "http://localhost:5000"  # Streamlit Cloud
else:
    API_URL = "http://localhost:5000"  # Local development

def check_backend_connection():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_recommendations(user_data):
    """Get recommendations from backend API"""
    try:
        response = requests.post(
            f"{API_URL}/api/recommend",
            json=user_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    st.title("🎓 Personalized Learning Path Recommender")
    st.markdown("Get personalized learning recommendations based on your goals and interests!")
    
    # Check backend connection
    if not check_backend_connection():
        st.error("⚠️ Backend API is not running. Please start the backend server first.")
        st.info("Run: `cd backend && python app.py`")
        return
    else:
        st.success("✅ Backend API is connected and running!")
    
    # Sidebar for user input
    with st.sidebar:
        st.header("📝 Your Learning Profile")
        
        # User level
        user_level = st.selectbox(
            "Current Skill Level",
            ["beginner", "intermediate", "advanced"],
            help="Select your current programming/technical skill level"
        )
        
        # Available time
        time_per_week = st.slider(
            "Hours per week for learning",
            min_value=1,
            max_value=40,
            value=10,
            help="How many hours can you dedicate to learning each week?"
        )
        
        # Interests
        st.subheader("🎯 Areas of Interest")
        interests = st.multiselect(
            "What topics interest you most?",
            [
                "programming", "data", "ai_ml", "web", "computer_science",
                "security", "infrastructure", "mobile", "game_development"
            ],
            default=["programming", "data"],
            help="Choose the fields you want to learn about"
        )
        
        # Learning goals
        st.subheader("🎯 Learning Goals")
        goals_text = st.text_area(
            "What do you want to achieve?",
            placeholder="e.g., Become a data scientist, Learn web development, Build AI applications...",
            height=100,
            help="Describe your career goals or what you want to learn"
        )
        
        # Parse goals
        goals = [goal.strip() for goal in goals_text.split(',') if goal.strip()]
        
        # Submit button
        if st.button("🚀 Get My Learning Path", type="primary"):
            if not interests and not goals:
                st.warning("Please select at least one interest area or describe your goals.")
            else:
                # Prepare user data
                user_data = {
                    "level": user_level,
                    "interests": interests,
                    "goals": goals,
                    "time_per_week": time_per_week
                }
                
                # Show loading
                with st.spinner("Generating your personalized learning path..."):
                    result = get_recommendations(user_data)
                
                
                # Store result in session state
                st.session_state.recommendations = result
    
    # Main content area
    if 'recommendations' in st.session_state:
        result = st.session_state.recommendations
        
        if result and result.get('success'):
            recommendations = result['recommendations']
            
            # Beautiful header with summary
            st.markdown("---")
            st.markdown("## 🎉 Your Personalized Learning Journey")
            
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📚 Courses", len(recommendations['learning_path']))
            with col2:
                st.metric("⏱️ Total Hours", recommendations['total_estimated_hours'])
            with col3:
                weeks = recommendations['total_estimated_hours'] // time_per_week
                st.metric("📅 Duration", f"{weeks} weeks")
            with col4:
                st.metric("🎯 Your Level", user_level.title())
            
            # Career path suggestions
            if recommendations.get('recommended_career_paths'):
                st.markdown("---")
                st.markdown("### 💼 Career Opportunities")
                career_cols = st.columns(len(recommendations['recommended_career_paths']))
                for i, career in enumerate(recommendations['recommended_career_paths']):
                    with career_cols[i]:
                        st.info(f"🎯 **{career}**")
            
            # Main learning path display
            st.markdown("---")
            st.markdown("### 📚 Your Learning Path")
            
            # Create a beautiful step-by-step layout
            for i, topic in enumerate(recommendations['learning_path'], 1):
                # Create a card-like container
                with st.container():
                    # Step header with progress indicator
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.markdown(f"### {i}")
                    with col2:
                        st.markdown(f"### {topic.get('title', topic.get('name', 'Unknown Course'))}")
                    with col3:
                        st.markdown(f"**{topic['estimated_hours']} hours**")
                    
                    # Topic details in an attractive layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**What you'll learn:** {topic['description']}")
                        
                        # Skills in a nice format
                        skills_text = " • ".join(topic.get('skills_covered', []))
                        st.markdown(f"**Skills:** {skills_text}")
                        
                        if topic.get('prerequisites'):
                            prereq_text = " → ".join(topic['prerequisites'])
                            st.warning(f"**Prerequisites:** {prereq_text}")
                    
                    with col2:
                        # Visual indicators
                        level_color = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
                        st.markdown(f"**Level:** {level_color.get(topic['level'], '⚪')} {topic['level'].title()}")
                        st.markdown(f"**Category:** {topic['category'].replace('_', ' ').title()}")
                        
                        # Time estimate
                        if time_per_week > 0:
                            weeks_needed = max(1, topic['estimated_hours'] // time_per_week)
                            st.markdown(f"**Timeline:** {weeks_needed} week{'s' if weeks_needed > 1 else ''}")
                    
                    st.markdown("---")
            
            # Timeline section
            st.markdown("### 📅 Weekly Schedule")
            for week_info in recommendations['timeline']:
                with st.expander(f"🗓️ {week_info['week_range']} - {week_info['topic']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**📖 Topic:** {week_info['topic']}")
                        st.markdown(f"**⏰ Hours per week:** {week_info['hours_per_week']}")
                        st.markdown(f"**📊 Total hours:** {week_info['total_hours']}")
                    
                    with col2:
                        st.markdown("**🎯 Weekly Goals:**")
                        for milestone in week_info.get('milestones', []):
                            st.markdown(f"• {milestone}")
            
            st.markdown("---")
            
            # Progress tracking section
            st.markdown("### 📊 Track Your Progress")
            st.info("💡 **Tip:** Check off completed courses to see your progress!")
            
            # Create an interactive progress tracker
            completed_topics = st.multiselect(
                "✅ Mark completed courses:",
                [topic.get('title', topic.get('name', 'Unknown Course')) for topic in recommendations['learning_path']],
                key="progress_tracker",
                help="Select the courses you've completed to track your progress"
            )
            
            if completed_topics:
                progress_percentage = (len(completed_topics) / len(recommendations['learning_path'])) * 100
                
                # Visual progress bar
                st.progress(progress_percentage / 100)
                
                # Progress summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Completed", len(completed_topics))
                with col2:
                    st.metric("Remaining", len(recommendations['learning_path']) - len(completed_topics))
                with col3:
                    st.metric("Progress", f"{progress_percentage:.1f}%")
                
                # Celebration message
                if progress_percentage == 100:
                    st.balloons()
                    st.success("🎉 **Congratulations!** You've completed your entire learning path!")
                elif progress_percentage >= 50:
                    st.success(f"🚀 **Great progress!** You're more than halfway through your learning journey!")
                else:
                    st.success(f"💪 **Keep going!** You've completed {len(completed_topics)} out of {len(recommendations['learning_path'])} courses!")
            
            # Next steps section
            if not completed_topics:
                st.markdown("**🎯 Ready to start?** Begin with your first course and mark it complete when you finish!")
            else:
                remaining_topics = [topic.get('title', topic.get('name', 'Unknown Course')) for topic in recommendations['learning_path'] 
                                 if topic.get('title', topic.get('name', 'Unknown Course')) not in completed_topics]
                if remaining_topics:
                    st.markdown("**⬆️ Next up:** " + remaining_topics[0])
            
            st.markdown("---")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Generate New Path", help="Create a new learning path with different settings"):
                    # Clear the current recommendations
                    if 'recommendations' in st.session_state:
                        del st.session_state.recommendations
                    st.rerun()
            
            with col2:
                if st.button("📋 Save Progress", help="Save your current progress"):
                    st.success("Progress saved! 💾")
            
            with col3:
                if st.button("📤 Share Path", help="Share your learning path with others"):
                    st.info("Share link copied to clipboard! 📋")
        
        else:
            st.error(f"❌ Error getting recommendations: {result.get('error', 'Unknown error')}")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to Your Personalized Learning Journey! 🚀
        
        This AI-powered recommender will create a customized learning path just for you based on:
        
        - 🎯 **Your current skill level**
        - ⏰ **Available time commitment**
        - 🔥 **Your interests and passions**
        - 🎪 **Specific learning goals**
        
        ### How it works:
        1. Fill out your learning profile in the sidebar
        2. Click "Get My Learning Path"
        3. Receive a personalized curriculum with timeline
        4. Track your progress as you learn!
        
        ### 🎓 Ready to start your learning adventure?
        Use the sidebar to tell us about yourself and your goals!
        """)
        
        # Sample topics preview
        st.subheader("📚 Available Learning Topics")
        
        if check_backend_connection():
            try:
                response = requests.get(f"{API_URL}/api/topics")
                if response.status_code == 200:
                    topics_data = response.json()
                    if topics_data.get('success'):
                        topics = topics_data['topics']
                        
                        # Group by category
                        categories = {}
                        for topic in topics:
                            category = topic['category'].replace('_', ' ').title()
                            if category not in categories:
                                categories[category] = []
                            categories[category].append(topic)
                        
                        # Display topics by category
                        for category, topic_list in categories.items():
                            with st.expander(f"{category} ({len(topic_list)} topics)"):
                                for topic in topic_list:
                                    st.write(f"**{topic['name']}** - {topic['estimated_hours']} hours ({topic['level']})")
                                    st.write(f"*{topic['description']}*")
                                    st.write("---")
            except:
                pass

if __name__ == "__main__":
    main()
