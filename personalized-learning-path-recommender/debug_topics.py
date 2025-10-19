#!/usr/bin/env python3
"""
Debug script to test SAMPLE_TOPICS data structure
"""

import json

# Test if we can import and process SAMPLE_TOPICS
try:
    from streamlit_app import SAMPLE_TOPICS
    print(f"✅ Successfully imported SAMPLE_TOPICS with {len(SAMPLE_TOPICS)} topics")
    
    # Test if we can process the data
    for i, topic in enumerate(SAMPLE_TOPICS[:3]):  # Test first 3 topics
        print(f"\n--- Topic {i+1} ---")
        print(f"ID: {topic.get('id', 'MISSING')}")
        print(f"Title: {topic.get('title', 'MISSING')}")
        print(f"Category: {topic.get('category', 'MISSING')}")
        print(f"Skills: {len(topic.get('skills_covered', []))} skills")
        
        # Check for any malformed data
        if not isinstance(topic, dict):
            print(f"❌ Topic {i} is not a dictionary!")
        
        required_fields = ['id', 'title', 'description', 'category']
        missing_fields = [field for field in required_fields if field not in topic]
        if missing_fields:
            print(f"❌ Topic {i} missing fields: {missing_fields}")
    
    print("\n✅ SAMPLE_TOPICS data structure is valid")
    
except Exception as e:
    print(f"❌ Error with SAMPLE_TOPICS: {e}")
    import traceback
    traceback.print_exc()