"""
Advanced Knowledge Graph System for Learning Path Recommender
Implements sophisticated topic relationships, skill dependencies, and learning prerequisites
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
import pickle
import os

@dataclass
class Skill:
    """Represents a skill in the knowledge graph"""
    skill_id: str
    name: str
    description: str
    category: str
    difficulty_level: int  # 1-5 scale
    estimated_hours: int
    prerequisites: List[str]
    learning_objectives: List[str]
    assessment_criteria: List[str]

@dataclass
class Topic:
    """Represents a learning topic with enhanced metadata"""
    topic_id: str
    title: str
    description: str
    category: str
    subcategory: str
    difficulty_level: int  # 1-5 scale
    estimated_hours: int
    skills_covered: List[str]
    prerequisites: List[str]
    learning_outcomes: List[str]
    assessment_methods: List[str]
    resources: List[Dict[str, str]]
    tags: List[str]
    popularity_score: float
    industry_relevance: float

class KnowledgeGraph:
    """
    Advanced knowledge graph system that models the relationships between
    learning topics, skills, and competencies using graph theory
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.skills = {}
        self.topics = {}
        self.skill_to_topics = {}  # Maps skills to topics that teach them
        self.topic_to_skills = {}  # Maps topics to skills they teach
        
    def add_skill(self, skill: Skill):
        """Add a skill to the knowledge graph"""
        self.skills[skill.skill_id] = skill
        self.graph.add_node(skill.skill_id, 
                           type='skill',
                           name=skill.name,
                           difficulty=skill.difficulty_level,
                           category=skill.category)
        
        # Add prerequisite relationships
        for prereq in skill.prerequisites:
            if prereq in self.skills:
                self.graph.add_edge(prereq, skill.skill_id, 
                                  relationship='prerequisite',
                                  weight=1.0)
    
    def add_topic(self, topic: Topic):
        """Add a topic to the knowledge graph"""
        self.topics[topic.topic_id] = topic
        self.graph.add_node(topic.topic_id,
                           type='topic',
                           name=topic.title,
                           difficulty=topic.difficulty_level,
                           category=topic.category,
                           hours=topic.estimated_hours)
        
        # Map skills to topics
        for skill_id in topic.skills_covered:
            if skill_id not in self.skill_to_topics:
                self.skill_to_topics[skill_id] = []
            self.skill_to_topics[skill_id].append(topic.topic_id)
            
            if topic.topic_id not in self.topic_to_skills:
                self.topic_to_skills[topic.topic_id] = []
            self.topic_to_skills[topic.topic_id].append(skill_id)
        
        # Add prerequisite relationships between topics
        for prereq in topic.prerequisites:
            if prereq in self.topics:
                self.graph.add_edge(prereq, topic.topic_id,
                                  relationship='prerequisite',
                                  weight=1.0)
    
    def find_learning_path(self, start_skills: List[str], target_skills: List[str], 
                          max_depth: int = 10) -> List[str]:
        """Find optimal learning path from start skills to target skills"""
        
        # Create subgraph with only relevant nodes
        relevant_nodes = set(start_skills + target_skills)
        
        # Expand to include prerequisites and skills taught by topics
        for _ in range(max_depth):
            new_nodes = set()
            for node in relevant_nodes:
                if node in self.graph:
                    # Add predecessors (prerequisites)
                    new_nodes.update(self.graph.predecessors(node))
                    # Add successors (what this enables)
                    new_nodes.update(self.graph.successors(node))
            relevant_nodes.update(new_nodes)
            if len(new_nodes) == 0:
                break
        
        # Create subgraph
        subgraph = self.graph.subgraph(relevant_nodes)
        
        # Find shortest paths from start to target skills
        paths = []
        for start_skill in start_skills:
            for target_skill in target_skills:
                try:
                    path = nx.shortest_path(subgraph, start_skill, target_skill)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        if not paths:
            return []
        
        # Return the shortest path
        shortest_path = min(paths, key=len)
        return shortest_path
    
    def get_topic_recommendations(self, user_skills: List[str], 
                                 interests: List[str],
                                 difficulty_preference: str = 'medium') -> List[Tuple[str, float]]:
        """Get topic recommendations based on user skills and interests"""
        
        recommendations = []
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        target_difficulty = difficulty_map.get(difficulty_preference, 2)
        
        for topic_id, topic in self.topics.items():
            # Check if user has prerequisites
            has_prerequisites = all(prereq in user_skills for prereq in topic.prerequisites)
            
            if not has_prerequisites:
                continue
            
            # Calculate recommendation score
            score = 0.0
            
            # Interest matching
            for interest in interests:
                if (interest.lower() in topic.category.lower() or
                    interest.lower() in topic.title.lower() or
                    any(interest.lower() in tag.lower() for tag in topic.tags)):
                    score += 0.3
            
            # Difficulty preference
            difficulty_diff = abs(topic.difficulty_level - target_difficulty)
            difficulty_score = max(0, 1.0 - difficulty_diff * 0.2)
            score += difficulty_score * 0.2
            
            # Skill gap analysis
            new_skills = [skill for skill in topic.skills_covered if skill not in user_skills]
            skill_gap_score = len(new_skills) / max(len(topic.skills_covered), 1)
            score += skill_gap_score * 0.3
            
            # Popularity and relevance
            score += topic.popularity_score * 0.1
            score += topic.industry_relevance * 0.1
            
            recommendations.append((topic_id, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def analyze_skill_gaps(self, user_skills: List[str], 
                          target_role: str) -> Dict[str, Any]:
        """Analyze skill gaps for a target role"""
        
        # Define role-specific skill requirements (this could be loaded from a database)
        role_requirements = {
            'data_scientist': ['python', 'statistics', 'machine_learning', 'data_analysis', 'sql'],
            'web_developer': ['html', 'css', 'javascript', 'python', 'databases'],
            'ai_engineer': ['python', 'machine_learning', 'deep_learning', 'neural_networks', 'tensorflow'],
            'cybersecurity_analyst': ['networking', 'security', 'python', 'linux', 'cryptography']
        }
        
        if target_role not in role_requirements:
            return {'error': 'Target role not found'}
        
        required_skills = role_requirements[target_role]
        user_skill_set = set(user_skills)
        required_skill_set = set(required_skills)
        
        # Find missing skills
        missing_skills = required_skill_set - user_skill_set
        
        # Find learning paths for missing skills
        learning_paths = {}
        for missing_skill in missing_skills:
            if missing_skill in self.skills:
                # Find topics that teach this skill
                topics = self.skill_to_topics.get(missing_skill, [])
                if topics:
                    # Find the best topic (lowest difficulty, shortest path)
                    best_topic = None
                    min_difficulty = float('inf')
                    
                    for topic_id in topics:
                        topic = self.topics[topic_id]
                        if topic.difficulty_level < min_difficulty:
                            min_difficulty = topic.difficulty_level
                            best_topic = topic_id
                    
                    if best_topic:
                        # Find learning path to this topic
                        path = self.find_learning_path(user_skills, [missing_skill])
                        learning_paths[missing_skill] = {
                            'topic': best_topic,
                            'path': path,
                            'difficulty': min_difficulty,
                            'hours': self.topics[best_topic].estimated_hours
                        }
        
        # Calculate overall readiness
        readiness_score = len(user_skill_set & required_skill_set) / len(required_skill_set)
        
        return {
            'target_role': target_role,
            'readiness_score': readiness_score,
            'missing_skills': list(missing_skills),
            'learning_paths': learning_paths,
            'estimated_hours_to_ready': sum(path['hours'] for path in learning_paths.values()),
            'recommended_learning_order': sorted(learning_paths.values(), 
                                               key=lambda x: (x['difficulty'], x['hours']))
        }
    
    def get_skill_relationships(self, skill_id: str) -> Dict[str, Any]:
        """Get detailed relationships for a specific skill"""
        if skill_id not in self.graph:
            return {}
        
        node_data = self.graph.nodes[skill_id]
        
        # Get prerequisites
        prerequisites = []
        for pred in self.graph.predecessors(skill_id):
            if self.graph[pred][skill_id].get('relationship') == 'prerequisite':
                prerequisites.append({
                    'skill_id': pred,
                    'name': self.graph.nodes[pred].get('name', pred),
                    'difficulty': self.graph.nodes[pred].get('difficulty', 1)
                })
        
        # Get skills this enables
        enabled_skills = []
        for succ in self.graph.successors(skill_id):
            if self.graph[skill_id][succ].get('relationship') == 'prerequisite':
                enabled_skills.append({
                    'skill_id': succ,
                    'name': self.graph.nodes[succ].get('name', succ),
                    'difficulty': self.graph.nodes[succ].get('difficulty', 1)
                })
        
        # Get topics that teach this skill
        teaching_topics = []
        for topic_id in self.skill_to_topics.get(skill_id, []):
            topic = self.topics[topic_id]
            teaching_topics.append({
                'topic_id': topic_id,
                'title': topic.title,
                'difficulty': topic.difficulty_level,
                'hours': topic.estimated_hours,
                'category': topic.category
            })
        
        return {
            'skill_id': skill_id,
            'name': node_data.get('name', skill_id),
            'difficulty': node_data.get('difficulty', 1),
            'category': node_data.get('category', ''),
            'prerequisites': prerequisites,
            'enabled_skills': enabled_skills,
            'teaching_topics': teaching_topics
        }
    
    def find_alternative_paths(self, start_skill: str, target_skill: str, 
                              max_paths: int = 5) -> List[List[str]]:
        """Find alternative learning paths between two skills"""
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(self.graph, start_skill, target_skill, cutoff=10))
            
            # Sort by length and difficulty
            scored_paths = []
            for path in paths:
                total_difficulty = sum(self.graph.nodes[node].get('difficulty', 1) for node in path)
                score = len(path) + total_difficulty * 0.1  # Prefer shorter, easier paths
                scored_paths.append((score, path))
            
            scored_paths.sort(key=lambda x: x[0])
            return [path for _, path in scored_paths[:max_paths]]
            
        except nx.NetworkXNoPath:
            return []
    
    def get_learning_community(self, skill_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get learning community around a skill (related skills and topics)"""
        if skill_id not in self.graph:
            return {}
        
        # Get nodes within specified depth
        community_nodes = set()
        current_level = {skill_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node in self.graph:
                    next_level.update(self.graph.neighbors(node))
            community_nodes.update(current_level)
            current_level = next_level
        
        # Categorize nodes
        skills = []
        topics = []
        
        for node in community_nodes:
            if self.graph.nodes[node].get('type') == 'skill':
                skills.append({
                    'skill_id': node,
                    'name': self.graph.nodes[node].get('name', node),
                    'difficulty': self.graph.nodes[node].get('difficulty', 1)
                })
            elif self.graph.nodes[node].get('type') == 'topic':
                topics.append({
                    'topic_id': node,
                    'name': self.graph.nodes[node].get('name', node),
                    'difficulty': self.graph.nodes[node].get('difficulty', 1),
                    'hours': self.graph.nodes[node].get('hours', 0)
                })
        
        return {
            'central_skill': skill_id,
            'skills': skills,
            'topics': topics,
            'community_size': len(community_nodes)
        }
    
    def calculate_learning_complexity(self, topic_sequence: List[str]) -> Dict[str, float]:
        """Calculate complexity metrics for a learning sequence"""
        if not topic_sequence:
            return {}
        
        total_hours = 0
        total_difficulty = 0
        complexity_score = 0.0
        dependency_depth = 0
        
        for i, topic_id in enumerate(topic_sequence):
            if topic_id in self.topics:
                topic = self.topics[topic_id]
                total_hours += topic.estimated_hours
                total_difficulty += topic.difficulty_level
                
                # Calculate dependency complexity
                prereq_count = len(topic.prerequisites)
                dependency_depth = max(dependency_depth, prereq_count)
                
                # Complexity increases with difficulty and prerequisites
                complexity_score += topic.difficulty_level * (1 + prereq_count * 0.1)
        
        # Normalize complexity score
        complexity_score = complexity_score / len(topic_sequence) if topic_sequence else 0
        
        return {
            'total_hours': total_hours,
            'average_difficulty': total_difficulty / len(topic_sequence),
            'complexity_score': complexity_score,
            'dependency_depth': dependency_depth,
            'learning_curve': 'gentle' if complexity_score < 2 else 'moderate' if complexity_score < 4 else 'steep'
        }
    
    def save_graph(self, filepath: str):
        """Save the knowledge graph to disk"""
        graph_data = {
            'graph': nx.node_link_data(self.graph),
            'skills': {k: {
                'skill_id': v.skill_id,
                'name': v.name,
                'description': v.description,
                'category': v.category,
                'difficulty_level': v.difficulty_level,
                'estimated_hours': v.estimated_hours,
                'prerequisites': v.prerequisites,
                'learning_objectives': v.learning_objectives,
                'assessment_criteria': v.assessment_criteria
            } for k, v in self.skills.items()},
            'topics': {k: {
                'topic_id': v.topic_id,
                'title': v.title,
                'description': v.description,
                'category': v.category,
                'subcategory': v.subcategory,
                'difficulty_level': v.difficulty_level,
                'estimated_hours': v.estimated_hours,
                'skills_covered': v.skills_covered,
                'prerequisites': v.prerequisites,
                'learning_outcomes': v.learning_outcomes,
                'assessment_methods': v.assessment_methods,
                'resources': v.resources,
                'tags': v.tags,
                'popularity_score': v.popularity_score,
                'industry_relevance': v.industry_relevance
            } for k, v in self.topics.items()},
            'skill_to_topics': self.skill_to_topics,
            'topic_to_skills': self.topic_to_skills
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_data, f)
    
    def load_graph(self, filepath: str):
        """Load the knowledge graph from disk"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Reconstruct graph
        self.graph = nx.node_link_graph(graph_data['graph'])
        
        # Reconstruct skills
        self.skills = {}
        for k, v in graph_data['skills'].items():
            self.skills[k] = Skill(**v)
        
        # Reconstruct topics
        self.topics = {}
        for k, v in graph_data['topics'].items():
            self.topics[k] = Topic(**v)
        
        # Restore mappings
        self.skill_to_topics = graph_data.get('skill_to_topics', {})
        self.topic_to_skills = graph_data.get('topic_to_skills', {})
