import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const apiEndpoints = {
  // Auth
  login: '/api/auth/login',
  register: '/api/auth/register',
  logout: '/api/auth/logout',
  
  // User
  profile: '/api/user/profile',
  progress: (userId: string) => `/api/user/${userId}/progress`,
  
  // Recommendations
  recommend: '/api/recommend',
  topics: '/api/topics',
  
  // Chat
  chat: '/api/chat',
  
  // Analytics
  analytics: '/api/analytics',
  
  // Feedback
  feedback: (userId: string) => `/api/user/${userId}/feedback`,
  
  // Health
  health: '/api/health',
};

// Auth API
export const authAPI = {
  login: async (credentials: { email: string; password: string }) => {
    const response = await api.post(apiEndpoints.login, credentials);
    return response.data;
  },
  
  register: async (userData: { 
    email: string; 
    password: string; 
    username: string; 
    fullName: string;
  }) => {
    const response = await api.post(apiEndpoints.register, userData);
    return response.data;
  },
  
  logout: async () => {
    const response = await api.post(apiEndpoints.logout);
    return response.data;
  },
};

// User API
export const userAPI = {
  getProfile: async (userId: string) => {
    const response = await api.get(`${apiEndpoints.profile}/${userId}`);
    return response.data;
  },
  
  updateProfile: async (userId: string, profileData: any) => {
    const response = await api.put(`${apiEndpoints.profile}/${userId}`, profileData);
    return response.data;
  },
  
  getProgress: async (userId: string) => {
    const response = await api.get(apiEndpoints.progress(userId));
    return response.data;
  },
};

// Recommendations API
export const recommendationsAPI = {
  getRecommendations: async (userId: string, userData: any) => {
    const response = await api.post(apiEndpoints.recommend, {
      user_id: userId,
      ...userData,
    });
    return response.data;
  },
  
  getTopics: async () => {
    const response = await api.get(apiEndpoints.topics);
    return response.data;
  },
  
  getTopicDetails: async (topicId: string) => {
    const response = await api.get(`${apiEndpoints.topics}/${topicId}`);
    return response.data;
  },
};

// Chat API
export const chatAPI = {
  sendMessage: async (userId: string, message: string) => {
    const response = await api.post(apiEndpoints.chat, {
      user_id: userId,
      message: message,
    });
    return response.data;
  },
};

// Analytics API
export const analyticsAPI = {
  getSystemAnalytics: async () => {
    const response = await api.get(apiEndpoints.analytics);
    return response.data;
  },
  
  getUserAnalytics: async (userId: string) => {
    const response = await api.get(`/api/analytics/user/${userId}`);
    return response.data;
  },
};

// Feedback API
export const feedbackAPI = {
  submitFeedback: async (userId: string, feedbackData: {
    topic_id: string;
    rating: number;
    feedback?: string;
  }) => {
    const response = await api.post(apiEndpoints.feedback(userId), feedbackData);
    return response.data;
  },
  
  getFeedback: async (userId: string) => {
    const response = await api.get(apiEndpoints.feedback(userId));
    return response.data;
  },
};

// Learning Sessions API
export const learningSessionsAPI = {
  startSession: async (userId: string, topicId: string) => {
    const response = await api.post('/api/learning-sessions/start', {
      user_id: userId,
      topic_id: topicId,
    });
    return response.data;
  },
  
  endSession: async (sessionId: string, sessionData: {
    completion_percentage: number;
    engagement_score: number;
    difficulty_rating?: number;
    session_notes?: string;
  }) => {
    const response = await api.post(`/api/learning-sessions/${sessionId}/end`, sessionData);
    return response.data;
  },
  
  getSessions: async (userId: string, limit = 20) => {
    const response = await api.get(`/api/learning-sessions/user/${userId}?limit=${limit}`);
    return response.data;
  },
};

// Social Features API
export const socialAPI = {
  getConnections: async (userId: string) => {
    const response = await api.get(`/api/social/connections/${userId}`);
    return response.data;
  },
  
  connectUser: async (userId: string, targetUserId: string) => {
    const response = await api.post('/api/social/connect', {
      user_id: userId,
      target_user_id: targetUserId,
    });
    return response.data;
  },
  
  getStudyGroups: async (userId: string) => {
    const response = await api.get(`/api/social/study-groups/${userId}`);
    return response.data;
  },
  
  createStudyGroup: async (groupData: {
    name: string;
    description?: string;
    topic_id?: string;
    max_members?: number;
  }) => {
    const response = await api.post('/api/social/study-groups', groupData);
    return response.data;
  },
};

// Utility functions
export const checkHealth = async () => {
  try {
    const response = await api.get(apiEndpoints.health);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    return null;
  }
};

// Export convenience functions
export const getRecommendations = recommendationsAPI.getRecommendations;
export const getTopics = recommendationsAPI.getTopics;
export const sendChatMessage = chatAPI.sendMessage;
export const getUserProgress = userAPI.getProgress;
export const submitFeedback = feedbackAPI.submitFeedback;

export default api;
