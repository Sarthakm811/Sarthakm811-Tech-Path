import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { authAPI, userAPI } from '../services/api';

interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  preferred_learning_style: string;
  difficulty_preference: string;
  available_time_per_week: number;
  learning_goals: string[];
  interests: string[];
  created_at: string;
  last_login: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (userData: Partial<User>) => Promise<void>;
}

interface RegisterData {
  email: string;
  password: string;
  username: string;
  fullName: string;
  learningStyle?: string;
  interests?: string[];
  goals?: string[];
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user;

  useEffect(() => {
    // Check for existing auth token on app load
    const token = localStorage.getItem('auth_token');
    const userId = localStorage.getItem('user_id');
    
    if (token && userId) {
      // Verify token and get user data
      userAPI.getProfile(userId)
        .then(response => {
          if (response.success) {
            setUser(response.user);
          } else {
            // Token is invalid, clear storage
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user_id');
          }
        })
        .catch(() => {
          // Token verification failed, clear storage
          localStorage.removeItem('auth_token');
          localStorage.removeItem('user_id');
        })
        .finally(() => {
          setIsLoading(false);
        });
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (email: string, password: string) => {
    try {
      setIsLoading(true);
      const response = await authAPI.login({ email, password });
      
      if (response.success) {
        const { user: userData, token } = response;
        
        // Store auth data
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_id', userData.id);
        
        // Set user in state
        setUser(userData);
      } else {
        throw new Error(response.error || 'Login failed');
      }
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterData) => {
    try {
      setIsLoading(true);
      const response = await authAPI.register({
        email: userData.email,
        password: userData.password,
        username: userData.username,
        fullName: userData.fullName,
      });
      
      if (response.success) {
        const { user: newUser, token } = response;
        
        // Store auth data
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_id', newUser.id);
        
        // Set user in state
        setUser(newUser);
        
        // Update user preferences if provided
        if (userData.learningStyle || userData.interests || userData.goals) {
          await updateUser({
            preferred_learning_style: userData.learningStyle || 'visual',
            interests: userData.interests || [],
            learning_goals: userData.goals || [],
          });
        }
      } else {
        throw new Error(response.error || 'Registration failed');
      }
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      // Call logout API
      await authAPI.logout();
    } catch (error) {
      console.error('Logout error:', error);
      // Continue with local logout even if API call fails
    } finally {
      // Clear local storage
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_id');
      
      // Clear user state
      setUser(null);
    }
  };

  const updateUser = async (userData: Partial<User>) => {
    if (!user) {
      throw new Error('No user logged in');
    }

    try {
      const response = await userAPI.updateProfile(user.id, userData);
      
      if (response.success) {
        // Update user in state
        setUser(prev => prev ? { ...prev, ...response.user } : null);
      } else {
        throw new Error(response.error || 'Profile update failed');
      }
    } catch (error) {
      console.error('Update user error:', error);
      throw error;
    }
  };

  const value: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    updateUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
