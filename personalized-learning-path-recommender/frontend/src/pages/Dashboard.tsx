import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BookOpen, 
  TrendingUp, 
  Clock, 
  Target, 
  Brain, 
  Award,
  ChevronRight,
  Play,
  Star,
  Users,
  Calendar,
  BarChart3
} from 'lucide-react';
import { useQuery } from 'react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useAuth } from '../contexts/AuthContext';
import { getRecommendations, getUserProgress } from '../services/api';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import TopicCard from '../components/Learning/TopicCard';
import ProgressRing from '../components/UI/ProgressRing';
import AchievementBadge from '../components/UI/AchievementBadge';

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [selectedTimeframe, setSelectedTimeframe] = useState('week');

  // Fetch user data
  const { data: recommendations, isLoading: recommendationsLoading } = useQuery(
    'recommendations',
    () => getRecommendations(user?.id || '', {
      level: user?.preferred_learning_style || 'beginner',
      interests: user?.interests || [],
      goals: user?.learning_goals || [],
      time_per_week: user?.available_time_per_week || 10
    }),
    {
      enabled: !!user?.id,
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  const { data: progress, isLoading: progressLoading } = useQuery(
    'userProgress',
    () => getUserProgress(user?.id || ''),
    {
      enabled: !!user?.id,
    }
  );

  // Mock data for charts (replace with real data)
  const learningData = [
    { name: 'Mon', hours: 2.5 },
    { name: 'Tue', hours: 1.8 },
    { name: 'Wed', hours: 3.2 },
    { name: 'Thu', hours: 2.1 },
    { name: 'Fri', hours: 2.9 },
    { name: 'Sat', hours: 4.1 },
    { name: 'Sun', hours: 1.5 },
  ];

  const skillDistribution = [
    { name: 'Python', value: 45, color: '#3b82f6' },
    { name: 'Data Science', value: 30, color: '#10b981' },
    { name: 'Machine Learning', value: 15, color: '#f59e0b' },
    { name: 'Web Development', value: 10, color: '#ef4444' },
  ];

  const stats = [
    {
      title: 'Learning Streak',
      value: progress?.current_streak_days || 0,
      change: '+3',
      icon: <Calendar className="w-6 h-6" />,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      title: 'Hours This Week',
      value: progress?.weekly_hours || 0,
      change: '+2.5h',
      icon: <Clock className="w-6 h-6" />,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      title: 'Topics Completed',
      value: progress?.completed_topics || 0,
      change: '+1',
      icon: <BookOpen className="w-6 h-6" />,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      title: 'Skill Level',
      value: user?.difficulty_preference || 'Beginner',
      change: 'Improving',
      icon: <TrendingUp className="w-6 h-6" />,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
    },
  ];

  const achievements = [
    { id: 1, title: 'First Steps', description: 'Completed your first topic', earned: true },
    { id: 2, title: 'Week Warrior', description: '7-day learning streak', earned: true },
    { id: 3, title: 'Data Explorer', description: 'Completed data science track', earned: false },
    { id: 4, title: 'Python Master', description: 'Advanced Python certification', earned: false },
  ];

  if (recommendationsLoading || progressLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Welcome back, {user?.username || 'Learner'}! 👋
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Ready to continue your learning journey? Here's your personalized dashboard.
        </p>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
      >
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + index * 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-soft hover:shadow-medium transition-all duration-300"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                  {stat.title}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {stat.value}
                </p>
                <p className={`text-sm font-medium ${stat.color}`}>
                  {stat.change}
                </p>
              </div>
              <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                {stat.icon}
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Learning Progress */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 space-y-8"
        >
          {/* Weekly Learning Chart */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-soft">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Learning Activity
              </h2>
              <div className="flex space-x-2">
                {['week', 'month', 'year'].map((timeframe) => (
                  <button
                    key={timeframe}
                    onClick={() => setSelectedTimeframe(timeframe)}
                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                      selectedTimeframe === timeframe
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {timeframe.charAt(0).toUpperCase() + timeframe.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="hours"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Recommended Topics */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-soft">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Recommended for You
              </h2>
              <button className="text-primary-600 hover:text-primary-700 font-medium flex items-center">
                View All <ChevronRight className="w-4 h-4 ml-1" />
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations?.recommendations?.slice(0, 4).map((topic: any, index: number) => (
                <motion.div
                  key={topic.topic_id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                >
                  <TopicCard topic={topic} />
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Sidebar */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-8"
        >
          {/* Overall Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-soft">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Overall Progress
            </h2>
            <div className="flex items-center justify-center mb-4">
              <ProgressRing
                progress={progress?.overall_progress || 0}
                size={120}
                strokeWidth={8}
              />
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {Math.round(progress?.overall_progress || 0)}%
              </p>
              <p className="text-gray-600 dark:text-gray-300">
                Complete
              </p>
            </div>
          </div>

          {/* Skill Distribution */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-soft">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Skill Distribution
            </h2>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={skillDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {skillDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 space-y-2">
              {skillDistribution.map((skill) => (
                <div key={skill.name} className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: skill.color }}
                    />
                    <span className="text-sm text-gray-600 dark:text-gray-300">
                      {skill.name}
                    </span>
                  </div>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {skill.value}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Achievements */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-soft">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Achievements
            </h2>
            <div className="space-y-4">
              {achievements.map((achievement) => (
                <AchievementBadge
                  key={achievement.id}
                  achievement={achievement}
                />
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
