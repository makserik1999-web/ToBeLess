import { Shield, Swords, Volume2, Users, TrendingUp, TrendingDown, Activity, Clock, MapPin, AlertTriangle } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';
import { useTheme } from '../Dashboard';

export function Overview() {
  const { theme } = useTheme();
  const [timeRange, setTimeRange] = useState('7d');

  const stats = [
    { icon: Shield, label: 'Total Detections', value: '2,847', change: '+12.5%', trend: 'up', color: 'purple' },
    { icon: Swords, label: 'Active Threats', value: '3', change: '-25%', trend: 'down', color: 'red' },
    { icon: Activity, label: 'Avg Response Time', value: '1.8m', change: '-15%', trend: 'down', color: 'blue' },
    { icon: Users, label: 'Monitored Zones', value: '24', change: '+8%', trend: 'up', color: 'emerald' },
  ];

  const activityData = [
    { time: '00:00', value: 12 },
    { time: '04:00', value: 8 },
    { time: '08:00', value: 25 },
    { time: '12:00', value: 35 },
    { time: '16:00', value: 42 },
    { time: '20:00', value: 31 },
    { time: '23:59', value: 18 },
  ];

  const weeklyData = [
    { day: 'Mon', fights: 4, weapons: 1, screams: 8, crowds: 45 },
    { day: 'Tue', fights: 3, weapons: 2, screams: 5, crowds: 38 },
    { day: 'Wed', fights: 5, weapons: 0, screams: 12, crowds: 52 },
    { day: 'Thu', fights: 2, weapons: 1, screams: 6, crowds: 41 },
    { day: 'Fri', fights: 6, weapons: 3, screams: 15, crowds: 67 },
    { day: 'Sat', fights: 8, weapons: 2, screams: 18, crowds: 89 },
    { day: 'Sun', fights: 4, weapons: 1, screams: 9, crowds: 72 },
  ];

  const recentIncidents = [
    { id: 1, type: 'Fight', location: 'Building A - East Wing', time: '3 min ago', severity: 'high', status: 'active' },
    { id: 2, type: 'Crowd', location: 'Building B - Lobby', time: '12 min ago', severity: 'medium', status: 'monitoring' },
    { id: 3, type: 'Scream', location: 'Parking Lot C', time: '28 min ago', severity: 'medium', status: 'resolved' },
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.15 }}
          className={`border rounded-xl p-3 shadow-2xl ${
          theme === 'light'
            ? 'bg-white border-purple-200'
            : 'bg-zinc-950 border-zinc-800'
        }`}>
          <p className={`text-xs mb-1 font-medium ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-400'}`}>
            {payload[0].payload.time || payload[0].payload.day}
          </p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
              {entry.name}: <span>{entry.value}</span>
            </p>
          ))}
        </motion.div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div>
          <motion.h1 
            className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            Security Overview
          </motion.h1>
          <motion.p 
            className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            Real-time monitoring across all locations
          </motion.p>
        </div>
        <motion.div 
          className="flex items-center gap-2"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          {['24h', '7d', '30d'].map((range, idx) => (
            <motion.button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                timeRange === range
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                  : theme === 'light'
                  ? 'bg-white border border-purple-200 text-zinc-700 hover:text-zinc-900 hover:bg-purple-50'
                  : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.2 + idx * 0.05 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {range}
            </motion.button>
          ))}
        </motion.div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 30, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: index * 0.08, duration: 0.5, type: "spring", stiffness: 100 }}
            whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.2 } }}
            className={`backdrop-blur-xl border rounded-2xl p-6 transition-all group cursor-pointer ${
              theme === 'light'
                ? 'bg-white border-purple-200 hover:shadow-xl hover:shadow-purple-100 hover:border-purple-300'
                : 'bg-zinc-950 border-zinc-900 hover:border-zinc-700 hover:shadow-xl hover:shadow-purple-900/20'
            }`}
          >
            <div className="flex items-start justify-between mb-4">
              <motion.div 
                className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                  theme === 'light' ? 'bg-purple-100' : 'bg-zinc-900'
                }`}
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
              >
                <stat.icon className={`w-6 h-6 ${
                  theme === 'light' ? 'text-purple-600' : 'text-purple-500'
                }`} strokeWidth={2} />
              </motion.div>
              <motion.div 
                className={`flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg font-medium ${
                  stat.trend === 'up' ? 'bg-emerald-500/10 text-emerald-600' : 'bg-red-500/10 text-red-600'
                }`}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.08 + 0.2 }}
              >
                <motion.div
                  animate={{ y: stat.trend === 'up' ? [-1, 1, -1] : [1, -1, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  {stat.trend === 'up' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                </motion.div>
                {stat.change}
              </motion.div>
            </div>
            <motion.div 
              className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.08 + 0.3, type: "spring", stiffness: 200 }}
            >
              {stat.value}
            </motion.div>
            <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
              {stat.label}
            </div>
            
            {/* Subtle animated underline */}
            <motion.div
              className="h-1 bg-gradient-to-r from-purple-500 to-purple-600 rounded-full mt-4 opacity-0 group-hover:opacity-100"
              initial={{ scaleX: 0 }}
              whileHover={{ scaleX: 1 }}
              transition={{ duration: 0.3 }}
              style={{ transformOrigin: 'left' }}
            />
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Activity Chart */}
        <motion.div 
          className={`backdrop-blur-xl border rounded-2xl p-8 ${
            theme === 'light'
              ? 'bg-white border-purple-200'
              : 'bg-zinc-950 border-zinc-900'
          }`}
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          whileHover={{ scale: 1.01 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                Detection Activity
              </h3>
              <p className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                Last 24 hours
              </p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={activityData}>
              <defs>
                <linearGradient id="activityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#9333ea" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#9333ea" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={theme === 'light' ? '#e9d5ff' : '#27272a'} />
              <XAxis dataKey="time" stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <YAxis stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="value" stroke="#9333ea" strokeWidth={2} fill="url(#activityGradient)" name="Detections" />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Weekly Breakdown */}
        <motion.div 
          className={`backdrop-blur-xl border rounded-2xl p-8 ${
            theme === 'light'
              ? 'bg-white border-purple-200'
              : 'bg-zinc-950 border-zinc-900'
          }`}
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          whileHover={{ scale: 1.01 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                Weekly Breakdown
              </h3>
              <p className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                Detection types by day
              </p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme === 'light' ? '#e9d5ff' : '#27272a'} />
              <XAxis dataKey="day" stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <YAxis stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="fights" fill="#ef4444" radius={[6, 6, 0, 0]} name="Fights" />
              <Bar dataKey="weapons" fill="#f97316" radius={[6, 6, 0, 0]} name="Weapons" />
              <Bar dataKey="screams" fill="#eab308" radius={[6, 6, 0, 0]} name="Screams" />
              <Bar dataKey="crowds" fill="#3b82f6" radius={[6, 6, 0, 0]} name="Crowds" />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Recent Incidents */}
      <motion.div 
        className={`backdrop-blur-xl border rounded-2xl p-8 ${
          theme === 'light'
            ? 'bg-white border-purple-200'
            : 'bg-zinc-950 border-zinc-900'
        }`}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.6 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Recent Incidents
          </h3>
          <motion.button 
            className="text-sm text-purple-600 hover:text-purple-700 font-medium transition-colors"
            whileHover={{ scale: 1.05, x: 3 }}
            whileTap={{ scale: 0.95 }}
          >
            View All
          </motion.button>
        </div>
        <div className="space-y-3">
          <AnimatePresence>
            {recentIncidents.map((incident, index) => (
              <motion.div
                key={incident.id}
                initial={{ opacity: 0, x: -30, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 30, scale: 0.95 }}
                transition={{ delay: index * 0.1, type: "spring", stiffness: 100 }}
                whileHover={{ scale: 1.02, x: 5, transition: { duration: 0.2 } }}
                className={`flex items-center gap-4 p-5 rounded-xl border transition-all cursor-pointer group ${
                  theme === 'light'
                    ? 'bg-purple-50/50 hover:bg-purple-100/50 border-purple-200 hover:border-purple-300 hover:shadow-lg hover:shadow-purple-100'
                    : 'bg-zinc-900/50 hover:bg-zinc-800/80 border-zinc-800 hover:border-zinc-700 hover:shadow-lg hover:shadow-purple-900/20'
                }`}
              >
                <motion.div 
                  className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    incident.severity === 'high'
                      ? 'bg-red-500/10'
                      : 'bg-yellow-500/10'
                  }`}
                  whileHover={{ rotate: [0, -10, 10, -10, 0], scale: 1.1 }}
                  transition={{ duration: 0.5 }}
                >
                  <AlertTriangle className={`w-6 h-6 ${
                    incident.severity === 'high' ? 'text-red-500' : 'text-yellow-500'
                  }`} />
                </motion.div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                      {incident.type}
                    </span>
                    <motion.span 
                      className={`px-2 py-0.5 rounded-md text-xs font-medium ${
                        incident.status === 'active' ? 'bg-red-500/20 text-red-600' :
                        incident.status === 'monitoring' ? 'bg-yellow-500/20 text-yellow-600' :
                        'bg-emerald-500/20 text-emerald-600'
                      }`}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 + 0.2 }}
                    >
                      {incident.status}
                    </motion.span>
                  </div>
                  <div className={`flex items-center gap-4 text-sm font-medium ${
                    theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                  }`}>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-3.5 h-3.5" />
                      {incident.location}
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      {incident.time}
                    </div>
                  </div>
                </div>
                <motion.button 
                  className="opacity-0 group-hover:opacity-100 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-xl text-sm font-medium transition-all shadow-lg shadow-purple-500/25"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Review
                </motion.button>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}