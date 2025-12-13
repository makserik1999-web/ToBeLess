import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { motion } from 'motion/react';
import { useState } from 'react';
import { Calendar, TrendingUp } from 'lucide-react';

export function Charts() {
  const [timeRange, setTimeRange] = useState('7d');

  // Sample data for activity over time
  const activityData = [
    { time: 'Mon', fights: 4, weapons: 1, screams: 8, crowds: 45 },
    { time: 'Tue', fights: 3, weapons: 2, screams: 5, crowds: 38 },
    { time: 'Wed', fights: 5, weapons: 0, screams: 12, crowds: 52 },
    { time: 'Thu', fights: 2, weapons: 1, screams: 6, crowds: 41 },
    { time: 'Fri', fights: 6, weapons: 3, screams: 15, crowds: 67 },
    { time: 'Sat', fights: 8, weapons: 2, screams: 18, crowds: 89 },
    { time: 'Sun', fights: 4, weapons: 1, screams: 9, crowds: 72 },
  ];

  const alertTrendsData = [
    { time: '00:00', alerts: 12 },
    { time: '04:00', alerts: 8 },
    { time: '08:00', alerts: 25 },
    { time: '12:00', alerts: 35 },
    { time: '16:00', alerts: 42 },
    { time: '20:00', alerts: 38 },
    { time: '23:59', alerts: 18 },
  ];

  const detectionTypeData = [
    { type: 'Crowds', count: 190 },
    { type: 'Screams', count: 34 },
    { type: 'Fights', count: 18 },
    { type: 'Weapons', count: 5 },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl p-4 shadow-2xl">
          <p className="text-purple-200 mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <section className="space-y-6">
      {/* Header */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <h2 className="text-2xl text-purple-100 mb-1">Analytics & Trends</h2>
          <p className="text-purple-300">Activity patterns and insights</p>
        </div>
        
        {/* Time Range Selector */}
        <div className="flex items-center gap-2 backdrop-blur-xl bg-white/5 rounded-xl p-1 border border-white/10">
          {['24h', '7d', '30d', '90d'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm transition-all ${
                timeRange === range
                  ? 'bg-purple-600 text-white'
                  : 'text-purple-300 hover:text-purple-200'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Activity Over Time Chart */}
        <motion.div
          className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg text-purple-100 mb-1">Detection Activity</h3>
              <p className="text-sm text-purple-400">Last 7 days breakdown</p>
            </div>
            <div className="w-10 h-10 backdrop-blur-lg bg-purple-500/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-purple-400" />
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={activityData}>
              <defs>
                <linearGradient id="colorFights" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorWeapons" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorScreens" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#eab308" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#eab308" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorCrowds" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="time" stroke="#c084fc" />
              <YAxis stroke="#c084fc" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ color: '#c084fc' }} />
              <Area type="monotone" dataKey="fights" stroke="#ef4444" fillOpacity={1} fill="url(#colorFights)" name="Fights" />
              <Area type="monotone" dataKey="weapons" stroke="#f97316" fillOpacity={1} fill="url(#colorWeapons)" name="Weapons" />
              <Area type="monotone" dataKey="screams" stroke="#eab308" fillOpacity={1} fill="url(#colorScreens)" name="Screams" />
              <Area type="monotone" dataKey="crowds" stroke="#3b82f6" fillOpacity={1} fill="url(#colorCrowds)" name="Crowds" />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Alert Trends Chart */}
        <motion.div
          className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg text-purple-100 mb-1">Alert Trends</h3>
              <p className="text-sm text-purple-400">24-hour pattern analysis</p>
            </div>
            <div className="w-10 h-10 backdrop-blur-lg bg-purple-500/20 rounded-lg flex items-center justify-center">
              <Calendar className="w-5 h-5 text-purple-400" />
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={alertTrendsData}>
              <defs>
                <linearGradient id="colorAlerts" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a855f7" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#a855f7" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="time" stroke="#c084fc" />
              <YAxis stroke="#c084fc" />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="alerts" 
                stroke="#a855f7" 
                strokeWidth={3}
                dot={{ fill: '#a855f7', r: 5 }}
                activeDot={{ r: 8 }}
                name="Alerts"
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Detection Type Distribution */}
        <motion.div
          className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10 lg:col-span-2"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg text-purple-100 mb-1">Detection Distribution</h3>
              <p className="text-sm text-purple-400">Breakdown by event type</p>
            </div>
          </div>
          
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={detectionTypeData}>
              <defs>
                <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a855f7" stopOpacity={0.9}/>
                  <stop offset="95%" stopColor="#7c3aed" stopOpacity={0.7}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="type" stroke="#c084fc" />
              <YAxis stroke="#c084fc" />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="count" 
                fill="url(#barGradient)" 
                radius={[10, 10, 0, 0]}
                name="Count"
              />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>
    </section>
  );
}
