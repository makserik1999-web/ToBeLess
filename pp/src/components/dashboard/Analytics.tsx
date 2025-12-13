import { TrendingUp, Calendar, Download, Filter, Eye, AlertTriangle, Users, Shield } from 'lucide-react';
import { useState } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';

export function Analytics() {
  const { theme } = useTheme();
  const [timeRange, setTimeRange] = useState('30d');
  const [selectedMetric, setSelectedMetric] = useState('all');

  const detectionTrends = [
    { month: 'Jan', fights: 45, weapons: 12, screams: 67, crowds: 234 },
    { month: 'Feb', fights: 52, weapons: 15, screams: 71, crowds: 256 },
    { month: 'Mar', fights: 48, weapons: 10, screams: 65, crowds: 289 },
    { month: 'Apr', fights: 61, weapons: 18, screams: 83, crowds: 312 },
    { month: 'May', fights: 55, weapons: 14, screams: 76, crowds: 298 },
    { month: 'Jun', fights: 49, weapons: 11, screams: 69, crowds: 276 },
  ];

  const responseTimeData = [
    { hour: '00:00', avgTime: 2.1 },
    { hour: '04:00', avgTime: 1.8 },
    { hour: '08:00', avgTime: 2.5 },
    { hour: '12:00', avgTime: 3.2 },
    { hour: '16:00', avgTime: 2.9 },
    { hour: '20:00', avgTime: 2.4 },
  ];

  const detectionDistribution = [
    { name: 'Crowds', value: 1254, color: '#3b82f6' },
    { name: 'Screams', value: 431, color: '#eab308' },
    { name: 'Fights', value: 310, color: '#ef4444' },
    { name: 'Weapons', value: 80, color: '#f97316' },
  ];

  const locationAnalytics = [
    { location: 'Building A', incidents: 156, severity: 'high' },
    { location: 'Building B', incidents: 98, severity: 'medium' },
    { location: 'Building C', incidents: 134, severity: 'high' },
    { location: 'Parking Lot', incidents: 67, severity: 'low' },
    { location: 'Main Entrance', incidents: 189, severity: 'medium' },
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className={`border rounded-xl p-3 shadow-2xl ${
          theme === 'light'
            ? 'bg-white border-purple-200'
            : 'bg-zinc-950 border-zinc-800'
        }`}>
          <p className={`text-xs mb-1 font-medium ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-400'}`}>
            {payload[0].payload.month || payload[0].payload.hour || payload[0].payload.name}
          </p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
              {entry.name}: <span>{entry.value}</span>
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Analytics & Insights
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Comprehensive security analytics and trends
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className={`flex items-center gap-2 px-4 py-2 border rounded-xl transition-colors ${
            theme === 'light'
              ? 'bg-white border-purple-200 text-zinc-700 hover:bg-purple-50'
              : 'bg-zinc-900 border-zinc-800 text-zinc-300 hover:bg-zinc-800'
          }`}>
            <Filter className="w-4 h-4" />
            <span className="text-sm font-medium">Filter</span>
          </button>
          <button className={`flex items-center gap-2 px-4 py-2 border rounded-xl transition-colors ${
            theme === 'light'
              ? 'bg-white border-purple-200 text-zinc-700 hover:bg-purple-50'
              : 'bg-zinc-900 border-zinc-800 text-zinc-300 hover:bg-zinc-800'
          }`}>
            <Download className="w-4 h-4" />
            <span className="text-sm font-medium">Export</span>
          </button>
        </div>
      </div>

      {/* Time Range Selector */}
      <div className="flex gap-2">
        {['7d', '30d', '90d', '1y'].map((range) => (
          <button
            key={range}
            onClick={() => setTimeRange(range)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              timeRange === range
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                : theme === 'light'
                ? 'bg-white border border-purple-200 text-zinc-700 hover:bg-purple-50'
                : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:bg-zinc-800'
            }`}
          >
            {range === '7d' ? 'Last 7 Days' : range === '30d' ? 'Last 30 Days' : range === '90d' ? 'Last 90 Days' : 'Last Year'}
          </button>
        ))}
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { icon: Eye, label: 'Total Detections', value: '2,075', change: '+15.3%', trend: 'up' },
          { icon: TrendingUp, label: 'Avg Response Time', value: '2.4m', change: '-12%', trend: 'down' },
          { icon: AlertTriangle, label: 'Critical Alerts', value: '23', change: '-8%', trend: 'down' },
          { icon: Shield, label: 'Prevention Rate', value: '94.5%', change: '+2.1%', trend: 'up' },
        ].map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`backdrop-blur-xl border rounded-2xl p-6 ${
              theme === 'light'
                ? 'bg-white border-purple-200'
                : 'bg-zinc-950 border-zinc-900'
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                theme === 'light' ? 'bg-purple-100' : 'bg-zinc-900'
              }`}>
                <metric.icon className={`w-6 h-6 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
              </div>
              <span className={`text-xs px-2 py-1 rounded-lg font-medium ${
                metric.trend === 'up' ? 'bg-emerald-500/10 text-emerald-600' : 'bg-emerald-500/10 text-emerald-600'
              }`}>
                {metric.change}
              </span>
            </div>
            <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
              {metric.value}
            </div>
            <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
              {metric.label}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Detection Trends */}
        <div className={`lg:col-span-2 backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <h3 className={`font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Detection Trends
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={detectionTrends}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme === 'light' ? '#e9d5ff' : '#27272a'} />
              <XAxis dataKey="month" stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <YAxis stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line type="monotone" dataKey="fights" stroke="#ef4444" strokeWidth={2} name="Fights" />
              <Line type="monotone" dataKey="weapons" stroke="#f97316" strokeWidth={2} name="Weapons" />
              <Line type="monotone" dataKey="screams" stroke="#eab308" strokeWidth={2} name="Screams" />
              <Line type="monotone" dataKey="crowds" stroke="#3b82f6" strokeWidth={2} name="Crowds" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Distribution Pie Chart */}
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <h3 className={`font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Detection Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={detectionDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {detectionDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Response Time Chart */}
        <div className={`lg:col-span-2 backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <h3 className={`font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Average Response Time (minutes)
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme === 'light' ? '#e9d5ff' : '#27272a'} />
              <XAxis dataKey="hour" stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <YAxis stroke={theme === 'light' ? '#71717a' : '#71717a'} style={{ fontSize: '12px', fontWeight: 500 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="avgTime" fill="#9333ea" radius={[8, 8, 0, 0]} name="Avg Time (min)" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Location Analytics */}
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <h3 className={`font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Incidents by Location
          </h3>
          <div className="space-y-4">
            {locationAnalytics.map((location, index) => (
              <div key={location.location} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                    {location.location}
                  </span>
                  <span className={`text-sm font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                    {location.incidents}
                  </span>
                </div>
                <div className="relative w-full h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`absolute top-0 left-0 h-full rounded-full ${
                      location.severity === 'high' ? 'bg-red-500' :
                      location.severity === 'medium' ? 'bg-yellow-500' :
                      'bg-emerald-500'
                    }`}
                    style={{ width: `${(location.incidents / 200) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
