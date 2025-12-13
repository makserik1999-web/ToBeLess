import { Shield, Swords, Volume2, Users, Filter, Download, Search, MapPin, Clock } from 'lucide-react';
import { useState } from 'react';
import { motion } from 'motion/react';
import { useTheme } from '../Dashboard';

export function AlertsView() {
  const { theme } = useTheme();
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  const alerts = [
    { id: 1, type: 'fight', icon: Swords, title: 'Physical Altercation', location: 'Building A - East Wing, Floor 2', time: '2024-12-11 14:23', severity: 'critical', status: 'active', description: 'Multiple individuals engaged in physical conflict', assignee: 'John Doe' },
    { id: 2, type: 'weapon', icon: Shield, title: 'Weapon Detected', location: 'Building B - Parking Lot, Zone 3', time: '2024-12-11 13:45', severity: 'critical', status: 'investigating', description: 'Potential firearm identified by detection system', assignee: 'Jane Smith' },
    { id: 3, type: 'scream', icon: Volume2, title: 'Distress Vocalization', location: 'Building C - Floor 3, Room 301', time: '2024-12-11 12:18', severity: 'high', status: 'resolved', description: 'Audio analysis detected distress patterns', assignee: 'Mike Johnson' },
    { id: 4, type: 'crowd', icon: Users, title: 'High Density Event', location: 'Building A - Main Lobby', time: '2024-12-11 11:52', severity: 'medium', status: 'monitoring', description: 'Crowd density exceeded safety threshold', assignee: 'Sarah Chen' },
  ];

  const filters = [
    { id: 'all', label: 'All Alerts', count: alerts.length },
    { id: 'critical', label: 'Critical', count: alerts.filter(a => a.severity === 'critical').length },
    { id: 'active', label: 'Active', count: alerts.filter(a => a.status === 'active').length },
    { id: 'resolved', label: 'Resolved', count: alerts.filter(a => a.status === 'resolved').length },
  ];

  const severityColors = {
    critical: { bg: 'bg-red-500/10', text: 'text-red-600', border: 'border-red-500/30' },
    high: { bg: 'bg-orange-500/10', text: 'text-orange-600', border: 'border-orange-500/30' },
    medium: { bg: 'bg-yellow-500/10', text: 'text-yellow-600', border: 'border-yellow-500/30' },
    low: { bg: 'bg-blue-500/10', text: 'text-blue-600', border: 'border-blue-500/30' },
  };

  const statusColors = {
    active: 'bg-red-500/20 text-red-600',
    investigating: 'bg-orange-500/20 text-orange-600',
    monitoring: 'bg-blue-500/20 text-blue-600',
    resolved: 'bg-emerald-500/20 text-emerald-600',
  };

  const filteredAlerts = alerts.filter(alert => {
    if (selectedFilter === 'all') return true;
    if (selectedFilter === 'critical') return alert.severity === 'critical';
    if (selectedFilter === 'active') return alert.status === 'active';
    if (selectedFilter === 'resolved') return alert.status === 'resolved';
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-2xl mb-1 ${theme === 'light' ? 'text-purple-900' : 'text-purple-50'}`}>
            Security Alerts
          </h1>
          <p className={theme === 'light' ? 'text-purple-600' : 'text-purple-400'}>
            Manage and review all security incidents
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-colors ${
            theme === 'light'
              ? 'bg-purple-200/50 hover:bg-purple-300/50 border-purple-300/50 text-purple-800'
              : 'bg-purple-900/50 hover:bg-purple-800 border-purple-700/50 text-purple-200'
          }`}>
            <Filter className="w-4 h-4" />
            Filters
          </button>
          <button className={`flex items-center gap-2 px-4 py-2 border rounded-lg transition-colors ${
            theme === 'light'
              ? 'bg-purple-200/50 hover:bg-purple-300/50 border-purple-300/50 text-purple-800'
              : 'bg-purple-900/50 hover:bg-purple-800 border-purple-700/50 text-purple-200'
          }`}>
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="flex-1">
          <div className="relative">
            <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${
              theme === 'light' ? 'text-purple-600' : 'text-purple-400'
            }`} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by location, type, or description..."
              className={`w-full pl-12 pr-4 py-3 border rounded-xl focus:outline-none ${
                theme === 'light'
                  ? 'bg-purple-50 border-purple-300/50 text-purple-900 placeholder-purple-500 focus:border-purple-500'
                  : 'bg-purple-950/50 border-purple-800/50 text-purple-50 placeholder-purple-500 focus:border-purple-500/50'
              }`}
            />
          </div>
        </div>
        <div className="flex gap-2">
          {filters.map((filter) => (
            <button
              key={filter.id}
              onClick={() => setSelectedFilter(filter.id)}
              className={`px-4 py-3 rounded-xl text-sm transition-all whitespace-nowrap ${
                selectedFilter === filter.id
                  ? 'bg-purple-600 text-purple-50'
                  : theme === 'light'
                  ? 'bg-purple-50 border border-purple-300/50 text-purple-700 hover:text-purple-900 hover:border-purple-400'
                  : 'bg-purple-950/50 border border-purple-800/50 text-purple-300 hover:text-purple-100 hover:border-purple-700'
              }`}
            >
              {filter.label}
              <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                selectedFilter === filter.id 
                  ? 'bg-purple-700'
                  : theme === 'light'
                  ? 'bg-purple-200'
                  : 'bg-purple-900'
              }`}>
                {filter.count}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {filteredAlerts.map((alert, index) => {
          const colors = severityColors[alert.severity as keyof typeof severityColors];
          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`backdrop-blur-xl border rounded-xl p-5 transition-all cursor-pointer group ${
                theme === 'light'
                  ? `bg-purple-50/80 border-purple-300/50 hover:border-purple-400`
                  : `bg-purple-950/50 border-purple-800/50 hover:border-purple-700`
              }`}
            >
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 ${colors.bg} rounded-xl flex items-center justify-center flex-shrink-0`}>
                  <alert.icon className={`w-6 h-6 ${colors.text}`} strokeWidth={2} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className={`text-lg ${theme === 'light' ? 'text-purple-900' : 'text-purple-50'}`}>
                          {alert.title}
                        </h3>
                        <span className={`px-2 py-1 rounded text-xs uppercase tracking-wide ${colors.bg} ${colors.text}`}>
                          {alert.severity}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs ${statusColors[alert.status as keyof typeof statusColors]}`}>
                          {alert.status}
                        </span>
                      </div>
                      <p className={`text-sm mb-3 ${theme === 'light' ? 'text-purple-600' : 'text-purple-400'}`}>
                        {alert.description}
                      </p>
                      <div className={`flex flex-wrap items-center gap-4 text-sm ${
                        theme === 'light' ? 'text-purple-700' : 'text-purple-500'
                      }`}>
                        <div className="flex items-center gap-1">
                          <MapPin className="w-4 h-4" />
                          {alert.location}
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {alert.time}
                        </div>
                        <div className="flex items-center gap-1">
                          <span>Assigned to:</span>
                          <span className="text-purple-600">{alert.assignee}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 mt-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-purple-50 rounded-lg text-sm transition-colors">
                      View Details
                    </button>
                    <button className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                      theme === 'light'
                        ? 'bg-purple-200 hover:bg-purple-300 text-purple-900'
                        : 'bg-purple-900 hover:bg-purple-800 text-purple-50'
                    }`}>
                      Take Action
                    </button>
                    {alert.status !== 'resolved' && (
                      <button className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-purple-50 rounded-lg text-sm transition-colors">
                        Mark Resolved
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
