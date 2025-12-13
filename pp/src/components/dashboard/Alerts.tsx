import { Shield, Swords, Volume2, Users, AlertCircle, MapPin, Clock } from 'lucide-react';
import { motion } from 'motion/react';

export function Alerts() {
  const alerts = [
    {
      id: 1,
      type: 'fight',
      icon: Swords,
      title: 'Physical Altercation Detected',
      location: 'Building A - Main Entrance',
      time: '2 minutes ago',
      severity: 'high',
      status: 'active',
      description: 'Multiple individuals engaged in physical conflict',
      color: 'red',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-400/30',
    },
    {
      id: 2,
      type: 'weapon',
      icon: Shield,
      title: 'Weapon Identified',
      location: 'Building B - Parking Lot',
      time: '5 minutes ago',
      severity: 'critical',
      status: 'investigating',
      description: 'Potential firearm detected in surveillance feed',
      color: 'orange',
      bgColor: 'bg-orange-500/10',
      borderColor: 'border-orange-400/30',
    },
    {
      id: 3,
      type: 'scream',
      icon: Volume2,
      title: 'Distress Call Detected',
      location: 'Building C - Floor 3',
      time: '8 minutes ago',
      severity: 'medium',
      status: 'resolved',
      description: 'Audio analysis detected distress vocalization',
      color: 'yellow',
      bgColor: 'bg-yellow-500/10',
      borderColor: 'border-yellow-400/30',
    },
    {
      id: 4,
      type: 'crowd',
      icon: Users,
      title: 'Crowd Density Alert',
      location: 'Building A - Lobby',
      time: '12 minutes ago',
      severity: 'low',
      status: 'monitoring',
      description: 'Unusual crowd gathering detected, density above threshold',
      color: 'blue',
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-400/30',
    },
    {
      id: 5,
      type: 'fight',
      icon: Swords,
      title: 'Aggressive Behavior',
      location: 'Building D - Cafeteria',
      time: '18 minutes ago',
      severity: 'medium',
      status: 'resolved',
      description: 'Aggressive gestures and posturing detected',
      color: 'red',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-400/30',
    },
  ];

  const severityColors = {
    critical: 'bg-red-600 text-white',
    high: 'bg-orange-600 text-white',
    medium: 'bg-yellow-600 text-white',
    low: 'bg-blue-600 text-white',
  };

  const statusColors = {
    active: 'bg-red-500/20 text-red-300 border-red-400/30',
    investigating: 'bg-orange-500/20 text-orange-300 border-orange-400/30',
    monitoring: 'bg-blue-500/20 text-blue-300 border-blue-400/30',
    resolved: 'bg-green-500/20 text-green-300 border-green-400/30',
  };

  return (
    <section>
      {/* Header */}
      <motion.div
        className="flex items-center justify-between mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <h2 className="text-2xl text-purple-100 mb-1">Recent Alerts</h2>
          <p className="text-purple-300">Real-time security event notifications</p>
        </div>
        
        <div className="flex items-center gap-3">
          <button className="px-4 py-2 backdrop-blur-xl bg-white/10 border border-white/20 rounded-lg text-purple-200 hover:bg-white/20 transition-all text-sm">
            Filter
          </button>
          <button className="px-4 py-2 backdrop-blur-xl bg-white/10 border border-white/20 rounded-lg text-purple-200 hover:bg-white/20 transition-all text-sm">
            Export
          </button>
        </div>
      </motion.div>

      {/* Alerts List */}
      <div className="space-y-4">
        {alerts.map((alert, index) => (
          <motion.div
            key={alert.id}
            className={`backdrop-blur-xl bg-white/5 rounded-2xl p-6 border ${alert.borderColor} hover:bg-white/10 transition-all cursor-pointer group`}
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            whileHover={{ scale: 1.01, x: 4 }}
          >
            <div className="flex items-start gap-4">
              {/* Icon */}
              <motion.div
                className={`w-14 h-14 ${alert.bgColor} rounded-xl flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform`}
                whileHover={{ rotate: [0, -5, 5, 0] }}
                transition={{ duration: 0.3 }}
              >
                <alert.icon className={`w-7 h-7 text-${alert.color}-400`} strokeWidth={1.5} />
              </motion.div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4 mb-2">
                  <div className="flex-1">
                    <h3 className="text-lg text-purple-100 mb-1 group-hover:text-white transition-colors">
                      {alert.title}
                    </h3>
                    <p className="text-sm text-purple-300">{alert.description}</p>
                  </div>

                  {/* Severity Badge */}
                  <span className={`px-3 py-1 rounded-full text-xs whitespace-nowrap ${severityColors[alert.severity as keyof typeof severityColors]}`}>
                    {alert.severity.toUpperCase()}
                  </span>
                </div>

                {/* Meta Information */}
                <div className="flex flex-wrap items-center gap-4 mt-4">
                  <div className="flex items-center gap-2 text-sm text-purple-400">
                    <MapPin className="w-4 h-4" />
                    <span>{alert.location}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-purple-400">
                    <Clock className="w-4 h-4" />
                    <span>{alert.time}</span>
                  </div>
                  <div className={`px-3 py-1 rounded-lg border text-xs ${statusColors[alert.status as keyof typeof statusColors]}`}>
                    {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-3 mt-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="px-4 py-2 bg-gradient-to-r from-purple-600 to-purple-700 rounded-lg text-white text-sm hover:shadow-lg transition-all">
                    View Details
                  </button>
                  <button className="px-4 py-2 backdrop-blur-lg bg-white/10 border border-white/20 rounded-lg text-purple-200 text-sm hover:bg-white/20 transition-all">
                    Take Action
                  </button>
                  <button className="px-4 py-2 backdrop-blur-lg bg-white/10 border border-white/20 rounded-lg text-purple-200 text-sm hover:bg-white/20 transition-all">
                    Mark Resolved
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Load More */}
      <motion.div
        className="text-center mt-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <button className="px-8 py-3 backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl text-purple-200 hover:bg-white/20 transition-all">
          Load More Alerts
        </button>
      </motion.div>
    </section>
  );
}
