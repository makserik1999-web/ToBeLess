import { Shield, Search, Filter, Download, MapPin, Clock, User, FileText, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';

export function IncidentsView() {
  const { theme } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedIncident, setSelectedIncident] = useState<number | null>(null);

  const incidents = [
    {
      id: 1,
      title: 'Physical Altercation - Building A',
      type: 'Fight',
      location: 'Building A - East Wing, Floor 2, Room 205',
      date: '2024-12-11',
      time: '14:23:15',
      status: 'closed',
      severity: 'high',
      assignee: 'John Doe',
      responder: 'Security Team Alpha',
      responseTime: '2.3 min',
      description: 'Two individuals engaged in physical conflict. Security intervened promptly. No injuries reported.',
      actions: ['Security dispatched', 'Incident documented', 'Parties separated', 'Report filed'],
      evidence: ['Camera footage', 'Witness statements', 'Incident report']
    },
    {
      id: 2,
      title: 'Weapon Detection - Parking Lot',
      type: 'Weapon',
      location: 'Building B - Parking Lot, Zone 3',
      date: '2024-12-11',
      time: '13:45:22',
      status: 'investigating',
      severity: 'critical',
      assignee: 'Jane Smith',
      responder: 'Security Team Bravo',
      responseTime: '1.8 min',
      description: 'Potential firearm detected by AI system. Law enforcement notified. Area secured.',
      actions: ['Area secured', 'Law enforcement contacted', 'Witnesses interviewed', 'Under investigation'],
      evidence: ['Camera footage', 'AI detection log', 'Police report']
    },
    {
      id: 3,
      title: 'Distress Call - Building C',
      type: 'Scream',
      location: 'Building C - Floor 3, Room 301',
      date: '2024-12-11',
      time: '12:18:45',
      status: 'closed',
      severity: 'medium',
      assignee: 'Mike Johnson',
      responder: 'Security Team Alpha',
      responseTime: '3.1 min',
      description: 'Audio system detected distress vocalization. False alarm - training exercise in progress.',
      actions: ['Security responded', 'Situation assessed', 'False alarm confirmed', 'All clear'],
      evidence: ['Audio recording', 'Response log']
    },
    {
      id: 4,
      title: 'Crowd Density Alert - Main Lobby',
      type: 'Crowd',
      location: 'Building A - Main Lobby',
      date: '2024-12-11',
      time: '11:52:30',
      status: 'closed',
      severity: 'low',
      assignee: 'Sarah Chen',
      responder: 'Security Team Charlie',
      responseTime: '4.2 min',
      description: 'Crowd density exceeded safety threshold during lunch hour. Traffic flow managed successfully.',
      actions: ['Crowd monitored', 'Additional staff deployed', 'Flow managed', 'Resolved'],
      evidence: ['Crowd analysis data', 'Camera footage']
    },
  ];

  const filteredIncidents = incidents.filter(incident => {
    if (statusFilter !== 'all' && incident.status !== statusFilter) return false;
    if (searchQuery && !incident.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !incident.location.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const statusCounts = {
    all: incidents.length,
    investigating: incidents.filter(i => i.status === 'investigating').length,
    closed: incidents.filter(i => i.status === 'closed').length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Incident Management
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Track and manage security incidents
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

      {/* Search and Status Filter */}
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="flex-1">
          <div className="relative">
            <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${
              theme === 'light' ? 'text-zinc-400' : 'text-zinc-500'
            }`} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search incidents by title or location..."
              className={`w-full pl-12 pr-4 py-3 border rounded-xl focus:outline-none transition-all ${
                theme === 'light'
                  ? 'bg-purple-50/50 border-purple-200 text-zinc-900 placeholder-zinc-500 focus:border-purple-400 focus:bg-white'
                  : 'bg-zinc-900 border-zinc-800 text-white placeholder-zinc-500 focus:border-purple-500 focus:bg-zinc-900/80'
              }`}
            />
          </div>
        </div>
        <div className="flex gap-2">
          {Object.entries(statusCounts).map(([status, count]) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={`px-4 py-3 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${
                statusFilter === status
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                  : theme === 'light'
                  ? 'bg-white border border-purple-200 text-zinc-700 hover:bg-purple-50'
                  : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:bg-zinc-800'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
              <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                statusFilter === status
                  ? 'bg-purple-700'
                  : theme === 'light'
                  ? 'bg-purple-100 text-purple-700'
                  : 'bg-zinc-800 text-zinc-400'
              }`}>
                {count}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Incidents List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Incidents */}
        <div className="space-y-4">
          {filteredIncidents.map((incident, index) => (
            <motion.div
              key={incident.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => setSelectedIncident(incident.id)}
              className={`backdrop-blur-xl border rounded-2xl p-6 transition-all cursor-pointer ${
                selectedIncident === incident.id
                  ? theme === 'light'
                    ? 'bg-purple-50 border-purple-400 shadow-lg shadow-purple-100'
                    : 'bg-purple-950/50 border-purple-700 shadow-lg shadow-purple-500/10'
                  : theme === 'light'
                  ? 'bg-white border-purple-200 hover:border-purple-300'
                  : 'bg-zinc-950 border-zinc-900 hover:border-zinc-800'
              }`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className={`text-lg font-semibold mb-2 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                    {incident.title}
                  </h3>
                  <div className="flex flex-wrap gap-2 mb-3">
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                      incident.severity === 'critical' ? 'bg-red-500/20 text-red-600' :
                      incident.severity === 'high' ? 'bg-orange-500/20 text-orange-600' :
                      incident.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-600' :
                      'bg-blue-500/20 text-blue-600'
                    }`}>
                      {incident.severity.toUpperCase()}
                    </span>
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                      incident.status === 'investigating' ? 'bg-orange-500/20 text-orange-600' :
                      'bg-emerald-500/20 text-emerald-600'
                    }`}>
                      {incident.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className={`space-y-2 text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  {incident.location}
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  {incident.date} at {incident.time}
                </div>
                <div className="flex items-center gap-2">
                  <User className="w-4 h-4" />
                  Assigned to {incident.assignee}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Incident Details */}
        <div className={`backdrop-blur-xl border rounded-2xl p-6 sticky top-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          {selectedIncident ? (
            <div>
              {(() => {
                const incident = incidents.find(i => i.id === selectedIncident)!;
                return (
                  <>
                    <h2 className={`text-2xl font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                      Incident Details
                    </h2>

                    <div className="space-y-6">
                      {/* Description */}
                      <div>
                        <h3 className={`text-sm font-semibold mb-2 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Description
                        </h3>
                        <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                          {incident.description}
                        </p>
                      </div>

                      {/* Response Details */}
                      <div>
                        <h3 className={`text-sm font-semibold mb-2 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Response Details
                        </h3>
                        <div className={`space-y-2 text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                          <p>Responder: {incident.responder}</p>
                          <p>Response Time: {incident.responseTime}</p>
                        </div>
                      </div>

                      {/* Actions Taken */}
                      <div>
                        <h3 className={`text-sm font-semibold mb-3 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Actions Taken
                        </h3>
                        <div className="space-y-2">
                          {incident.actions.map((action, index) => (
                            <div key={index} className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-emerald-500" />
                              <span className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                                {action}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Evidence */}
                      <div>
                        <h3 className={`text-sm font-semibold mb-3 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Evidence Collected
                        </h3>
                        <div className="space-y-2">
                          {incident.evidence.map((item, index) => (
                            <div key={index} className={`flex items-center gap-2 p-2 rounded-lg ${
                              theme === 'light' ? 'bg-purple-50' : 'bg-zinc-900'
                            }`}>
                              <FileText className="w-4 h-4 text-purple-500" />
                              <span className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                                {item}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex gap-3 pt-4 border-t border-zinc-200 dark:border-zinc-800">
                        <button className="flex-1 px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-all">
                          Generate Report
                        </button>
                        <button className={`px-4 py-3 border rounded-xl font-medium transition-all ${
                          theme === 'light'
                            ? 'bg-white border-purple-200 text-zinc-700 hover:bg-purple-50'
                            : 'bg-zinc-900 border-zinc-800 text-zinc-300 hover:bg-zinc-800'
                        }`}>
                          Edit
                        </button>
                      </div>
                    </div>
                  </>
                );
              })()}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full py-20">
              <Shield className={`w-16 h-16 mb-4 ${theme === 'light' ? 'text-zinc-300' : 'text-zinc-700'}`} />
              <p className={`font-medium ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-500'}`}>
                Select an incident to view details
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
