import { User, Settings, Bell, LogOut, ChevronDown, Moon, Sun, Globe } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';

export function UserProfile() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [darkMode, setDarkMode] = useState(true);

  return (
    <motion.aside
      className="w-80 backdrop-blur-xl bg-white/5 border-l border-white/10 p-6 space-y-6"
      initial={{ x: 100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      {/* User Profile Card */}
      <motion.div
        className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10"
        whileHover={{ scale: 1.02 }}
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            {/* Avatar */}
            <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-purple-700 rounded-full flex items-center justify-center shadow-lg shadow-purple-500/30">
              <User className="w-7 h-7 text-white" strokeWidth={1.5} />
            </div>
            
            <div>
              <h3 className="text-lg text-purple-100">Sarah Chen</h3>
              <p className="text-sm text-purple-400">Security Admin</p>
            </div>
          </div>

          {/* Dropdown Toggle */}
          <motion.button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-8 h-8 backdrop-blur-lg bg-white/10 rounded-lg flex items-center justify-center text-purple-300 hover:bg-white/20 transition-all"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChevronDown className="w-4 h-4" />
            </motion.div>
          </motion.button>
        </div>

        {/* Expanded Menu */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="pt-4 border-t border-white/10 space-y-2">
                <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-purple-300 hover:bg-white/10 transition-all text-sm">
                  <User className="w-4 h-4" />
                  <span>View Profile</span>
                </button>
                <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-purple-300 hover:bg-white/10 transition-all text-sm">
                  <Settings className="w-4 h-4" />
                  <span>Settings</span>
                </button>
                <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-red-400 hover:bg-red-500/10 transition-all text-sm">
                  <LogOut className="w-4 h-4" />
                  <span>Sign Out</span>
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Quick Stats */}
      <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg text-purple-100 mb-4">Your Activity</h3>
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-purple-300">Alerts Handled</span>
              <span className="text-purple-100">47</span>
            </div>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-purple-500 to-purple-700"
                initial={{ width: 0 }}
                animate={{ width: '78%' }}
                transition={{ duration: 1, delay: 0.5 }}
              />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-purple-300">Response Time</span>
              <span className="text-purple-100">2.3m</span>
            </div>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-green-500 to-green-700"
                initial={{ width: 0 }}
                animate={{ width: '92%' }}
                transition={{ duration: 1, delay: 0.7 }}
              />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-purple-300">Resolution Rate</span>
              <span className="text-purple-100">94%</span>
            </div>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-700"
                initial={{ width: 0 }}
                animate={{ width: '94%' }}
                transition={{ duration: 1, delay: 0.9 }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Notifications */}
      <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg text-purple-100">Notifications</h3>
          <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
            <span className="text-xs text-white">3</span>
          </div>
        </div>

        <div className="space-y-3">
          {[
            { text: 'New alert in Building A', time: '2m ago', unread: true },
            { text: 'System update available', time: '1h ago', unread: true },
            { text: 'Monthly report ready', time: '3h ago', unread: true },
          ].map((notification, index) => (
            <motion.div
              key={index}
              className="p-3 backdrop-blur-lg bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-all cursor-pointer"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              whileHover={{ x: -4 }}
            >
              <div className="flex items-start gap-3">
                {notification.unread && (
                  <div className="w-2 h-2 bg-purple-500 rounded-full mt-1.5"></div>
                )}
                <div className="flex-1">
                  <p className="text-sm text-purple-200">{notification.text}</p>
                  <p className="text-xs text-purple-400 mt-1">{notification.time}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <button className="w-full mt-4 py-2 text-sm text-purple-300 hover:text-purple-200 transition-colors">
          View All
        </button>
      </div>

      {/* Settings Quick Access */}
      <div className="backdrop-blur-xl bg-white/5 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg text-purple-100 mb-4">Quick Settings</h3>
        
        <div className="space-y-3">
          {/* Dark Mode Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {darkMode ? <Moon className="w-4 h-4 text-purple-400" /> : <Sun className="w-4 h-4 text-purple-400" />}
              <span className="text-sm text-purple-300">Dark Mode</span>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`w-12 h-6 rounded-full transition-all relative ${
                darkMode ? 'bg-purple-600' : 'bg-white/20'
              }`}
            >
              <motion.div
                className="w-5 h-5 bg-white rounded-full absolute top-0.5"
                animate={{ x: darkMode ? 26 : 2 }}
                transition={{ duration: 0.3 }}
              />
            </button>
          </div>

          {/* Notifications Toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bell className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-purple-300">Notifications</span>
            </div>
            <button className="w-12 h-6 bg-purple-600 rounded-full transition-all relative">
              <motion.div
                className="w-5 h-5 bg-white rounded-full absolute top-0.5"
                initial={{ x: 26 }}
              />
            </button>
          </div>

          {/* Language */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Globe className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-purple-300">Language</span>
            </div>
            <select className="bg-white/10 border border-white/20 rounded-lg px-3 py-1 text-sm text-purple-200 focus:outline-none focus:border-purple-400">
              <option>English</option>
              <option>Spanish</option>
              <option>French</option>
            </select>
          </div>
        </div>
      </div>
    </motion.aside>
  );
}
