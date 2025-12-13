import { Search, Bell, Settings, User, ChevronDown, LogOut, HelpCircle, Moon, Sun } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { useTheme } from '../Dashboard';

export function TopBar() {
  const { theme, toggleTheme } = useTheme();
  const [searchFocused, setSearchFocused] = useState(false);
  const [searchValue, setSearchValue] = useState('');
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const notifRef = useRef<HTMLDivElement>(null);
  const profileRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(event.target as Node)) {
        setNotificationOpen(false);
      }
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setProfileOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const notifications = [
    { id: 1, type: 'alert', message: 'Fight detected in Building A', time: '2 min ago', unread: true },
    { id: 2, type: 'info', message: 'System update completed', time: '1 hour ago', unread: true },
    { id: 3, type: 'warning', message: 'Camera 5 offline', time: '3 hours ago', unread: false },
  ];

  return (
    <header className={`backdrop-blur-xl border-b px-6 py-4 ${
      theme === 'light'
        ? 'bg-white/80 border-purple-200'
        : 'bg-zinc-950/95 border-zinc-900'
    }`}>
      <div className="flex items-center justify-between gap-4">
        {/* Search */}
        <div className="flex-1 max-w-2xl">
          <div className={`relative transition-all ${searchFocused ? 'scale-[1.01]' : ''}`}>
            <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${
              theme === 'light' ? 'text-zinc-400' : 'text-zinc-500'
            }`} />
            <input
              type="text"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setSearchFocused(false)}
              placeholder="Search alerts, locations, or incidents..."
              className={`w-full pl-12 pr-4 py-3 border rounded-xl focus:outline-none transition-all ${
                theme === 'light'
                  ? 'bg-purple-50/50 border-purple-200 text-zinc-900 placeholder-zinc-500 focus:border-purple-400 focus:bg-white'
                  : 'bg-zinc-900 border-zinc-800 text-white placeholder-zinc-500 focus:border-purple-500 focus:bg-zinc-900/80'
              }`}
            />
            {searchValue && (
              <button
                onClick={() => setSearchValue('')}
                className={`absolute right-4 top-1/2 -translate-y-1/2 transition-colors ${
                  theme === 'light' ? 'text-zinc-400 hover:text-zinc-900' : 'text-zinc-500 hover:text-white'
                }`}
              >
                ×
              </button>
            )}
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-3">
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className={`w-10 h-10 border rounded-xl flex items-center justify-center transition-all ${
              theme === 'light'
                ? 'bg-purple-50 hover:bg-purple-100 border-purple-200 text-purple-600'
                : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-purple-400'
            }`}
          >
            {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
          </button>

          {/* Notifications */}
          <div className="relative" ref={notifRef}>
            <button
              onClick={() => setNotificationOpen(!notificationOpen)}
              className={`relative w-10 h-10 border rounded-xl flex items-center justify-center transition-all ${
                theme === 'light'
                  ? 'bg-purple-50 hover:bg-purple-100 border-purple-200 text-zinc-700'
                  : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-zinc-300'
              }`}
            >
              <Bell className="w-5 h-5" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>

            <AnimatePresence>
              {notificationOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className={`absolute right-0 top-12 w-80 border rounded-xl shadow-2xl overflow-hidden z-50 ${
                    theme === 'light'
                      ? 'bg-white border-purple-200'
                      : 'bg-zinc-950 border-zinc-900'
                  }`}
                >
                  <div className={`p-4 border-b ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
                    <div className="flex items-center justify-between">
                      <h3 className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                        Notifications
                      </h3>
                      <button className="text-xs text-purple-600 hover:text-purple-700 font-medium">
                        Mark all read
                      </button>
                    </div>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {notifications.map((notif) => (
                      <div
                        key={notif.id}
                        className={`p-4 border-b cursor-pointer transition-colors ${
                          theme === 'light'
                            ? `border-purple-100 hover:bg-purple-50 ${notif.unread ? 'bg-purple-50/50' : ''}`
                            : `border-zinc-900 hover:bg-zinc-900 ${notif.unread ? 'bg-zinc-900/50' : ''}`
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          {notif.unread && <div className="w-2 h-2 bg-purple-600 rounded-full mt-2"></div>}
                          <div className="flex-1 min-w-0">
                            <p className={`text-sm ${theme === 'light' ? 'text-zinc-900' : 'text-zinc-200'}`}>
                              {notif.message}
                            </p>
                            <p className={`text-xs mt-1 ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-500'}`}>
                              {notif.time}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className={`p-3 border-t ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
                    <button className="w-full text-center text-sm text-purple-600 hover:text-purple-700 font-medium">
                      View all notifications
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Settings */}
          <button className={`w-10 h-10 border rounded-xl flex items-center justify-center transition-all ${
            theme === 'light'
              ? 'bg-purple-50 hover:bg-purple-100 border-purple-200 text-zinc-700'
              : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-zinc-300'
          }`}>
            <Settings className="w-5 h-5" />
          </button>

          {/* Profile */}
          <div className="relative" ref={profileRef}>
            <button
              onClick={() => setProfileOpen(!profileOpen)}
              className={`flex items-center gap-3 pl-3 pr-4 py-2 border rounded-xl transition-all ${
                theme === 'light'
                  ? 'bg-purple-50 hover:bg-purple-100 border-purple-200'
                  : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800'
              }`}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                <User className="w-5 h-5 text-white" />
              </div>
              <span className={`text-sm font-medium hidden lg:block ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                Sarah Chen
              </span>
              <ChevronDown className={`w-4 h-4 ${theme === 'light' ? 'text-zinc-400' : 'text-zinc-500'}`} />
            </button>

            <AnimatePresence>
              {profileOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className={`absolute right-0 top-12 w-64 border rounded-xl shadow-2xl overflow-hidden z-50 ${
                    theme === 'light'
                      ? 'bg-white border-purple-200'
                      : 'bg-zinc-950 border-zinc-900'
                  }`}
                >
                  <div className={`p-4 border-b ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                        <User className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <p className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                          Sarah Chen
                        </p>
                        <p className={`text-xs ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-400'}`}>
                          Security Admin
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="p-2">
                    <button className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                      theme === 'light'
                        ? 'text-zinc-700 hover:bg-purple-50'
                        : 'text-zinc-300 hover:bg-zinc-900'
                    }`}>
                      <User className="w-4 h-4" />
                      <span className="text-sm font-medium">My Profile</span>
                    </button>
                    <button className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                      theme === 'light'
                        ? 'text-zinc-700 hover:bg-purple-50'
                        : 'text-zinc-300 hover:bg-zinc-900'
                    }`}>
                      <Settings className="w-4 h-4" />
                      <span className="text-sm font-medium">Settings</span>
                    </button>
                    <button className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                      theme === 'light'
                        ? 'text-zinc-700 hover:bg-purple-50'
                        : 'text-zinc-300 hover:bg-zinc-900'
                    }`}>
                      <HelpCircle className="w-4 h-4" />
                      <span className="text-sm font-medium">Help & Support</span>
                    </button>
                  </div>
                  <div className={`p-2 border-t ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
                    <button className="w-full flex items-center gap-3 px-3 py-2 text-red-500 hover:bg-red-500/10 rounded-lg transition-colors">
                      <LogOut className="w-4 h-4" />
                      <span className="text-sm font-medium">Sign Out</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </header>
  );
}
