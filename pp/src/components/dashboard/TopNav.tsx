import { Cpu, Search, Bell, Settings, User, ChevronDown, LogOut, Moon, Sun, LayoutDashboard, Activity, Shield, BarChart3, Users, FileText, Menu, X } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { useTheme } from '../Dashboard';

interface TopNavProps {
  activeView: string;
  setActiveView: (view: string) => void;
  onBackToLanding: () => void;
}

export function TopNav({ activeView, setActiveView, onBackToLanding }: TopNavProps) {
  const { theme, toggleTheme } = useTheme();
  const [searchValue, setSearchValue] = useState('');
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
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

  const menuItems = [
    { id: 'overview', icon: LayoutDashboard, label: 'Overview' },
    { id: 'monitoring', icon: Activity, label: 'Monitoring' },
    { id: 'alerts', icon: Bell, label: 'Alerts', badge: 8 },
    { id: 'analytics', icon: BarChart3, label: 'Analytics' },
    { id: 'incidents', icon: Shield, label: 'Incidents' },
    { id: 'users', icon: Users, label: 'Users' },
    { id: 'reports', icon: FileText, label: 'Reports' },
    { id: 'settings', icon: Settings, label: 'Settings' },
  ];

  const notifications = [
    { id: 1, message: 'Fight detected in Building A', time: '2 min ago', unread: true },
    { id: 2, message: 'System update completed', time: '1 hour ago', unread: true },
    { id: 3, message: 'Camera 5 offline', time: '3 hours ago', unread: false },
  ];

  return (
    <>
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className={`fixed top-0 left-0 right-0 z-50 backdrop-blur-2xl ${
          theme === 'light'
            ? 'bg-white/80 border-b border-purple-200'
            : 'bg-zinc-950/95 border-b border-zinc-900'
        }`}
      >
        <div className="max-w-[2000px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between gap-6">
            {/* Logo */}
            <motion.button
              onClick={onBackToLanding}
              className="flex items-center gap-3 group"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20 relative overflow-hidden">
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  animate={{ x: ['-100%', '200%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
                <Cpu className="w-6 h-6 text-white relative z-10" strokeWidth={2} />
              </div>
              <div className="hidden md:block">
                <h1 className={`text-lg font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                  Tobeles AI
                </h1>
                <p className={`text-xs font-medium ${theme === 'light' ? 'text-purple-600' : 'text-purple-400'}`}>
                  Security Platform
                </p>
              </div>
            </motion.button>

            {/* Navigation Menu - Desktop */}
            <div className="hidden lg:flex items-center gap-2">
              {menuItems.map((item) => (
                <motion.button
                  key={item.id}
                  onClick={() => setActiveView(item.id)}
                  className={`relative px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                    activeView === item.id
                      ? 'text-white'
                      : theme === 'light'
                      ? 'text-zinc-700 hover:text-zinc-900 hover:bg-purple-50'
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {activeView === item.id && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 bg-gradient-to-r from-purple-600 to-purple-700 rounded-xl shadow-lg shadow-purple-500/25"
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-2">
                    <item.icon className="w-4 h-4" />
                    {item.label}
                    {item.badge && (
                      <span className="px-1.5 py-0.5 bg-red-500 text-white text-xs rounded-full">
                        {item.badge}
                      </span>
                    )}
                  </span>
                </motion.button>
              ))}
            </div>

            {/* Right Actions */}
            <div className="flex items-center gap-3">
              {/* Search */}
              <div className="hidden md:block relative">
                <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${
                  theme === 'light' ? 'text-zinc-400' : 'text-zinc-500'
                }`} />
                <input
                  type="text"
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value)}
                  placeholder="Search..."
                  className={`pl-10 pr-4 py-2 w-64 border rounded-xl text-sm focus:outline-none transition-all ${
                    theme === 'light'
                      ? 'bg-purple-50/50 border-purple-200 text-zinc-900 placeholder-zinc-500 focus:border-purple-400 focus:bg-white'
                      : 'bg-zinc-900 border-zinc-800 text-white placeholder-zinc-500 focus:border-purple-500'
                  }`}
                />
              </div>

              {/* Theme Toggle */}
              <motion.button
                onClick={toggleTheme}
                className={`w-10 h-10 border rounded-xl flex items-center justify-center transition-all ${
                  theme === 'light'
                    ? 'bg-purple-50 hover:bg-purple-100 border-purple-200 text-purple-600'
                    : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-purple-400'
                }`}
                whileHover={{ scale: 1.1, rotate: 180 }}
                whileTap={{ scale: 0.9 }}
              >
                {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
              </motion.button>

              {/* Notifications */}
              <div className="relative" ref={notifRef}>
                <motion.button
                  onClick={() => setNotificationOpen(!notificationOpen)}
                  className={`relative w-10 h-10 border rounded-xl flex items-center justify-center transition-all ${
                    theme === 'light'
                      ? 'bg-purple-50 hover:bg-purple-100 border-purple-200 text-zinc-700'
                      : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-zinc-300'
                  }`}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Bell className="w-5 h-5" />
                  <motion.span 
                    className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </motion.button>

                <AnimatePresence>
                  {notificationOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className={`absolute right-0 top-14 w-80 border rounded-2xl shadow-2xl overflow-hidden ${
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
                          <motion.div
                            key={notif.id}
                            className={`p-4 border-b cursor-pointer transition-colors ${
                              theme === 'light'
                                ? `border-purple-100 hover:bg-purple-50 ${notif.unread ? 'bg-purple-50/50' : ''}`
                                : `border-zinc-900 hover:bg-zinc-900 ${notif.unread ? 'bg-zinc-900/50' : ''}`
                            }`}
                            whileHover={{ x: 4 }}
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
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Profile */}
              <div className="relative hidden md:block" ref={profileRef}>
                <motion.button
                  onClick={() => setProfileOpen(!profileOpen)}
                  className={`flex items-center gap-2 pl-2 pr-3 py-2 border rounded-xl transition-all ${
                    theme === 'light'
                      ? 'bg-purple-50 hover:bg-purple-100 border-purple-200'
                      : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                    <User className="w-5 h-5 text-white" />
                  </div>
                  <span className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                    Айгерім
                  </span>
                  <ChevronDown className={`w-4 h-4 ${theme === 'light' ? 'text-zinc-400' : 'text-zinc-500'}`} />
                </motion.button>

                <AnimatePresence>
                  {profileOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      className={`absolute right-0 top-14 w-64 border rounded-2xl shadow-2xl overflow-hidden ${
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
                              Айгерім Нұрланова
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
                      </div>
                      <div className={`p-2 border-t ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
                        <button
                          onClick={onBackToLanding}
                          className="w-full flex items-center gap-3 px-3 py-2 text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          <span className="text-sm font-medium">Sign Out</span>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Mobile Menu Button */}
              <motion.button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className={`lg:hidden w-10 h-10 border rounded-xl flex items-center justify-center transition-all ${
                  theme === 'light'
                    ? 'bg-purple-50 hover:bg-purple-100 border-purple-200 text-zinc-700'
                    : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-zinc-300'
                }`}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </motion.button>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 lg:hidden"
            onClick={() => setMobileMenuOpen(false)}
          >
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className={`absolute right-0 top-0 bottom-0 w-80 max-w-[85vw] ${
                theme === 'light' ? 'bg-white' : 'bg-zinc-950'
              } shadow-2xl`}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 h-full overflow-y-auto">
                <div className="flex items-center justify-between mb-8">
                  <h2 className={`text-xl font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                    Menu
                  </h2>
                  <button
                    onClick={() => setMobileMenuOpen(false)}
                    className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                      theme === 'light' ? 'hover:bg-purple-50' : 'hover:bg-zinc-900'
                    }`}
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="space-y-2">
                  {menuItems.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => {
                        setActiveView(item.id);
                        setMobileMenuOpen(false);
                      }}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                        activeView === item.id
                          ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                          : theme === 'light'
                          ? 'text-zinc-700 hover:bg-purple-50'
                          : 'text-zinc-400 hover:bg-zinc-900'
                      }`}
                    >
                      <item.icon className="w-5 h-5" />
                      {item.label}
                      {item.badge && (
                        <span className="ml-auto px-2 py-0.5 bg-red-500 text-white text-xs rounded-full">
                          {item.badge}
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                <div className={`mt-8 pt-8 border-t ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                      <User className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <p className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                        Айгерім Нұрланова
                      </p>
                      <p className={`text-xs ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-400'}`}>
                        Security Admin
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={onBackToLanding}
                    className="w-full flex items-center gap-3 px-4 py-3 text-red-500 hover:bg-red-500/10 rounded-xl transition-colors"
                  >
                    <LogOut className="w-5 h-5" />
                    <span className="text-sm font-medium">Sign Out</span>
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
