import { 
  Cpu, 
  LayoutDashboard, 
  Shield, 
  Activity, 
  Bell, 
  Settings, 
  Users, 
  BarChart3,
  FileText,
  ChevronLeft,
  ChevronRight,
  LogOut
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useTheme } from '../Dashboard';

interface SidebarProps {
  activeView: string;
  setActiveView: (view: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  onBackToLanding: () => void;
}

export function Sidebar({ activeView, setActiveView, collapsed, setCollapsed, onBackToLanding }: SidebarProps) {
  const { theme } = useTheme();
  
  const menuItems = [
    { id: 'overview', icon: LayoutDashboard, label: 'Overview' },
    { id: 'monitoring', icon: Activity, label: 'Live Monitoring' },
    { id: 'alerts', icon: Bell, label: 'Alerts', badge: 8 },
    { id: 'analytics', icon: BarChart3, label: 'Analytics' },
    { id: 'incidents', icon: Shield, label: 'Incidents' },
    { id: 'users', icon: Users, label: 'Users' },
    { id: 'reports', icon: FileText, label: 'Reports' },
    { id: 'settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <motion.aside
      className={`backdrop-blur-xl border-r flex flex-col relative ${
        theme === 'light'
          ? 'bg-white/80 border-purple-200'
          : 'bg-zinc-950/95 border-zinc-900'
      }`}
      initial={false}
      animate={{ width: collapsed ? 80 : 280 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
    >
      {/* Toggle Button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className={`absolute -right-3 top-20 z-50 w-6 h-6 border rounded-full flex items-center justify-center transition-colors ${
          theme === 'light'
            ? 'bg-white border-purple-300 text-purple-600 hover:bg-purple-50'
            : 'bg-zinc-900 border-zinc-800 text-purple-400 hover:bg-zinc-800'
        }`}
      >
        {collapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
      </button>

      {/* Logo */}
      <div className={`p-6 border-b ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
        <button onClick={onBackToLanding} className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
            <Cpu className="w-6 h-6 text-white" strokeWidth={2} />
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <h1 className={`text-lg whitespace-nowrap ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                  Tobeles AI
                </h1>
                <p className={`text-xs whitespace-nowrap ${theme === 'light' ? 'text-purple-600' : 'text-purple-400'}`}>
                  Security Platform
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {menuItems.map((item) => {
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all group relative ${
                isActive
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : theme === 'light'
                  ? 'text-zinc-700 hover:text-zinc-900 hover:bg-purple-50'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-900'
              }`}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" strokeWidth={isActive ? 2 : 1.5} />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden whitespace-nowrap text-sm font-medium"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
              {item.badge && (
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0 }}
                      className="ml-auto bg-red-500 text-white text-xs px-2 py-0.5 rounded-full font-medium"
                    >
                      {item.badge}
                    </motion.span>
                  )}
                </AnimatePresence>
              )}
              {item.badge && collapsed && (
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Status & Logout */}
      <div className={`p-4 border-t space-y-2 ${theme === 'light' ? 'border-purple-200' : 'border-zinc-900'}`}>
        <div className={`rounded-xl p-3 ${
          theme === 'light' ? 'bg-purple-50' : 'bg-zinc-900'
        } ${collapsed ? 'flex justify-center' : ''}`}>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            <AnimatePresence>
              {!collapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={`text-xs font-medium ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}
                >
                  All Systems Operational
                </motion.span>
              )}
            </AnimatePresence>
          </div>
        </div>

        <button
          onClick={onBackToLanding}
          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
            theme === 'light'
              ? 'text-red-600 hover:bg-red-50'
              : 'text-red-500 hover:bg-red-500/10'
          } ${collapsed ? 'justify-center' : ''}`}
        >
          <LogOut className="w-5 h-5 flex-shrink-0" strokeWidth={1.5} />
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden whitespace-nowrap text-sm font-medium"
              >
                Sign Out
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside>
  );
}
