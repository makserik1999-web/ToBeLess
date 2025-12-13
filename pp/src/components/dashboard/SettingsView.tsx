import { Settings, Bell, Shield, Users, Palette, Globe, Database, Key } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from '../Dashboard';

export function SettingsView() {
  const { theme } = useTheme();
  const [activeTab, setActiveTab] = useState('general');
  const [notifications, setNotifications] = useState({
    email: true,
    push: true,
    sms: false,
    criticalOnly: false,
  });

  const tabs = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'team', label: 'Team', icon: Users },
    { id: 'appearance', label: 'Appearance', icon: Palette },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
          Settings
        </h1>
        <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
          Manage your account and system preferences
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Tabs */}
        <div className={`backdrop-blur-xl border rounded-2xl p-4 h-fit ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                  activeTab === tab.id
                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                    : theme === 'light'
                    ? 'text-zinc-700 hover:bg-purple-50'
                    : 'text-zinc-400 hover:bg-zinc-900'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="lg:col-span-3 space-y-6">
          {activeTab === 'general' && (
            <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
              theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
            }`}>
              <h2 className={`text-xl font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                General Settings
              </h2>
              <div className="space-y-6">
                <div>
                  <label className={`block text-sm font-semibold mb-2 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                    Organization Name
                  </label>
                  <input
                    type="text"
                    defaultValue="Tobeles AI Security"
                    className={`w-full px-4 py-3 border rounded-xl focus:outline-none transition-all ${
                      theme === 'light'
                        ? 'bg-purple-50/50 border-purple-200 text-zinc-900 focus:border-purple-400 focus:bg-white'
                        : 'bg-zinc-900 border-zinc-800 text-white focus:border-purple-500'
                    }`}
                  />
                </div>

                <div>
                  <label className={`block text-sm font-semibold mb-2 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                    Timezone
                  </label>
                  <select className={`w-full px-4 py-3 border rounded-xl focus:outline-none transition-all ${
                    theme === 'light'
                      ? 'bg-purple-50/50 border-purple-200 text-zinc-900 focus:border-purple-400 focus:bg-white'
                      : 'bg-zinc-900 border-zinc-800 text-white focus:border-purple-500'
                  }`}>
                    <option>UTC-8 (Pacific Time)</option>
                    <option>UTC-5 (Eastern Time)</option>
                    <option>UTC+0 (GMT)</option>
                  </select>
                </div>

                <div>
                  <label className={`block text-sm font-semibold mb-2 ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                    Language
                  </label>
                  <select className={`w-full px-4 py-3 border rounded-xl focus:outline-none transition-all ${
                    theme === 'light'
                      ? 'bg-purple-50/50 border-purple-200 text-zinc-900 focus:border-purple-400 focus:bg-white'
                      : 'bg-zinc-900 border-zinc-800 text-white focus:border-purple-500'
                  }`}>
                    <option>English</option>
                    <option>Spanish</option>
                    <option>French</option>
                  </select>
                </div>

                <button className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-all">
                  Save Changes
                </button>
              </div>
            </div>
          )}

          {activeTab === 'notifications' && (
            <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
              theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
            }`}>
              <h2 className={`text-xl font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                Notification Preferences
              </h2>
              <div className="space-y-6">
                {Object.entries(notifications).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <div>
                      <div className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                        {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                      </div>
                      <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                        Receive notifications via {key}
                      </div>
                    </div>
                    <button
                      onClick={() => setNotifications({ ...notifications, [key]: !value })}
                      className={`w-14 h-7 rounded-full transition-all relative ${
                        value ? 'bg-purple-600' : theme === 'light' ? 'bg-zinc-300' : 'bg-zinc-700'
                      }`}
                    >
                      <div className={`absolute top-1 w-5 h-5 bg-white rounded-full transition-all ${
                        value ? 'right-1' : 'left-1'
                      }`}></div>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'security' && (
            <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
              theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
            }`}>
              <h2 className={`text-xl font-semibold mb-6 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                Security Settings
              </h2>
              <div className="space-y-4">
                <button className={`w-full flex items-center justify-between p-4 border rounded-xl transition-all ${
                  theme === 'light'
                    ? 'border-purple-200 hover:bg-purple-50'
                    : 'border-zinc-800 hover:bg-zinc-900'
                }`}>
                  <div className="flex items-center gap-3">
                    <Key className={`w-5 h-5 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
                    <div className="text-left">
                      <div className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                        Change Password
                      </div>
                      <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                        Update your password regularly
                      </div>
                    </div>
                  </div>
                  <span className="text-purple-600">→</span>
                </button>

                <button className={`w-full flex items-center justify-between p-4 border rounded-xl transition-all ${
                  theme === 'light'
                    ? 'border-purple-200 hover:bg-purple-50'
                    : 'border-zinc-800 hover:bg-zinc-900'
                }`}>
                  <div className="flex items-center gap-3">
                    <Shield className={`w-5 h-5 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
                    <div className="text-left">
                      <div className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                        Two-Factor Authentication
                      </div>
                      <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                        Enable 2FA for extra security
                      </div>
                    </div>
                  </div>
                  <span className="text-purple-600">→</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
