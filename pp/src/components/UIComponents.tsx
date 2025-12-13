import { ArrowRight, Sparkles, Download, Play, Bell, Settings, User, Search } from 'lucide-react';

export function UIComponents() {
  return (
    <div className="space-y-8">
      {/* Buttons */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Buttons</h3>
        <div className="flex flex-wrap gap-4">
          {/* Primary Button */}
          <button className="px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 rounded-xl text-white shadow-lg shadow-purple-500/30 hover:shadow-xl hover:shadow-purple-500/40 hover:scale-105 transition-all duration-200 flex items-center gap-2">
            <Sparkles className="w-4 h-4" />
            Primary Action
          </button>

          {/* Secondary Button */}
          <button className="px-6 py-3 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl text-purple-100 hover:bg-white/20 transition-all duration-200 flex items-center gap-2">
            <Download className="w-4 h-4" />
            Secondary
          </button>

          {/* Ghost Button */}
          <button className="px-6 py-3 text-purple-300 hover:bg-white/10 rounded-xl transition-all duration-200 flex items-center gap-2">
            Learn More
            <ArrowRight className="w-4 h-4" />
          </button>

          {/* Icon Button */}
          <button className="w-12 h-12 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl text-purple-200 hover:bg-white/20 hover:scale-105 transition-all duration-200 flex items-center justify-center">
            <Play className="w-5 h-5" />
          </button>

          {/* Disabled Button */}
          <button disabled className="px-6 py-3 backdrop-blur-lg bg-white/5 border border-white/10 rounded-xl text-purple-400 opacity-50 cursor-not-allowed">
            Disabled
          </button>
        </div>
      </div>

      {/* Cards */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Cards</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Feature Card */}
          <div className="backdrop-blur-xl bg-white/5 rounded-xl p-6 border border-white/10 hover:border-purple-400/30 hover:bg-white/10 transition-all duration-300 group">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg mb-4 flex items-center justify-center shadow-lg shadow-purple-500/30">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <h4 className="text-lg mb-2 text-purple-100">AI-Powered</h4>
            <p className="text-sm text-purple-300">Advanced machine learning algorithms for intelligent automation</p>
          </div>

          {/* Stat Card */}
          <div className="backdrop-blur-xl bg-gradient-to-br from-purple-900/20 to-purple-800/20 rounded-xl p-6 border border-purple-400/20">
            <p className="text-sm text-purple-300 mb-1">Total Users</p>
            <p className="text-4xl text-purple-100 mb-2">24,567</p>
            <p className="text-sm text-purple-400">↑ 12% from last month</p>
          </div>

          {/* Interactive Card */}
          <div className="backdrop-blur-xl bg-white/5 rounded-xl p-6 border border-white/10 hover:shadow-2xl hover:shadow-purple-500/20 hover:-translate-y-1 transition-all duration-300 cursor-pointer">
            <div className="flex items-start justify-between mb-4">
              <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
                <Settings className="w-5 h-5 text-white" />
              </div>
              <span className="text-xs text-purple-400 backdrop-blur-lg bg-purple-500/20 px-2 py-1 rounded">New</span>
            </div>
            <h4 className="text-lg mb-2 text-purple-100">Settings</h4>
            <p className="text-sm text-purple-300">Configure your preferences</p>
          </div>
        </div>
      </div>

      {/* Input Fields */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Input Fields</h3>
        <div className="space-y-4 max-w-md">
          {/* Standard Input */}
          <div>
            <label className="block text-sm text-purple-200 mb-2">Email Address</label>
            <input
              type="email"
              placeholder="you@example.com"
              className="w-full px-4 py-3 backdrop-blur-lg bg-white/5 border border-white/20 rounded-xl text-purple-100 placeholder-purple-400 focus:outline-none focus:border-purple-400 focus:bg-white/10 transition-all"
            />
          </div>

          {/* Input with Icon */}
          <div>
            <label className="block text-sm text-purple-200 mb-2">Search</label>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-purple-400" />
              <input
                type="text"
                placeholder="Search for anything..."
                className="w-full pl-12 pr-4 py-3 backdrop-blur-lg bg-white/5 border border-white/20 rounded-xl text-purple-100 placeholder-purple-400 focus:outline-none focus:border-purple-400 focus:bg-white/10 transition-all"
              />
            </div>
          </div>

          {/* Textarea */}
          <div>
            <label className="block text-sm text-purple-200 mb-2">Message</label>
            <textarea
              placeholder="Type your message here..."
              rows={4}
              className="w-full px-4 py-3 backdrop-blur-lg bg-white/5 border border-white/20 rounded-xl text-purple-100 placeholder-purple-400 focus:outline-none focus:border-purple-400 focus:bg-white/10 transition-all resize-none"
            ></textarea>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Navigation</h3>
        
        {/* Horizontal Nav */}
        <div className="backdrop-blur-xl bg-white/5 rounded-xl p-2 border border-white/10 flex gap-2 mb-6">
          <button className="px-4 py-2 bg-purple-600 rounded-lg text-white">Dashboard</button>
          <button className="px-4 py-2 text-purple-200 hover:bg-white/10 rounded-lg transition-colors">Analytics</button>
          <button className="px-4 py-2 text-purple-200 hover:bg-white/10 rounded-lg transition-colors">Settings</button>
          <button className="px-4 py-2 text-purple-200 hover:bg-white/10 rounded-lg transition-colors">Profile</button>
        </div>

        {/* Icon Nav */}
        <div className="flex gap-2">
          <button className="w-12 h-12 backdrop-blur-lg bg-purple-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-purple-500/30">
            <User className="w-5 h-5" />
          </button>
          <button className="w-12 h-12 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl flex items-center justify-center text-purple-200 hover:bg-white/20 transition-all">
            <Bell className="w-5 h-5" />
          </button>
          <button className="w-12 h-12 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl flex items-center justify-center text-purple-200 hover:bg-white/20 transition-all">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Badges & Tags */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Badges & Tags</h3>
        <div className="flex flex-wrap gap-3">
          <span className="px-3 py-1 bg-gradient-to-r from-purple-600 to-purple-700 rounded-full text-sm text-white shadow-lg shadow-purple-500/30">
            Premium
          </span>
          <span className="px-3 py-1 backdrop-blur-lg bg-purple-500/20 border border-purple-400/30 rounded-full text-sm text-purple-200">
            Beta
          </span>
          <span className="px-3 py-1 backdrop-blur-lg bg-white/10 border border-white/20 rounded-full text-sm text-purple-300">
            Active
          </span>
          <span className="px-3 py-1 backdrop-blur-lg bg-green-500/20 border border-green-400/30 rounded-full text-sm text-green-300">
            Online
          </span>
          <span className="px-3 py-1 backdrop-blur-lg bg-red-500/20 border border-red-400/30 rounded-full text-sm text-red-300">
            Alert
          </span>
        </div>
      </div>

      {/* Notification/Alert */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Notifications</h3>
        <div className="space-y-4">
          <div className="backdrop-blur-xl bg-purple-500/10 border border-purple-400/30 rounded-xl p-4 flex items-start gap-3">
            <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-purple-100 mb-1">New Feature Available</h4>
              <p className="text-sm text-purple-300">Check out the latest AI-powered analytics tools in your dashboard.</p>
            </div>
          </div>

          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-xl p-4 flex items-start gap-3">
            <div className="w-10 h-10 backdrop-blur-lg bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0">
              <Bell className="w-5 h-5 text-purple-300" />
            </div>
            <div className="flex-1">
              <h4 className="text-purple-100 mb-1">System Update</h4>
              <p className="text-sm text-purple-300">Your system will be updated tonight at 2 AM EST.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
