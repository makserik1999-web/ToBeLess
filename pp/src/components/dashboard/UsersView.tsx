import { Users, Search, Plus, Mail, Shield, Calendar, MoreVertical, Edit, Trash2, UserCheck } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';

export function UsersView() {
  const { theme } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');

  const users = [
    {
      id: 1,
      name: 'Айгерім Нұрланова',
      email: 'aigerim.nurlanova@tobeles.ai',
      role: 'admin',
      department: 'Security Operations',
      lastActive: '2 minutes ago',
      status: 'active',
      joinDate: '2023-01-15',
      permissions: ['Full Access', 'User Management', 'System Settings']
    },
    {
      id: 2,
      name: 'Ерлан Жұмабаев',
      email: 'erlan.zhumabaev@tobeles.ai',
      role: 'operator',
      department: 'Security Team Alpha',
      lastActive: '15 minutes ago',
      status: 'active',
      joinDate: '2023-03-22',
      permissions: ['View Alerts', 'Respond to Incidents', 'View Reports']
    },
    {
      id: 3,
      name: 'Дана Сейітқызы',
      email: 'dana.seitkyzy@tobeles.ai',
      role: 'operator',
      department: 'Security Team Bravo',
      lastActive: '1 hour ago',
      status: 'active',
      joinDate: '2023-02-10',
      permissions: ['View Alerts', 'Respond to Incidents', 'View Reports']
    },
    {
      id: 4,
      name: 'Нұрлан Асқаров',
      email: 'nurlan.askarov@tobeles.ai',
      role: 'viewer',
      department: 'Management',
      lastActive: '3 hours ago',
      status: 'active',
      joinDate: '2023-04-05',
      permissions: ['View Alerts', 'View Reports']
    },
    {
      id: 5,
      name: 'Гүлнар Әбдіқадырова',
      email: 'gulnar.abdikadyrova@tobeles.ai',
      role: 'operator',
      department: 'Security Team Charlie',
      lastActive: '2 days ago',
      status: 'inactive',
      joinDate: '2023-05-18',
      permissions: ['View Alerts', 'Respond to Incidents']
    },
    {
      id: 6,
      name: 'Асан Төлеуов',
      email: 'asan.toleuov@tobeles.ai',
      role: 'operator',
      department: 'Security Team Alpha',
      lastActive: '30 minutes ago',
      status: 'active',
      joinDate: '2023-06-12',
      permissions: ['View Alerts', 'Respond to Incidents', 'View Reports']
    },
    {
      id: 7,
      name: 'Мәдина Қайратқызы',
      email: 'madina.kairatkyzy@tobeles.ai',
      role: 'admin',
      department: 'Security Operations',
      lastActive: '5 minutes ago',
      status: 'active',
      joinDate: '2023-01-20',
      permissions: ['Full Access', 'User Management', 'System Settings']
    },
  ];

  const filteredUsers = users.filter(user => {
    if (roleFilter !== 'all' && user.role !== roleFilter) return false;
    if (searchQuery && !user.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !user.email.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const roleCounts = {
    all: users.length,
    admin: users.filter(u => u.role === 'admin').length,
    operator: users.filter(u => u.role === 'operator').length,
    viewer: users.filter(u => u.role === 'viewer').length,
  };

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-purple-500/20 text-purple-600';
      case 'operator': return 'bg-blue-500/20 text-blue-600';
      case 'viewer': return 'bg-zinc-500/20 text-zinc-600';
      default: return 'bg-zinc-500/20 text-zinc-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            User Management
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Manage team members and permissions
          </p>
        </div>
        <button className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-all shadow-lg shadow-purple-500/25">
          <Plus className="w-5 h-5" />
          Add User
        </button>
      </div>

      {/* Search and Role Filter */}
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
              placeholder="Search users by name or email..."
              className={`w-full pl-12 pr-4 py-3 border rounded-xl focus:outline-none transition-all ${
                theme === 'light'
                  ? 'bg-purple-50/50 border-purple-200 text-zinc-900 placeholder-zinc-500 focus:border-purple-400 focus:bg-white'
                  : 'bg-zinc-900 border-zinc-800 text-white placeholder-zinc-500 focus:border-purple-500 focus:bg-zinc-900/80'
              }`}
            />
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(roleCounts).map(([role, count]) => (
            <button
              key={role}
              onClick={() => setRoleFilter(role)}
              className={`px-4 py-3 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${
                roleFilter === role
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                  : theme === 'light'
                  ? 'bg-white border border-purple-200 text-zinc-700 hover:bg-purple-50'
                  : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:bg-zinc-800'
              }`}
            >
              {role.charAt(0).toUpperCase() + role.slice(1)}
              <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                roleFilter === role
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

      {/* Users Table */}
      <div className={`backdrop-blur-xl border rounded-2xl overflow-hidden ${
        theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
      }`}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className={`border-b ${theme === 'light' ? 'border-purple-200 bg-purple-50' : 'border-zinc-900 bg-zinc-900/50'}`}>
              <tr>
                <th className={`px-6 py-4 text-left text-xs font-semibold uppercase tracking-wide ${
                  theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'
                }`}>User</th>
                <th className={`px-6 py-4 text-left text-xs font-semibold uppercase tracking-wide ${
                  theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'
                }`}>Role</th>
                <th className={`px-6 py-4 text-left text-xs font-semibold uppercase tracking-wide ${
                  theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'
                }`}>Department</th>
                <th className={`px-6 py-4 text-left text-xs font-semibold uppercase tracking-wide ${
                  theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'
                }`}>Last Active</th>
                <th className={`px-6 py-4 text-left text-xs font-semibold uppercase tracking-wide ${
                  theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'
                }`}>Status</th>
                <th className={`px-6 py-4 text-right text-xs font-semibold uppercase tracking-wide ${
                  theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'
                }`}>Actions</th>
              </tr>
            </thead>
            <tbody className={`divide-y ${theme === 'light' ? 'divide-purple-100' : 'divide-zinc-900'}`}>
              {filteredUsers.map((user, index) => (
                <motion.tr
                  key={user.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`transition-colors ${
                    theme === 'light' ? 'hover:bg-purple-50' : 'hover:bg-zinc-900/50'
                  }`}
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                        <span className="text-white font-semibold text-sm">{user.name.charAt(0)}</span>
                      </div>
                      <div>
                        <div className={`font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                          {user.name}
                        </div>
                        <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-500'}`}>
                          {user.email}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 rounded-lg text-xs font-semibold uppercase ${getRoleBadgeColor(user.role)}`}>
                      {user.role}
                    </span>
                  </td>
                  <td className={`px-6 py-4 font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                    {user.department}
                  </td>
                  <td className={`px-6 py-4 text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                    {user.lastActive}
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold ${
                      user.status === 'active'
                        ? 'bg-emerald-500/20 text-emerald-600'
                        : 'bg-zinc-500/20 text-zinc-600'
                    }`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${
                        user.status === 'active' ? 'bg-emerald-500' : 'bg-zinc-500'
                      }`}></span>
                      {user.status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center justify-end gap-2">
                      <button className={`p-2 rounded-lg transition-colors ${
                        theme === 'light'
                          ? 'hover:bg-purple-100 text-zinc-600'
                          : 'hover:bg-zinc-800 text-zinc-400'
                      }`}>
                        <Edit className="w-4 h-4" />
                      </button>
                      <button className={`p-2 rounded-lg transition-colors ${
                        theme === 'light'
                          ? 'hover:bg-red-50 text-red-600'
                          : 'hover:bg-red-500/10 text-red-500'
                      }`}>
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {users.length}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Total Users
          </div>
        </div>
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {users.filter(u => u.status === 'active').length}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Active Users
          </div>
        </div>
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {users.filter(u => u.role === 'admin').length}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Administrators
          </div>
        </div>
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {users.filter(u => u.role === 'operator').length}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Operators
          </div>
        </div>
      </div>
    </div>
  );
}
