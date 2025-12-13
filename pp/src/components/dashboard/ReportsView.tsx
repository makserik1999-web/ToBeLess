import { FileText, Download, Calendar, Filter, TrendingUp, Plus } from 'lucide-react';
import { useState } from 'react';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';

export function ReportsView() {
  const { theme } = useTheme();
  const [reportType, setReportType] = useState('all');

  const reports = [
    {
      id: 1,
      title: 'Monthly Security Summary - November 2024',
      type: 'monthly',
      date: '2024-11-30',
      size: '2.4 MB',
      description: 'Comprehensive overview of all security incidents and detections for November',
      downloads: 45
    },
    {
      id: 2,
      title: 'Incident Response Analysis - Q4 2024',
      type: 'quarterly',
      date: '2024-12-01',
      size: '5.1 MB',
      description: 'Detailed analysis of incident response times and effectiveness',
      downloads: 32
    },
    {
      id: 3,
      title: 'Weekly Detection Report - Week 49',
      type: 'weekly',
      date: '2024-12-08',
      size: '856 KB',
      description: 'All AI detections from December 2-8, 2024',
      downloads: 78
    },
    {
      id: 4,
      title: 'Annual Security Review 2024',
      type: 'annual',
      date: '2024-12-10',
      size: '12.3 MB',
      description: 'Complete security review and statistics for 2024',
      downloads: 12
    },
    {
      id: 5,
      title: 'Custom Report - Building A Analysis',
      type: 'custom',
      date: '2024-12-09',
      size: '1.8 MB',
      description: 'Focused analysis on Building A security metrics',
      downloads: 23
    },
  ];

  const filteredReports = reports.filter(report => {
    if (reportType !== 'all' && report.type !== reportType) return false;
    return true;
  });

  const reportTypes = [
    { id: 'all', label: 'All Reports', count: reports.length },
    { id: 'weekly', label: 'Weekly', count: reports.filter(r => r.type === 'weekly').length },
    { id: 'monthly', label: 'Monthly', count: reports.filter(r => r.type === 'monthly').length },
    { id: 'quarterly', label: 'Quarterly', count: reports.filter(r => r.type === 'quarterly').length },
    { id: 'annual', label: 'Annual', count: reports.filter(r => r.type === 'annual').length },
    { id: 'custom', label: 'Custom', count: reports.filter(r => r.type === 'custom').length },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Reports & Analytics
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Generate and download security reports
          </p>
        </div>
        <button className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-all shadow-lg shadow-purple-500/25">
          <Plus className="w-5 h-5" />
          Create Custom Report
        </button>
      </div>

      {/* Report Type Filter */}
      <div className="flex flex-wrap gap-2">
        {reportTypes.map((type) => (
          <button
            key={type.id}
            onClick={() => setReportType(type.id)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              reportType === type.id
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                : theme === 'light'
                ? 'bg-white border border-purple-200 text-zinc-700 hover:bg-purple-50'
                : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:bg-zinc-800'
            }`}
          >
            {type.label}
            <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
              reportType === type.id
                ? 'bg-purple-700'
                : theme === 'light'
                ? 'bg-purple-100 text-purple-700'
                : 'bg-zinc-800 text-zinc-400'
            }`}>
              {type.count}
            </span>
          </button>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <FileText className={`w-8 h-8 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
            <TrendingUp className="w-5 h-5 text-emerald-500" />
          </div>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {reports.length}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Total Reports
          </div>
        </div>

        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <Download className={`w-8 h-8 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
          </div>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            190
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Downloads This Month
          </div>
        </div>

        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <Calendar className={`w-8 h-8 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
          </div>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            3
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Scheduled Reports
          </div>
        </div>
      </div>

      {/* Reports List */}
      <div className="grid grid-cols-1 gap-4">
        {filteredReports.map((report, index) => (
          <motion.div
            key={report.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`backdrop-blur-xl border rounded-2xl p-6 transition-all group ${
              theme === 'light'
                ? 'bg-white border-purple-200 hover:shadow-lg hover:shadow-purple-100'
                : 'bg-zinc-950 border-zinc-900 hover:border-zinc-800'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4 flex-1">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
                  theme === 'light' ? 'bg-purple-100' : 'bg-zinc-900'
                }`}>
                  <FileText className={`w-6 h-6 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
                </div>
                <div className="flex-1">
                  <h3 className={`text-lg font-semibold mb-2 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                    {report.title}
                  </h3>
                  <p className={`text-sm font-medium mb-3 ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                    {report.description}
                  </p>
                  <div className={`flex items-center gap-4 text-sm font-medium ${
                    theme === 'light' ? 'text-zinc-500' : 'text-zinc-500'
                  }`}>
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      {report.date}
                    </div>
                    <div>Size: {report.size}</div>
                    <div>{report.downloads} downloads</div>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                  theme === 'light'
                    ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                    : 'bg-purple-900 text-purple-300 hover:bg-purple-800'
                }`}>
                  View
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-xl text-sm font-medium transition-all">
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Report Templates */}
      <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
        theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
      }`}>
        <h2 className={`text-xl font-semibold mb-4 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
          Available Report Templates
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { name: 'Incident Summary', description: 'Overview of all incidents in a time period' },
            { name: 'Detection Analytics', description: 'Detailed analysis of AI detections and patterns' },
            { name: 'Response Performance', description: 'Team response times and effectiveness metrics' },
          ].map((template, index) => (
            <div
              key={index}
              className={`p-4 rounded-xl border transition-all cursor-pointer ${
                theme === 'light'
                  ? 'border-purple-200 hover:bg-purple-50'
                  : 'border-zinc-800 hover:bg-zinc-900'
              }`}
            >
              <h3 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                {template.name}
              </h3>
              <p className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                {template.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
