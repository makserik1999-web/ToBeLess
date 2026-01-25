import { FileText, Download, Calendar, FolderOpen, Loader2, RefreshCw } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';
import { detectionApi } from '../../api/detection';
import { ReportInfo } from '../../types/analytics';
import { toast } from 'sonner';

export function ReportsView() {
  const { theme } = useTheme();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatingFormat, setGeneratingFormat] = useState<string | null>(null);
  const [realReports, setRealReports] = useState<ReportInfo[]>([]);
  const [isLoadingReports, setIsLoadingReports] = useState(true);

  // Load real reports on mount
  const loadReports = async () => {
    setIsLoadingReports(true);
    try {
      const result = await detectionApi.listReports();
      if (result.success) {
        setRealReports(result.reports);
      }
    } catch (err) {
      // Backend might not be running
      setRealReports([]);
    } finally {
      setIsLoadingReports(false);
    }
  };

  useEffect(() => {
    loadReports();
  }, []);

  // Generate report function
  const handleGenerateReport = async (format: 'pdf' | 'excel' | 'json') => {
    setIsGenerating(true);
    setGeneratingFormat(format);
    try {
      const result = await detectionApi.generateReport(format);
      if (result.success && result.filename) {
        toast.success('Report generated', { description: `${result.filename} is ready for download` });
        // Download the file
        window.open(detectionApi.getReportDownloadUrl(result.filename), '_blank');
        // Refresh the reports list
        await loadReports();
      } else {
        toast.error('Generation failed', { description: result.error || 'Unknown error' });
      }
    } catch (err) {
      toast.error('Connection error', { description: 'Make sure the backend is running (python app.py)' });
    } finally {
      setIsGenerating(false);
      setGeneratingFormat(null);
    }
  };

  // Handle download of existing report
  const handleDownloadReport = (filename: string) => {
    window.open(detectionApi.getReportDownloadUrl(filename), '_blank');
  };

  const reportTemplates = [
    { name: 'PDF Report', format: 'pdf' as const, color: 'red', description: 'Professional PDF with charts and statistics' },
    { name: 'Excel Report', format: 'excel' as const, color: 'green', description: 'Spreadsheet with raw data for analysis' },
    { name: 'JSON Export', format: 'json' as const, color: 'blue', description: 'Machine-readable data export' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Reports
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Generate and download detection reports
          </p>
        </div>
        <button
          onClick={loadReports}
          disabled={isLoadingReports}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-colors ${
            theme === 'light'
              ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
              : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-800'
          }`}
        >
          <RefreshCw className={`w-4 h-4 ${isLoadingReports ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Generate Report Section */}
      <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
        theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
      }`}>
        <h2 className={`text-xl font-semibold mb-4 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
          Generate New Report
        </h2>
        <p className={`mb-6 ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
          Generate a report from the current detection session. Make sure a video stream is running for best results.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {reportTemplates.map((template) => (
            <motion.button
              key={template.format}
              onClick={() => handleGenerateReport(template.format)}
              disabled={isGenerating}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`p-6 rounded-xl border text-left transition-all ${
                theme === 'light'
                  ? 'border-purple-200 hover:border-purple-400 hover:shadow-lg'
                  : 'border-zinc-800 hover:border-zinc-700'
              } ${isGenerating ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className={`w-12 h-12 rounded-xl mb-4 flex items-center justify-center ${
                template.color === 'red' ? 'bg-red-500/10' :
                template.color === 'green' ? 'bg-green-500/10' : 'bg-blue-500/10'
              }`}>
                {generatingFormat === template.format ? (
                  <Loader2 className={`w-6 h-6 animate-spin ${
                    template.color === 'red' ? 'text-red-500' :
                    template.color === 'green' ? 'text-green-500' : 'text-blue-500'
                  }`} />
                ) : (
                  <FileText className={`w-6 h-6 ${
                    template.color === 'red' ? 'text-red-500' :
                    template.color === 'green' ? 'text-green-500' : 'text-blue-500'
                  }`} />
                )}
              </div>
              <h3 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                {template.name}
              </h3>
              <p className={`text-sm ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                {template.description}
              </p>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <FileText className={`w-8 h-8 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
          </div>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {realReports.length}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Generated Reports
          </div>
        </div>

        <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
          theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <Calendar className={`w-8 h-8 ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
          </div>
          <div className={`text-3xl font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            {realReports.length > 0 ? new Date(realReports[0].created).toLocaleDateString() : '-'}
          </div>
          <div className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Last Report Generated
          </div>
        </div>
      </div>

      {/* Generated Reports List */}
      <div className={`backdrop-blur-xl border rounded-2xl p-6 ${
        theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-900'
      }`}>
        <h2 className={`text-xl font-semibold mb-4 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
          Generated Reports
        </h2>

        {isLoadingReports ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className={`w-8 h-8 animate-spin ${theme === 'light' ? 'text-purple-600' : 'text-purple-500'}`} />
          </div>
        ) : realReports.length > 0 ? (
          <div className="space-y-3">
            {realReports.map((report, index) => (
              <motion.div
                key={report.filename}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`flex items-center justify-between p-4 rounded-xl border transition-all ${
                  theme === 'light'
                    ? 'border-purple-200 hover:bg-purple-50'
                    : 'border-zinc-800 hover:bg-zinc-900'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    report.type === 'PDF' ? 'bg-red-500/10' :
                    report.type === 'XLSX' ? 'bg-green-500/10' : 'bg-blue-500/10'
                  }`}>
                    <FileText className={`w-5 h-5 ${
                      report.type === 'PDF' ? 'text-red-500' :
                      report.type === 'XLSX' ? 'text-green-500' : 'text-blue-500'
                    }`} />
                  </div>
                  <div>
                    <div className={`font-medium ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                      {report.filename}
                    </div>
                    <div className={`text-sm ${theme === 'light' ? 'text-zinc-500' : 'text-zinc-500'}`}>
                      {new Date(report.created).toLocaleString()} • {(report.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleDownloadReport(report.filename)}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-all"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <FolderOpen className={`w-16 h-16 mb-4 ${theme === 'light' ? 'text-purple-200' : 'text-zinc-700'}`} />
            <h3 className={`text-lg font-semibold mb-2 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
              No Reports Yet
            </h3>
            <p className={`max-w-md ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
              Start a detection stream and generate your first report using the buttons above.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
