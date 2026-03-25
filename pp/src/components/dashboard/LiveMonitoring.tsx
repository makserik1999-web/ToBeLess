import { Camera, Upload, Radio, Video, Shield, Zap, Eye } from 'lucide-react';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';
import { useState, useEffect, useCallback } from 'react';
import { AddCameraModal } from './AddCameraModal';
import { LiveDetectionView } from './LiveDetectionView';
import { detectionApi, DetectionMode, PrivacyMode } from '../../api/detection';

const DETECTION_MODE_OPTIONS: { value: DetectionMode; label: string; note?: string }[] = [
  { value: 'fight',  label: '🥊 Fight Detection' },
  { value: 'weapon', label: '🔫 Weapon Detection' },
  { value: 'scream', label: '🔊 Scream Detection', note: 'Requires video upload' },
];

const PRIVACY_MODE_OPTIONS: { value: PrivacyMode; label: string }[] = [
  { value: 'off',         label: '🚫 Off' },
  { value: 'recognition', label: '👤 Face Recognition' },
  { value: 'blur',        label: '👁️ Face Blur' },
];

export function LiveMonitoring() {
  const { theme } = useTheme();
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDetectionView, setShowDetectionView] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [pendingVideoFilename, setPendingVideoFilename] = useState<string | undefined>(undefined);

  const [detectionMode, setDetectionMode] = useState<DetectionMode>('fight');
  const [privacyMode, setPrivacyMode] = useState<PrivacyMode>('off');
  const [events, setEvents] = useState<any[]>([]);

  // Load current modes from backend on mount
  useEffect(() => {
    const loadModes = async () => {
      try {
        const [dm, pm] = await Promise.all([
          detectionApi.getDetectionMode(),
          detectionApi.getPrivacyMode(),
        ]);
        if (dm.success && dm.mode) setDetectionMode(dm.mode as DetectionMode);
        if (pm.success && pm.mode) setPrivacyMode(pm.mode as PrivacyMode);
      } catch { /* ignore if backend not running */ }
    };
    loadModes();
  }, []);

  const loadEvents = useCallback(async () => {
    try {
      const data = await detectionApi.getEvents();
      if (data.success) setEvents([...data.events].reverse().slice(0, 40));
    } catch { /* ignore */ }
  }, []);

  // Auto-refresh events every 15 s
  useEffect(() => {
    void loadEvents();
    const id = setInterval(() => { void loadEvents(); }, 15000);
    return () => clearInterval(id);
  }, [loadEvents]);

  const handleDetectionModeChange = async (mode: DetectionMode) => {
    try {
      const result = await detectionApi.setDetectionMode(mode);
      if (result.success) setDetectionMode(mode);
    } catch { /* ignore */ }
  };

  const handlePrivacyModeChange = async (mode: PrivacyMode) => {
    try {
      const result = await detectionApi.setPrivacyMode(mode);
      if (result.success) setPrivacyMode(mode);
    } catch { /* ignore */ }
  };

  const handleGenerateReport = async (format: 'pdf' | 'excel' | 'json') => {
    try {
      const result = await detectionApi.generateReport(format);
      if (result.success && result.filename) {
        window.location.href = detectionApi.getReportDownloadUrl(result.filename);
      }
    } catch { /* ignore */ }
  };

  const handleClearEvents = async () => {
    await detectionApi.clearEvents();
    setEvents([]);
  };

  const sourceOptions = [
    { id: 'webcam', icon: Camera, title: 'Webcam',       description: 'Use your device camera for live detection' },
    { id: 'file',   icon: Upload, title: 'Video File',   description: 'Upload a video file for analysis' },
    { id: 'rtsp',   icon: Radio,  title: 'RTSP Stream',  description: 'Connect to IP camera or RTSP stream' },
  ];

  const features = [
    { icon: Shield, title: 'Fight Detection',  description: 'AI-powered violence detection using pose estimation' },
    { icon: Eye,    title: 'Face Recognition', description: 'Identify known individuals in real-time' },
    { icon: Zap,    title: 'Real-time Alerts', description: 'Instant Telegram notifications on detection' },
  ];

  return (
    <div className="flex gap-6 items-start">
      {/* ── Left: main content ── */}
      <div className="flex-1 min-w-0 space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Live Monitoring
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Start real-time violence detection
          </p>
        </motion.div>

        {/* Empty State / Start */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className={`rounded-3xl border-2 border-dashed p-12 text-center ${
            theme === 'light' ? 'border-purple-300 bg-purple-50/50' : 'border-zinc-700 bg-zinc-900/50'
          }`}
        >
          <motion.div
            className={`w-24 h-24 mx-auto mb-6 rounded-3xl flex items-center justify-center ${
              theme === 'light' ? 'bg-purple-100' : 'bg-purple-900/30'
            }`}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Video className={`w-12 h-12 ${theme === 'light' ? 'text-purple-600' : 'text-purple-400'}`} />
          </motion.div>

          <h2 className={`text-2xl font-semibold mb-3 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            No Active Stream
          </h2>
          <p className={`mb-8 max-w-md mx-auto ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Connect a video source to start real-time violence detection. Choose from webcam, video file, or RTSP stream.
          </p>

          <motion.button
            onClick={() => setShowAddModal(true)}
            className="px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-2xl transition-all font-semibold text-lg shadow-lg shadow-purple-500/25"
            whileHover={{ scale: 1.02, boxShadow: '0 20px 40px rgba(168, 85, 247, 0.3)' }}
            whileTap={{ scale: 0.98 }}
          >
            <Camera className="w-5 h-5 inline mr-2" />
            Start Detection
          </motion.button>
        </motion.div>

        {/* Source Options */}
        <div>
          <h3 className={`text-lg font-semibold mb-4 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Available Sources
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {sourceOptions.map((source, index) => (
              <motion.div
                key={source.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 + index * 0.1 }}
                onClick={() => setShowAddModal(true)}
                className={`p-6 rounded-2xl border cursor-pointer transition-all group ${
                  theme === 'light'
                    ? 'bg-white border-purple-200 hover:border-purple-400 hover:shadow-lg hover:shadow-purple-100'
                    : 'bg-zinc-950 border-zinc-800 hover:border-purple-600 hover:shadow-lg hover:shadow-purple-900/20'
                }`}
              >
                <div className={`w-12 h-12 rounded-xl mb-4 flex items-center justify-center transition-colors ${
                  theme === 'light' ? 'bg-purple-100 group-hover:bg-purple-200' : 'bg-purple-900/30 group-hover:bg-purple-900/50'
                }`}>
                  <source.icon className={`w-6 h-6 ${theme === 'light' ? 'text-purple-600' : 'text-purple-400'}`} />
                </div>
                <h4 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                  {source.title}
                </h4>
                <p className={`text-sm ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                  {source.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Features */}
        <div>
          <h3 className={`text-lg font-semibold mb-4 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Detection Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.4 + index * 0.1 }}
                className={`p-6 rounded-2xl border ${
                  theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-800'
                }`}
              >
                <div className={`w-10 h-10 rounded-lg mb-3 flex items-center justify-center ${
                  theme === 'light' ? 'bg-emerald-100' : 'bg-emerald-900/30'
                }`}>
                  <feature.icon className={`w-5 h-5 ${theme === 'light' ? 'text-emerald-600' : 'text-emerald-400'}`} />
                </div>
                <h4 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                  {feature.title}
                </h4>
                <p className={`text-sm ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Right: permanent control sidebar ── */}
      <div className="w-72 flex-shrink-0 space-y-4 sticky top-6">
        {/* Detection Mode */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
          className={`p-4 rounded-2xl border-2 ${
            theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-800'
          }`}
        >
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Detection Mode
          </h3>
          <div className="flex flex-col gap-1">
            {DETECTION_MODE_OPTIONS.map(({ value, label, note }) => (
              <button
                key={value}
                onClick={() => handleDetectionModeChange(value)}
                className={`px-3 py-2 rounded-xl text-sm font-medium text-left transition-all ${
                  detectionMode === value
                    ? 'bg-purple-600 text-white'
                    : theme === 'light'
                    ? 'text-zinc-600 hover:bg-purple-50'
                    : 'text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                {label}
                {note && (
                  <span className={`block text-xs font-normal mt-0.5 ${
                    detectionMode === value ? 'text-white/70' : 'opacity-50'
                  }`}>
                    {note}
                  </span>
                )}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Privacy Mode */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.25 }}
          className={`p-4 rounded-2xl border-2 ${
            theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-800'
          }`}
        >
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Privacy Mode
          </h3>
          <div className="flex flex-col gap-1">
            {PRIVACY_MODE_OPTIONS.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => handlePrivacyModeChange(value)}
                className={`px-3 py-2 rounded-xl text-sm font-medium text-left transition-all ${
                  privacyMode === value
                    ? 'bg-purple-600 text-white'
                    : theme === 'light'
                    ? 'text-zinc-600 hover:bg-purple-50'
                    : 'text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Reports & Recent Events */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.35 }}
          className={`p-4 rounded-2xl border-2 ${
            theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-800'
          }`}
        >
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
            Reports
          </h3>
          <div className="flex flex-col gap-1 mb-4">
            {(['pdf', 'excel', 'json'] as const).map(fmt => (
              <button
                key={fmt}
                onClick={() => handleGenerateReport(fmt)}
                className={`px-3 py-2 rounded-xl text-sm font-medium text-left transition-all ${
                  theme === 'light'
                    ? 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                    : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                }`}
              >
                ↓ Download {fmt.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Recent Events */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              Recent Events
            </span>
            <div className="flex gap-1">
              <button
                onClick={() => { void loadEvents(); }}
                className={`text-xs px-2 py-1 rounded-lg transition-colors ${
                  theme === 'light' ? 'bg-purple-50 text-zinc-600 hover:bg-purple-100' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                }`}
              >↻</button>
              <button
                onClick={handleClearEvents}
                className={`text-xs px-2 py-1 rounded-lg transition-colors ${
                  theme === 'light' ? 'bg-red-50 text-red-500 hover:bg-red-100' : 'bg-zinc-800 text-red-400 hover:bg-zinc-700'
                }`}
              >Clear</button>
            </div>
          </div>

          <div className="max-h-48 overflow-y-auto space-y-0.5">
            {events.length === 0 ? (
              <p className="text-xs text-zinc-500 text-center py-4">No events yet</p>
            ) : (
              events.map((ev, i) => {
                const icon: Record<string, string> = { fight: '🚨', weapon: '🔫', scream: '🔊', fall: '🫸' };
                const color: Record<string, string> = {
                  fight: 'text-red-400', weapon: 'text-orange-400',
                  scream: 'text-yellow-400', fall: 'text-blue-400',
                };
                const ts = ev.timestamp ? String(ev.timestamp).slice(11, 19) : '';
                const conf = ev.confidence != null ? ` ${Math.round(Number(ev.confidence) * 100)}%` : '';
                return (
                  <div key={i} className="flex items-center gap-2 text-xs py-1.5 border-b border-zinc-800/50 last:border-0">
                    <span>{icon[ev.type] ?? '⚠️'}</span>
                    <span className="text-zinc-500 min-w-[50px] tabular-nums">{ts}</span>
                    <span className={`font-medium flex-1 ${color[ev.type] ?? 'text-zinc-300'}`}>
                      {ev.type}{conf}
                    </span>
                    {ev.details && (
                      <span className="text-zinc-600 truncate max-w-[72px] text-[10px]">{ev.details}</span>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </motion.div>
      </div>

      {/* Add Camera Modal */}
      <AddCameraModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSuccess={() => {
          setStreamKey(Date.now());
        }}
        onStartDetection={(videoFilename) => {
          setPendingVideoFilename(videoFilename);
          setShowDetectionView(true);
        }}
      />

      {/* Live Detection View */}
      <LiveDetectionView
        isOpen={showDetectionView}
        onClose={() => setShowDetectionView(false)}
        streamKey={streamKey}
        initialVideoFilename={pendingVideoFilename}
      />
    </div>
  );
}
