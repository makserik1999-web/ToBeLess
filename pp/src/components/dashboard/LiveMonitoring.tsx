import { Camera, Upload, Radio, Video, Shield, Zap, Eye } from 'lucide-react';
import { useTheme } from '../Dashboard';
import { motion } from 'motion/react';
import { useState } from 'react';
import { AddCameraModal } from './AddCameraModal';
import { LiveDetectionView } from './LiveDetectionView';

export function LiveMonitoring() {
  const { theme } = useTheme();
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDetectionView, setShowDetectionView] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now()); // Force video refresh on new stream

  const sourceOptions = [
    {
      id: 'webcam',
      icon: Camera,
      title: 'Webcam',
      description: 'Use your device camera for live detection',
    },
    {
      id: 'file',
      icon: Upload,
      title: 'Video File',
      description: 'Upload a video file for analysis',
    },
    {
      id: 'rtsp',
      icon: Radio,
      title: 'RTSP Stream',
      description: 'Connect to IP camera or RTSP stream',
    },
  ];

  const features = [
    {
      icon: Shield,
      title: 'Fight Detection',
      description: 'AI-powered violence detection using pose estimation',
    },
    {
      icon: Eye,
      title: 'Face Recognition',
      description: 'Identify known individuals in real-time',
    },
    {
      icon: Zap,
      title: 'Real-time Alerts',
      description: 'Instant Telegram notifications on detection',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Live Monitoring
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Start real-time violence detection
          </p>
        </div>
      </motion.div>

      {/* Main Empty State */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className={`rounded-3xl border-2 border-dashed p-12 text-center ${
          theme === 'light'
            ? 'border-purple-300 bg-purple-50/50'
            : 'border-zinc-700 bg-zinc-900/50'
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
          whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(168, 85, 247, 0.3)" }}
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
                theme === 'light'
                  ? 'bg-purple-100 group-hover:bg-purple-200'
                  : 'bg-purple-900/30 group-hover:bg-purple-900/50'
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
                theme === 'light'
                  ? 'bg-white border-purple-200'
                  : 'bg-zinc-950 border-zinc-800'
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

      {/* Add Camera Modal */}
      <AddCameraModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSuccess={() => {
          // Stream started - update key to force fresh video feed
          setStreamKey(Date.now());
        }}
        onStartDetection={() => {
          setShowDetectionView(true);
        }}
      />

      {/* Live Detection View */}
      <LiveDetectionView
        isOpen={showDetectionView}
        onClose={() => setShowDetectionView(false)}
        streamKey={streamKey}
      />
    </div>
  );
}