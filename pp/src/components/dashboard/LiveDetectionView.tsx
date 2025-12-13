import { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import {
  X,
  Users,
  AlertTriangle,
  Activity,
  Zap,
  StopCircle,
  Maximize2,
  Minimize2,
  Camera
} from 'lucide-react';
import { useTheme } from '../Dashboard';
import { streamApi } from '../../api/stream';
import { apiClient } from '../../api/client';

interface LiveDetectionViewProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Stats {
  people: number;
  fights: number;
  confidence: number;
  fps: number;
  timestamp?: number;
}

export function LiveDetectionView({ isOpen, onClose }: LiveDetectionViewProps) {
  const { theme } = useTheme();
  const [stats, setStats] = useState<Stats>({
    people: 0,
    fights: 0,
    confidence: 0,
    fps: 0,
  });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const videoFeedUrl = apiClient.getStreamUrl('/video_feed');

  // Connect to SSE stats stream
  useEffect(() => {
    if (!isOpen) return;

    const statsUrl = apiClient.getStreamUrl('/stats_stream');
    const eventSource = new EventSource(statsUrl);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setStats({
          people: data.people || 0,
          fights: data.fights || 0,
          confidence: data.confidence || 0,
          fps: data.fps || 0,
          timestamp: data.timestamp || Date.now(),
        });
      } catch (err) {
        console.error('Error parsing stats:', err);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      eventSource.close();
    };

    eventSourceRef.current = eventSource;

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [isOpen]);

  const handleStop = async () => {
    setIsStopping(true);
    try {
      await streamApi.stop();
      setTimeout(() => {
        onClose();
      }, 500);
    } catch (err) {
      console.error('Error stopping stream:', err);
    } finally {
      setIsStopping(false);
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  if (!isOpen) return null;

  const hasFight = stats.fights > 0;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className={`fixed inset-0 z-50 ${
        theme === 'light'
          ? 'bg-gradient-to-br from-purple-50 via-white to-purple-100'
          : 'bg-black'
      }`}
    >
      {/* Background pattern */}
      {theme === 'light' && (
        <div className="fixed inset-0 opacity-20">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(168, 85, 247, 0.15) 1px, transparent 0)`,
            backgroundSize: '40px 40px'
          }}></div>
        </div>
      )}

      <div className="relative h-full flex flex-col">
        {/* Header */}
        <div className={`px-6 py-4 border-b backdrop-blur-xl ${
          theme === 'light'
            ? 'bg-white/80 border-purple-200'
            : 'bg-zinc-950/80 border-zinc-800'
        }`}>
          <div className="flex items-center justify-between max-w-[2000px] mx-auto">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                hasFight
                  ? 'bg-red-500 animate-pulse'
                  : 'bg-purple-600'
              }`}>
                <Camera className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className={`text-xl font-semibold ${
                  theme === 'light' ? 'text-zinc-900' : 'text-white'
                }`}>
                  Live Detection
                </h1>
                <p className={`text-sm ${
                  theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                }`}>
                  Real-time violence detection active
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Status Indicator */}
              <div className={`px-4 py-2 rounded-xl font-semibold flex items-center gap-2 ${
                hasFight
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'bg-green-500 text-white'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  hasFight ? 'bg-white' : 'bg-white animate-pulse'
                }`}></div>
                {hasFight ? 'ALERT: Fight Detected' : 'Monitoring'}
              </div>

              <button
                onClick={toggleFullscreen}
                className={`p-3 rounded-xl transition-colors ${
                  theme === 'light'
                    ? 'hover:bg-purple-100 text-zinc-600'
                    : 'hover:bg-zinc-800 text-zinc-400'
                }`}
              >
                {isFullscreen ? (
                  <Minimize2 className="w-5 h-5" />
                ) : (
                  <Maximize2 className="w-5 h-5" />
                )}
              </button>

              <button
                onClick={handleStop}
                disabled={isStopping}
                className="px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl font-semibold transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                <StopCircle className="w-5 h-5" />
                {isStopping ? 'Stopping...' : 'Stop Detection'}
              </button>

              <button
                onClick={onClose}
                className={`p-3 rounded-xl transition-colors ${
                  theme === 'light'
                    ? 'hover:bg-purple-100 text-zinc-600'
                    : 'hover:bg-zinc-800 text-zinc-400'
                }`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          <div className={`h-full max-w-[2000px] mx-auto ${
            isFullscreen ? 'p-0' : 'p-6'
          } flex gap-6`}>
            {/* Video Feed */}
            <div className={`flex-1 ${isFullscreen ? 'w-full' : ''}`}>
              <div className={`h-full rounded-2xl overflow-hidden border-2 ${
                hasFight
                  ? 'border-red-500 shadow-lg shadow-red-500/50'
                  : theme === 'light'
                  ? 'border-purple-200 bg-black'
                  : 'border-zinc-800 bg-black'
              } relative`}>
                {/* Video Stream */}
                <img
                  src={`${videoFeedUrl}?t=${Date.now()}`}
                  alt="Live Detection Feed"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    console.error('Video feed error');
                  }}
                />

                {/* REC Indicator */}
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-2 rounded-lg">
                  <motion.div
                    className="w-3 h-3 bg-red-500 rounded-full"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                  <span className="text-sm text-white font-bold tracking-wider">REC</span>
                </div>

                {/* CCTV Corner Brackets */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute top-4 left-4 w-6 h-6 border-l-2 border-t-2 border-white/40"></div>
                  <div className="absolute top-4 right-4 w-6 h-6 border-r-2 border-t-2 border-white/40"></div>
                  <div className="absolute bottom-4 left-4 w-6 h-6 border-l-2 border-b-2 border-white/40"></div>
                  <div className="absolute bottom-4 right-4 w-6 h-6 border-r-2 border-b-2 border-white/40"></div>
                </div>

                {/* Live Stats Overlay */}
                <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm px-4 py-2 rounded-lg">
                  <div className="text-white text-sm font-mono">
                    FPS: {stats.fps.toFixed(1)}
                  </div>
                </div>
              </div>
            </div>

            {/* Stats Panel */}
            {!isFullscreen && (
              <div className="w-80 space-y-4">
                {/* Stats Cards */}
                <div className="space-y-3">
                  {/* People Count */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                    animate={{ scale: stats.people > 0 ? [1, 1.02, 1] : 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center">
                          <Users className="w-6 h-6 text-blue-500" />
                        </div>
                        <div>
                          <div className={`text-2xl font-bold ${
                            theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.people}
                          </div>
                          <div className={`text-sm ${
                            theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            People Detected
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Fights Count */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      hasFight
                        ? 'bg-red-500 border-red-600 shadow-lg shadow-red-500/50'
                        : theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                    animate={{
                      scale: hasFight ? [1, 1.05, 1] : 1,
                    }}
                    transition={{ duration: 0.5, repeat: hasFight ? Infinity : 0 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                          hasFight ? 'bg-white/20' : 'bg-red-500/10'
                        }`}>
                          <AlertTriangle className={`w-6 h-6 ${
                            hasFight ? 'text-white' : 'text-red-500'
                          }`} />
                        </div>
                        <div>
                          <div className={`text-2xl font-bold ${
                            hasFight ? 'text-white' : theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.fights}
                          </div>
                          <div className={`text-sm ${
                            hasFight ? 'text-white/90' : theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Fights Detected
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Confidence */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center">
                          <Activity className="w-6 h-6 text-purple-500" />
                        </div>
                        <div>
                          <div className={`text-2xl font-bold ${
                            theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {Math.round(stats.confidence)}%
                          </div>
                          <div className={`text-sm ${
                            theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Confidence
                          </div>
                        </div>
                      </div>
                    </div>
                    {/* Progress bar */}
                    <div className="mt-3">
                      <div className={`h-2 rounded-full overflow-hidden ${
                        theme === 'light' ? 'bg-purple-100' : 'bg-zinc-800'
                      }`}>
                        <motion.div
                          className="h-full bg-purple-600"
                          initial={{ width: 0 }}
                          animate={{ width: `${stats.confidence}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>
                  </motion.div>

                  {/* FPS */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center">
                          <Zap className="w-6 h-6 text-green-500" />
                        </div>
                        <div>
                          <div className={`text-2xl font-bold ${
                            theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.fps.toFixed(1)}
                          </div>
                          <div className={`text-sm ${
                            theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Frames/Second
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>

                {/* Alert Banner */}
                {hasFight && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-2xl bg-red-500 text-white"
                  >
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="w-6 h-6" />
                      <div>
                        <div className="font-bold">Violence Detected!</div>
                        <div className="text-sm text-white/90">
                          Authorities have been notified
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
