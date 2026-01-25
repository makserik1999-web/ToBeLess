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
  Camera,
  Sword,
  PersonStanding,
  Volume2,
  Settings
} from 'lucide-react';
import { useTheme } from '../Dashboard';
import { streamApi } from '../../api/stream';
import { apiClient } from '../../api/client';
import { detectionApi } from '../../api/detection';

interface LiveDetectionViewProps {
  isOpen: boolean;
  onClose: () => void;
  streamKey?: number; // Changes when a new stream starts to force video refresh
}

interface Stats {
  people: number;
  fights: number;
  weapons: number;
  falls: number;
  screams: number;
  confidence: number;
  fps: number;
  timestamp?: number;
}

interface DetectionToggles {
  weapon: boolean;
  fall: boolean;
  scream: boolean;
}

export function LiveDetectionView({ isOpen, onClose, streamKey }: LiveDetectionViewProps) {
  const { theme } = useTheme();
  const [stats, setStats] = useState<Stats>({
    people: 0,
    fights: 0,
    weapons: 0,
    falls: 0,
    screams: 0,
    confidence: 0,
    fps: 0,
  });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [toggles, setToggles] = useState<DetectionToggles>({
    weapon: true,
    fall: true,
    scream: false
  });
  const eventSourceRef = useRef<EventSource | null>(null);
  const [feedKey, setFeedKey] = useState(Date.now());
  const videoFeedUrl = `${apiClient.getStreamUrl('/video_feed')}?t=${feedKey}`;

  // Load detection status on mount and reset feed
  useEffect(() => {
    if (!isOpen) return;

    // Reset feed key to get fresh video - use streamKey if provided, otherwise Date.now()
    setFeedKey(streamKey || Date.now());

    // Reset stats
    setStats({
      people: 0,
      fights: 0,
      weapons: 0,
      falls: 0,
      screams: 0,
      confidence: 0,
      fps: 0,
    });

    const loadDetectionStatus = async () => {
      try {
        const status = await detectionApi.getStatus();
        if (status.success) {
          setToggles({
            weapon: status.weapon_detection,
            fall: status.fall_detection,
            scream: status.scream_detection
          });
        }
      } catch (err) {
        console.error('Error loading detection status:', err);
      }
    };
    loadDetectionStatus();
  }, [isOpen, streamKey]);

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
          weapons: data.weapons || 0,
          falls: data.falls || 0,
          screams: data.screams || 0,
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

  const handleToggleWeapon = async () => {
    try {
      const result = await detectionApi.toggleWeaponDetection();
      if (result.success && result.weapon_detection_enabled !== undefined) {
        setToggles(prev => ({ ...prev, weapon: result.weapon_detection_enabled! }));
      }
    } catch (err) {
      console.error('Error toggling weapon detection:', err);
    }
  };

  const handleToggleFall = async () => {
    try {
      const result = await detectionApi.toggleFallDetection();
      if (result.success && result.fall_detection_enabled !== undefined) {
        setToggles(prev => ({ ...prev, fall: result.fall_detection_enabled! }));
      }
    } catch (err) {
      console.error('Error toggling fall detection:', err);
    }
  };

  const handleToggleScream = async () => {
    try {
      const result = await detectionApi.toggleScreamDetection();
      if (result.success && result.scream_detection_enabled !== undefined) {
        setToggles(prev => ({ ...prev, scream: result.scream_detection_enabled! }));
      }
    } catch (err) {
      console.error('Error toggling scream detection:', err);
    }
  };

  if (!isOpen) return null;

  const hasFight = stats.fights > 0;
  const hasWeapon = stats.weapons > 0;
  const hasFall = stats.falls > 0;
  const hasScream = stats.screams > 0;
  const hasAlert = hasFight || hasWeapon || hasFall || hasScream;

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
                hasAlert
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'bg-green-500 text-white'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  hasAlert ? 'bg-white' : 'bg-white animate-pulse'
                }`}></div>
                {hasFight ? 'ALERT: Fight' : hasWeapon ? 'ALERT: Weapon' : hasFall ? 'ALERT: Fall' : hasScream ? 'ALERT: Scream' : 'Monitoring'}
              </div>

              {/* Settings Button */}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-3 rounded-xl transition-colors ${
                  showSettings
                    ? 'bg-purple-600 text-white'
                    : theme === 'light'
                    ? 'hover:bg-purple-100 text-zinc-600'
                    : 'hover:bg-zinc-800 text-zinc-400'
                }`}
              >
                <Settings className="w-5 h-5" />
              </button>

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
            isFullscreen ? 'p-0' : 'p-4'
          } flex gap-4`}>
            {/* Video Feed */}
            <div className={`flex-1 min-h-0 ${isFullscreen ? 'w-full' : ''}`}>
              <div className={`h-full rounded-2xl overflow-hidden border-2 ${
                hasFight
                  ? 'border-red-500 shadow-lg shadow-red-500/50'
                  : theme === 'light'
                  ? 'border-purple-200 bg-black'
                  : 'border-zinc-800 bg-black'
              } relative`}>
                {/* Video Stream */}
                <img
                  key={feedKey}
                  src={videoFeedUrl}
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
              <div className="w-72 overflow-y-auto space-y-3 pr-1" style={{ maxHeight: 'calc(100vh - 120px)' }}>
                {/* Stats Cards */}
                <div className="space-y-2">
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
                        <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center">
                          <Users className="w-5 h-5 text-blue-500" />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.people}
                          </div>
                          <div className={`text-xs ${
                            theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            People
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
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                          hasFight ? 'bg-white/20' : 'bg-red-500/10'
                        }`}>
                          <AlertTriangle className={`w-5 h-5 ${
                            hasFight ? 'text-white' : 'text-red-500'
                          }`} />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            hasFight ? 'text-white' : theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.fights}
                          </div>
                          <div className={`text-xs ${
                            hasFight ? 'text-white/90' : theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Fights
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Weapons Count */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      hasWeapon
                        ? 'bg-red-600 border-red-700 shadow-lg shadow-red-600/50'
                        : theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                    animate={{
                      scale: hasWeapon ? [1, 1.05, 1] : 1,
                    }}
                    transition={{ duration: 0.5, repeat: hasWeapon ? Infinity : 0 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                          hasWeapon ? 'bg-white/20' : 'bg-orange-500/10'
                        }`}>
                          <Sword className={`w-5 h-5 ${
                            hasWeapon ? 'text-white' : 'text-orange-500'
                          }`} />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            hasWeapon ? 'text-white' : theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.weapons}
                          </div>
                          <div className={`text-xs ${
                            hasWeapon ? 'text-white/90' : theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Weapons
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Falls Count */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      hasFall
                        ? 'bg-amber-500 border-amber-600 shadow-lg shadow-amber-500/50'
                        : theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                    animate={{
                      scale: hasFall ? [1, 1.05, 1] : 1,
                    }}
                    transition={{ duration: 0.5, repeat: hasFall ? Infinity : 0 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                          hasFall ? 'bg-white/20' : 'bg-amber-500/10'
                        }`}>
                          <PersonStanding className={`w-5 h-5 ${
                            hasFall ? 'text-white' : 'text-amber-500'
                          }`} />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            hasFall ? 'text-white' : theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.falls}
                          </div>
                          <div className={`text-xs ${
                            hasFall ? 'text-white/90' : theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Falls
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Screams Count */}
                  <motion.div
                    className={`p-4 rounded-2xl border-2 ${
                      hasScream
                        ? 'bg-purple-500 border-purple-600 shadow-lg shadow-purple-500/50'
                        : theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                    animate={{
                      scale: hasScream ? [1, 1.05, 1] : 1,
                    }}
                    transition={{ duration: 0.5, repeat: hasScream ? Infinity : 0 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                          hasScream ? 'bg-white/20' : 'bg-purple-500/10'
                        }`}>
                          <Volume2 className={`w-5 h-5 ${
                            hasScream ? 'text-white' : 'text-purple-500'
                          }`} />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            hasScream ? 'text-white' : theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.screams}
                          </div>
                          <div className={`text-xs ${
                            hasScream ? 'text-white/90' : theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Screams
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
                        <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center">
                          <Activity className="w-5 h-5 text-purple-500" />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {Math.round(stats.confidence)}%
                          </div>
                          <div className={`text-xs ${
                            theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            Confidence
                          </div>
                        </div>
                      </div>
                    </div>
                    {/* Progress bar */}
                    <div className="mt-2">
                      <div className={`h-1.5 rounded-full overflow-hidden ${
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
                        <div className="w-10 h-10 rounded-xl bg-green-500/10 flex items-center justify-center">
                          <Zap className="w-5 h-5 text-green-500" />
                        </div>
                        <div>
                          <div className={`text-xl font-bold ${
                            theme === 'light' ? 'text-zinc-900' : 'text-white'
                          }`}>
                            {stats.fps.toFixed(1)}
                          </div>
                          <div className={`text-xs ${
                            theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                          }`}>
                            FPS
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>

                {/* Alert Banner */}
                {hasAlert && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-2xl bg-red-500 text-white"
                  >
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="w-6 h-6" />
                      <div>
                        <div className="font-bold">
                          {hasFight ? 'Violence Detected!' : hasWeapon ? 'Weapon Detected!' : hasFall ? 'Fall Detected!' : 'Scream Detected!'}
                        </div>
                        <div className="text-sm text-white/90">
                          Authorities have been notified
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Settings Panel */}
                {showSettings && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className={`p-4 rounded-2xl border-2 ${
                      theme === 'light'
                        ? 'bg-white border-purple-200'
                        : 'bg-zinc-950 border-zinc-800'
                    }`}
                  >
                    <h3 className={`text-sm font-semibold mb-3 ${
                      theme === 'light' ? 'text-zinc-900' : 'text-white'
                    }`}>Detection Modules</h3>

                    <div className="space-y-3">
                      {/* Weapon Detection Toggle */}
                      <div className="flex items-center justify-between">
                        <span className={`text-sm ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Weapon Detection
                        </span>
                        <button
                          onClick={handleToggleWeapon}
                          className={`w-12 h-6 rounded-full transition-colors relative ${
                            toggles.weapon ? 'bg-purple-600' : 'bg-zinc-600'
                          }`}
                        >
                          <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all ${
                            toggles.weapon ? 'left-6' : 'left-0.5'
                          }`}></div>
                        </button>
                      </div>

                      {/* Fall Detection Toggle */}
                      <div className="flex items-center justify-between">
                        <span className={`text-sm ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Fall Detection
                        </span>
                        <button
                          onClick={handleToggleFall}
                          className={`w-12 h-6 rounded-full transition-colors relative ${
                            toggles.fall ? 'bg-purple-600' : 'bg-zinc-600'
                          }`}
                        >
                          <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all ${
                            toggles.fall ? 'left-6' : 'left-0.5'
                          }`}></div>
                        </button>
                      </div>

                      {/* Scream Detection Toggle */}
                      <div className="flex items-center justify-between">
                        <span className={`text-sm ${theme === 'light' ? 'text-zinc-700' : 'text-zinc-300'}`}>
                          Scream Detection
                        </span>
                        <button
                          onClick={handleToggleScream}
                          className={`w-12 h-6 rounded-full transition-colors relative ${
                            toggles.scream ? 'bg-purple-600' : 'bg-zinc-600'
                          }`}
                        >
                          <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all ${
                            toggles.scream ? 'left-6' : 'left-0.5'
                          }`}></div>
                        </button>
                      </div>
                    </div>

                  </motion.div>
                )}

                {/* Stop Button - Always visible at bottom of stats */}
                <button
                  onClick={handleStop}
                  disabled={isStopping}
                  className="w-full py-4 bg-red-600 hover:bg-red-700 text-white rounded-2xl font-semibold transition-colors disabled:opacity-50 flex items-center justify-center gap-2 sticky bottom-0"
                >
                  <StopCircle className="w-5 h-5" />
                  {isStopping ? 'Stopping...' : 'Stop Detection'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Floating Stop Button - Always visible */}
        <motion.button
          onClick={handleStop}
          disabled={isStopping}
          className="fixed bottom-8 left-1/2 -translate-x-1/2 px-8 py-4 bg-red-600 hover:bg-red-700 text-white rounded-2xl font-bold text-lg transition-all disabled:opacity-50 flex items-center gap-3 shadow-2xl shadow-red-500/50 z-50"
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <StopCircle className="w-6 h-6" />
          {isStopping ? 'Stopping...' : 'Stop Detection'}
        </motion.button>
      </div>
    </motion.div>
  );
}
