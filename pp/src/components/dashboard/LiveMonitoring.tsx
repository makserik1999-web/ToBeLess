import { Video, AlertCircle, CheckCircle, XCircle, Camera, Play, Pause, Maximize2 } from 'lucide-react';
import { useTheme } from '../Dashboard';
import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';
import { AddCameraModal } from './AddCameraModal';
import { LiveDetectionView } from './LiveDetectionView';

export function LiveMonitoring() {
  const { theme } = useTheme();
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDetectionView, setShowDetectionView] = useState(false);
  
  const cameras = [
    { 
      id: 1, 
      name: 'Main Entrance', 
      location: 'Building A', 
      status: 'active', 
      alerts: 0, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1739011121424-5e498ece6c3c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzZWN1cml0eSUyMGNhbWVyYSUyMGZvb3RhZ2UlMjBlbnRyYW5jZXxlbnwxfHx8fDE3NjU0NzYxMjV8MA&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 2, 
      name: 'Parking Lot East', 
      location: 'Building A', 
      status: 'active', 
      alerts: 1, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1653750366046-289780bd8125?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXJraW5nJTIwbG90JTIwc3VydmVpbGxhbmNlJTIwY2FtZXJhfGVufDF8fHx8MTc2NTQ3NjEyNnww&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 3, 
      name: 'Lobby', 
      location: 'Building B', 
      status: 'active', 
      alerts: 0, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1505841468529-d99f8d82ef8f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxvZmZpY2UlMjBsb2JieSUyMGludGVyaW9yfGVufDF8fHx8MTc2NTQ3NjEyNnww&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 4, 
      name: 'Cafeteria', 
      location: 'Building C', 
      status: 'warning', 
      alerts: 2, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1601351841251-766245326eee?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjYWZldGVyaWElMjBkaW5pbmclMjBoYWxsfGVufDF8fHx8MTc2NTQ3NjEyNnww&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 5, 
      name: 'Emergency Exit', 
      location: 'Building A', 
      status: 'active', 
      alerts: 0, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1551897922-6a919947ae24?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxlbWVyZ2VuY3klMjBleGl0JTIwY29ycmlkb3J8ZW58MXx8fHwxNzY1NDc2MTI3fDA&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 6, 
      name: 'Conference Room', 
      location: 'Building D', 
      status: 'offline', 
      alerts: 0, 
      isLive: false,
      image: 'https://images.unsplash.com/photo-1744095407215-66e40734e23a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjb25mZXJlbmNlJTIwcm9vbSUyMG1lZXRpbmd8ZW58MXx8fHwxNzY1MzU4NTM4fDA&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 7, 
      name: 'Stairwell North', 
      location: 'Building B', 
      status: 'active', 
      alerts: 0, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1519114284650-2924897ab2cf?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzdGFpcndlbGwlMjBidWlsZGluZyUyMGludGVyaW9yfGVufDF8fHx8MTc2NTQ3NjEyOHww&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      id: 8, 
      name: 'Loading Dock', 
      location: 'Building C', 
      status: 'active', 
      alerts: 1, 
      isLive: true,
      image: 'https://images.unsplash.com/photo-1673901035157-850dde83663b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx3YXJlaG91c2UlMjBsb2FkaW5nJTIwZG9ja3xlbnwxfHx8fDE3NjU0MDU3OTB8MA&ixlib=rb-4.1.0&q=80&w=1080'
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <h1 className={`text-3xl mb-1 font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
            Live Monitoring
          </h1>
          <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
            Real-time camera feeds and detection status
          </p>
        </motion.div>
        <motion.button
          onClick={() => setShowAddModal(true)}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition-colors font-medium"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          whileHover={{ scale: 1.05, boxShadow: "0 10px 30px rgba(168, 85, 247, 0.3)" }}
          whileTap={{ scale: 0.95 }}
        >
          <Camera className="w-4 h-4 inline mr-2" />
          Add Camera
        </motion.button>
      </motion.div>

      {/* Camera Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {cameras.map((camera, index) => (
          <motion.div
            key={camera.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.05, duration: 0.3 }}
            whileHover={{ scale: 1.02, y: -4 }}
            onClick={() => setSelectedCamera(camera.id)}
            className={`backdrop-blur-xl border rounded-2xl overflow-hidden transition-all cursor-pointer group ${
              selectedCamera === camera.id
                ? theme === 'light'
                  ? 'border-purple-400 shadow-lg shadow-purple-100'
                  : 'border-purple-600 shadow-lg shadow-purple-500/20'
                : theme === 'light'
                ? 'bg-white border-purple-200 hover:border-purple-300'
                : 'bg-zinc-950 border-zinc-900 hover:border-zinc-800'
            }`}
          >
            {/* Video Preview */}
            <div className={`aspect-video relative overflow-hidden ${
              theme === 'light' ? 'bg-purple-100' : 'bg-zinc-900'
            }`}>
              {/* Actual Camera Image */}
              <img 
                src={camera.image} 
                alt={camera.name}
                className={`w-full h-full object-cover ${camera.status === 'offline' ? 'grayscale opacity-50' : ''}`}
              />
              
              {/* Animated scan line */}
              {camera.isLive && (
                <motion.div
                  className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-purple-500 to-transparent"
                  animate={{ y: ['0%', '100%'] }}
                  transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                />
              )}

              {/* CCTV Overlay Effect */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Corner brackets */}
                <div className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-white/40"></div>
                <div className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-white/40"></div>
                <div className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-white/40"></div>
                <div className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-white/40"></div>
                
                {/* REC indicator */}
                {camera.isLive && (
                  <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/50 backdrop-blur-sm px-2 py-1 rounded">
                    <motion.div
                      className="w-2 h-2 bg-red-500 rounded-full"
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    />
                    <span className="text-xs text-white font-bold tracking-wider">REC</span>
                  </div>
                )}
              </div>

              {/* Gradient overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
              
              {/* Status Badge */}
              <div className="absolute top-3 right-3 flex items-center gap-2">
                <motion.div
                  className={`w-2 h-2 rounded-full ${
                    camera.status === 'active' ? 'bg-emerald-500' :
                    camera.status === 'warning' ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`}
                  animate={camera.status !== 'offline' ? { 
                    scale: [1, 1.2, 1],
                    opacity: [1, 0.7, 1]
                  } : {}}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                <span className="text-xs text-white uppercase tracking-wide font-semibold bg-black/50 backdrop-blur-sm px-2 py-1 rounded">
                  {camera.status === 'active' ? 'Live' : camera.status === 'warning' ? 'Alert' : 'Offline'}
                </span>
              </div>

              {/* Alert Badge */}
              {camera.alerts > 0 && (
                <motion.div 
                  className="absolute bottom-3 right-3 px-2 py-1 bg-red-500 text-white text-xs rounded-lg font-semibold shadow-lg"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 15 }}
                >
                  {camera.alerts} Alert{camera.alerts > 1 ? 's' : ''}
                </motion.div>
              )}

              {/* Play overlay on hover */}
              <motion.div
                className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/40 transition-opacity"
              >
                <motion.div
                  className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
                  whileHover={{ scale: 1.1 }}
                >
                  <Play className="w-8 h-8 text-white ml-1" />
                </motion.div>
              </motion.div>
            </div>

            {/* Camera Info */}
            <div className="p-4">
              <h3 className={`font-semibold mb-1 ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                {camera.name}
              </h3>
              <p className={`text-sm font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                {camera.location}
              </p>
              
              <div className={`mt-3 pt-3 border-t flex items-center justify-between ${
                theme === 'light' ? 'border-purple-200' : 'border-zinc-800'
              }`}>
                <div className="flex items-center gap-1 text-xs font-medium">
                  {camera.status === 'active' && <CheckCircle className="w-3 h-3 text-emerald-500" />}
                  {camera.status === 'warning' && <AlertCircle className="w-3 h-3 text-yellow-500" />}
                  {camera.status === 'offline' && <XCircle className="w-3 h-3 text-red-500" />}
                  <span className={
                    camera.status === 'active' ? 'text-emerald-500' :
                    camera.status === 'warning' ? 'text-yellow-500' :
                    'text-red-500'
                  }>
                    {camera.status === 'active' ? 'Operational' : camera.status === 'warning' ? 'Detecting' : 'Offline'}
                  </span>
                </div>
                <button className="text-xs text-purple-500 hover:text-purple-600 transition-colors font-medium">
                  View Feed
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Selected Camera Detail Modal */}
      <AnimatePresence>
        {selectedCamera && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            onClick={() => setSelectedCamera(null)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className={`max-w-6xl w-full rounded-3xl overflow-hidden ${
                theme === 'light' ? 'bg-white' : 'bg-zinc-950'
              }`}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="aspect-video bg-black relative">
                <img 
                  src={cameras.find(c => c.id === selectedCamera)?.image} 
                  alt={cameras.find(c => c.id === selectedCamera)?.name}
                  className="w-full h-full object-cover"
                />
                {cameras.find(c => c.id === selectedCamera)?.isLive && (
                  <>
                    <motion.div
                      className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent"
                      animate={{ y: ['0%', '100%'] }}
                      transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                    />
                    {/* REC indicator */}
                    <div className="absolute top-6 left-6 flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-2 rounded-lg">
                      <motion.div
                        className="w-3 h-3 bg-red-500 rounded-full"
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                      <span className="text-sm text-white font-bold tracking-wider">REC</span>
                    </div>
                  </>
                )}
              </div>
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className={`text-2xl font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                      {cameras.find(c => c.id === selectedCamera)?.name}
                    </h2>
                    <p className={`font-medium ${theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'}`}>
                      {cameras.find(c => c.id === selectedCamera)?.location}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button className={`p-3 rounded-xl transition-colors ${
                      theme === 'light' ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' : 'bg-zinc-900 text-purple-400 hover:bg-zinc-800'
                    }`}>
                      <Pause className="w-5 h-5" />
                    </button>
                    <button className={`p-3 rounded-xl transition-colors ${
                      theme === 'light' ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' : 'bg-zinc-900 text-purple-400 hover:bg-zinc-800'
                    }`}>
                      <Maximize2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add Camera Modal */}
      <AddCameraModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSuccess={() => {
          console.log('Stream started successfully');
        }}
        onStartDetection={() => {
          setShowDetectionView(true);
        }}
      />

      {/* Live Detection View */}
      <LiveDetectionView
        isOpen={showDetectionView}
        onClose={() => setShowDetectionView(false)}
      />
    </div>
  );
}