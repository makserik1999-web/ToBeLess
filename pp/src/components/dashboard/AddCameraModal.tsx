import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, Upload, Camera, Radio, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { useTheme } from '../Dashboard';
import { streamApi } from '../../api/stream';
import { facesApi } from '../../api/faces';

interface AddCameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  onStartDetection?: () => void;
}

type VideoSource = 'file' | 'webcam' | 'rtsp';

export function AddCameraModal({ isOpen, onClose, onSuccess, onStartDetection }: AddCameraModalProps) {
  const { theme } = useTheme();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedSource, setSelectedSource] = useState<VideoSource>('webcam');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rtspUrl, setRtspUrl] = useState('');
  const [webcamIndex, setWebcamIndex] = useState('0');

  // Feature toggles
  const [faceRecognitionEnabled, setFaceRecognitionEnabled] = useState(true);
  const [faceBlurEnabled, setFaceBlurEnabled] = useState(false);
  const [fightDetectionEnabled] = useState(true); // Always enabled

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleStartStream = async () => {
    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      // First, toggle features
      if (faceRecognitionEnabled) {
        await facesApi.toggleFaceRecognition(true);
      }
      if (faceBlurEnabled) {
        await facesApi.toggleFaceBlur(true);
      }

      // Then start the stream
      let response;

      if (selectedSource === 'file') {
        if (!selectedFile) {
          setError('Please select a video file');
          setIsLoading(false);
          return;
        }
        response = await streamApi.start({ file: selectedFile });
      } else if (selectedSource === 'rtsp') {
        if (!rtspUrl) {
          setError('Please enter an RTSP URL');
          setIsLoading(false);
          return;
        }
        response = await streamApi.start({ source: rtspUrl });
      } else {
        // webcam
        response = await streamApi.start({ source: webcamIndex });
      }

      if (response.success) {
        setSuccess(true);
        setTimeout(() => {
          onSuccess?.();
          onStartDetection?.();
          onClose();
        }, 1000);
      } else {
        setError(response.error || 'Failed to start stream');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 20 }}
          className={`max-w-2xl w-full max-h-[90vh] rounded-3xl overflow-hidden flex flex-col ${
            theme === 'light' ? 'bg-white' : 'bg-zinc-950 border border-zinc-800'
          }`}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className={`p-4 border-b ${theme === 'light' ? 'border-purple-200' : 'border-zinc-800'}`}>
            <div className="flex items-center justify-between">
              <h2 className={`text-xl font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                Add Camera Source
              </h2>
              <button
                onClick={onClose}
                className={`p-2 rounded-lg transition-colors ${
                  theme === 'light'
                    ? 'hover:bg-purple-100 text-zinc-600'
                    : 'hover:bg-zinc-800 text-zinc-400'
                }`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-4 space-y-3 overflow-y-auto flex-1">
            {/* Video Source Selection */}
            <div>
              <label className={`block text-sm font-semibold mb-2 ${
                theme === 'light' ? 'text-zinc-900' : 'text-white'
              }`}>
                Video Source
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setSelectedSource('webcam')}
                  className={`p-2 rounded-lg border-2 transition-all ${
                    selectedSource === 'webcam'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20'
                      : theme === 'light'
                      ? 'border-purple-200 hover:border-purple-300 bg-white'
                      : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900'
                  }`}
                >
                  <Camera className={`w-6 h-6 mx-auto mb-1 ${
                    selectedSource === 'webcam' ? 'text-purple-600' : 'text-zinc-400'
                  }`} />
                  <div className={`text-xs font-medium ${
                    selectedSource === 'webcam'
                      ? 'text-purple-600'
                      : theme === 'light' ? 'text-zinc-900' : 'text-white'
                  }`}>
                    Webcam
                  </div>
                </button>

                <button
                  onClick={() => setSelectedSource('file')}
                  className={`p-2 rounded-lg border-2 transition-all ${
                    selectedSource === 'file'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20'
                      : theme === 'light'
                      ? 'border-purple-200 hover:border-purple-300 bg-white'
                      : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900'
                  }`}
                >
                  <Upload className={`w-6 h-6 mx-auto mb-1 ${
                    selectedSource === 'file' ? 'text-purple-600' : 'text-zinc-400'
                  }`} />
                  <div className={`text-xs font-medium ${
                    selectedSource === 'file'
                      ? 'text-purple-600'
                      : theme === 'light' ? 'text-zinc-900' : 'text-white'
                  }`}>
                    Video File
                  </div>
                </button>

                <button
                  onClick={() => setSelectedSource('rtsp')}
                  className={`p-2 rounded-lg border-2 transition-all ${
                    selectedSource === 'rtsp'
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20'
                      : theme === 'light'
                      ? 'border-purple-200 hover:border-purple-300 bg-white'
                      : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900'
                  }`}
                >
                  <Radio className={`w-6 h-6 mx-auto mb-1 ${
                    selectedSource === 'rtsp' ? 'text-purple-600' : 'text-zinc-400'
                  }`} />
                  <div className={`text-xs font-medium ${
                    selectedSource === 'rtsp'
                      ? 'text-purple-600'
                      : theme === 'light' ? 'text-zinc-900' : 'text-white'
                  }`}>
                    RTSP
                  </div>
                </button>
              </div>
            </div>

            {/* Source-specific inputs */}
            {selectedSource === 'file' && (
              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`w-full p-3 rounded-lg border-2 border-dashed transition-all ${
                    selectedFile
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-950/20'
                      : theme === 'light'
                      ? 'border-purple-200 hover:border-purple-300 bg-white'
                      : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900'
                  }`}
                >
                  <div className={`text-sm font-medium ${
                    selectedFile
                      ? 'text-purple-600'
                      : theme === 'light' ? 'text-zinc-600' : 'text-zinc-400'
                  }`}>
                    {selectedFile ? selectedFile.name : 'Choose video file'}
                  </div>
                </button>
              </div>
            )}

            {selectedSource === 'webcam' && (
              <div>
                <input
                  type="text"
                  value={webcamIndex}
                  onChange={(e) => setWebcamIndex(e.target.value)}
                  placeholder="Webcam index (usually 0)"
                  className={`w-full px-3 py-2 rounded-lg border-2 transition-all text-sm ${
                    theme === 'light'
                      ? 'border-purple-200 bg-white focus:border-purple-500'
                      : 'border-zinc-800 bg-zinc-900 focus:border-purple-500 text-white'
                  } outline-none`}
                />
              </div>
            )}

            {selectedSource === 'rtsp' && (
              <div>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://username:password@ip:port/stream"
                  className={`w-full px-3 py-2 rounded-lg border-2 transition-all text-sm ${
                    theme === 'light'
                      ? 'border-purple-200 bg-white focus:border-purple-500'
                      : 'border-zinc-800 bg-zinc-900 focus:border-purple-500 text-white'
                  } outline-none`}
                />
              </div>
            )}

            {/* Detection Features */}
            <div>
              <label className={`block text-sm font-semibold mb-2 ${
                theme === 'light' ? 'text-zinc-900' : 'text-white'
              }`}>
                Detection Features
              </label>
              <div className="space-y-2">
                {/* Fight Detection - Always enabled */}
                <div className={`p-3 rounded-xl border-2 ${
                  theme === 'light'
                    ? 'border-purple-200 bg-purple-50'
                    : 'border-purple-900 bg-purple-950/20'
                }`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center">
                        <CheckCircle className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <div className={`text-sm font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                          Fight Detection
                        </div>
                        <div className="text-xs text-purple-600 dark:text-purple-400">
                          Always enabled
                        </div>
                      </div>
                    </div>
                    <div className="px-2 py-1 bg-purple-600 text-white text-xs font-semibold rounded-full">
                      ON
                    </div>
                  </div>
                </div>

                {/* Face Recognition Toggle */}
                <div className={`p-3 rounded-xl border-2 cursor-pointer transition-all ${
                  faceRecognitionEnabled
                    ? theme === 'light'
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-purple-600 bg-purple-950/20'
                    : theme === 'light'
                    ? 'border-purple-200 bg-white'
                    : 'border-zinc-800 bg-zinc-900'
                }`}
                onClick={() => {
                  const newValue = !faceRecognitionEnabled;
                  setFaceRecognitionEnabled(newValue);
                  if (newValue) setFaceBlurEnabled(false); // Turn off face blur when enabling recognition
                }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div>
                        <div className={`text-sm font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                          Face Recognition
                        </div>
                        <div className={`text-xs ${
                          theme === 'light' ? 'text-zinc-500' : 'text-zinc-400'
                        }`}>
                          Identify people
                        </div>
                      </div>
                    </div>
                    <div className={`w-11 h-6 rounded-full transition-colors ${
                      faceRecognitionEnabled ? 'bg-purple-600' : 'bg-zinc-300 dark:bg-zinc-700'
                    } relative`}>
                      <motion.div
                        className="absolute top-1 w-4 h-4 bg-white rounded-full"
                        animate={{ left: faceRecognitionEnabled ? '24px' : '4px' }}
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                      />
                    </div>
                  </div>
                </div>

                {/* Face Blur Toggle */}
                <div className={`p-3 rounded-xl border-2 cursor-pointer transition-all ${
                  faceBlurEnabled
                    ? theme === 'light'
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-purple-600 bg-purple-950/20'
                    : theme === 'light'
                    ? 'border-purple-200 bg-white'
                    : 'border-zinc-800 bg-zinc-900'
                }`}
                onClick={() => {
                  const newValue = !faceBlurEnabled;
                  setFaceBlurEnabled(newValue);
                  if (newValue) setFaceRecognitionEnabled(false); // Turn off face recognition when enabling blur
                }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div>
                        <div className={`text-sm font-semibold ${theme === 'light' ? 'text-zinc-900' : 'text-white'}`}>
                          Face Blur
                        </div>
                        <div className={`text-xs ${
                          theme === 'light' ? 'text-zinc-500' : 'text-zinc-400'
                        }`}>
                          Privacy protection
                        </div>
                      </div>
                    </div>
                    <div className={`w-11 h-6 rounded-full transition-colors ${
                      faceBlurEnabled ? 'bg-purple-600' : 'bg-zinc-300 dark:bg-zinc-700'
                    } relative`}>
                      <motion.div
                        className="absolute top-1 w-4 h-4 bg-white rounded-full"
                        animate={{ left: faceBlurEnabled ? '24px' : '4px' }}
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Error/Success Messages */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl bg-red-50 dark:bg-red-950/20 border-2 border-red-200 dark:border-red-900"
              >
                <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                  <AlertCircle className="w-5 h-5" />
                  <span className="text-sm font-medium">{error}</span>
                </div>
              </motion.div>
            )}

            {success && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl bg-green-50 dark:bg-green-950/20 border-2 border-green-200 dark:border-green-900"
              >
                <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm font-medium">Stream started successfully!</span>
                </div>
              </motion.div>
            )}
          </div>

          {/* Footer - Always visible at bottom */}
          <div className={`p-4 border-t flex-shrink-0 ${theme === 'light' ? 'border-purple-200' : 'border-zinc-800'}`}>
            <div className="flex gap-3">
              <button
                onClick={onClose}
                disabled={isLoading}
                className={`flex-1 px-6 py-3 rounded-xl font-semibold transition-colors ${
                  theme === 'light'
                    ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                    : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                Cancel
              </button>
              <button
                onClick={handleStartStream}
                disabled={isLoading || success}
                className="flex-1 px-6 py-3 rounded-xl font-semibold bg-purple-600 hover:bg-purple-700 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Starting...
                  </>
                ) : success ? (
                  <>
                    <CheckCircle className="w-5 h-5" />
                    Started
                  </>
                ) : (
                  'Start Stream'
                )}
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
