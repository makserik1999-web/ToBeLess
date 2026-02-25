"""
Scream/Audio Detection Module
Detects screams, shouts, and distress sounds using audio analysis
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple
import threading
import queue

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[ScreamDetector] pyaudio not available. Install with: pip install pyaudio")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[ScreamDetector] librosa not available. Install with: pip install librosa")


class ScreamDetector:
    """
    Scream and distress sound detector using audio analysis

    Detection methods:
    1. Volume spike detection (sudden loud sounds)
    2. Frequency analysis (screams have high-frequency components)
    3. Zero-crossing rate (screams have high ZCR)
    4. Spectral characteristics
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        chunk_size: int = 2048,
        volume_threshold: float = 0.3,
        scream_freq_min: float = 1000,
        scream_freq_max: float = 4000,
        detection_window: float = 0.5,
        cooldown_seconds: float = 2.0,
        device_index: Optional[int] = None,
        debug: bool = False
    ):
        """
        Initialize scream detector

        Args:
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size for processing
            volume_threshold: Volume threshold for potential scream (0-1)
            scream_freq_min: Minimum frequency for scream detection (Hz)
            scream_freq_max: Maximum frequency for scream detection (Hz)
            detection_window: Time window for analysis (seconds)
            cooldown_seconds: Cooldown between detections
            device_index: Audio input device index (None for default)
            debug: Enable debug output
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.volume_threshold = volume_threshold
        self.scream_freq_min = scream_freq_min
        self.scream_freq_max = scream_freq_max
        self.detection_window = detection_window
        self.cooldown_seconds = cooldown_seconds
        self.device_index = device_index
        self.debug = debug

        # Audio capture
        self.audio = None
        self.stream = None
        self.is_running = False
        self.audio_thread = None

        # Audio buffer for analysis
        buffer_size = int(sample_rate * detection_window)
        self.audio_buffer = deque(maxlen=buffer_size)

        # Detection state
        self.last_detection_time = 0
        self.current_volume = 0.0
        self.current_scream_score = 0.0
        self.scream_detected = False

        # History and stats
        self.detection_history = []
        self.stats = {
            'total_screams_detected': 0,
            'average_volume': 0.0,
            'max_volume': 0.0
        }

        # Detection queue for thread communication
        self.detection_queue = queue.Queue()

        # Video file mode state (pre-loaded audio)
        self._video_mode = False
        self._video_rms_times = None
        self._video_rms_values = None
        self._video_threshold = 0.08

        if self.debug:
            print(f"[ScreamDetector] Initialized with sample_rate={sample_rate}, chunk_size={chunk_size}")
            print(f"[ScreamDetector] pyaudio available: {PYAUDIO_AVAILABLE}")
            print(f"[ScreamDetector] librosa available: {LIBROSA_AVAILABLE}")

    def start(self) -> bool:
        """
        Start audio capture and analysis

        Returns:
            True if started successfully
        """
        if not PYAUDIO_AVAILABLE:
            if self.debug:
                print("[ScreamDetector] Cannot start: pyaudio not available")
            return False

        if self.is_running:
            return True

        try:
            self.audio = pyaudio.PyAudio()

            # Find input device
            if self.device_index is None:
                self.device_index = self.audio.get_default_input_device_info()['index']

            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            self.is_running = True
            self.stream.start_stream()

            if self.debug:
                print(f"[ScreamDetector] ✓ Audio capture started on device {self.device_index}")

            return True

        except Exception as e:
            if self.debug:
                print(f"[ScreamDetector] Failed to start: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop audio capture"""
        self.is_running = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None

        if self.debug:
            print("[ScreamDetector] Audio capture stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if not self.is_running:
            return (None, pyaudio.paComplete)

        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            # Add to buffer
            self.audio_buffer.extend(audio_data)

            # Analyze if buffer is full enough
            if len(self.audio_buffer) >= self.chunk_size * 2:
                self._analyze_audio()

        except Exception as e:
            if self.debug:
                print(f"[ScreamDetector] Callback error: {e}")

        return (None, pyaudio.paContinue)

    def _analyze_audio(self):
        """Analyze audio buffer for screams"""
        try:
            # Convert buffer to numpy array
            audio_array = np.array(list(self.audio_buffer), dtype=np.float32)

            # Calculate RMS volume
            rms = np.sqrt(np.mean(audio_array**2))
            self.current_volume = min(1.0, rms * 3)  # Scale for display

            # Update stats
            self.stats['average_volume'] = (self.stats['average_volume'] * 0.99 + self.current_volume * 0.01)
            self.stats['max_volume'] = max(self.stats['max_volume'], self.current_volume)

            # Check volume threshold
            if rms < self.volume_threshold * 0.3:
                self.current_scream_score = 0.0
                self.scream_detected = False
                return

            # Calculate scream score
            scream_score = self._calculate_scream_score(audio_array)
            self.current_scream_score = scream_score

            # Check for scream detection
            current_time = datetime.now().timestamp()
            if (scream_score > 0.6 and
                current_time - self.last_detection_time > self.cooldown_seconds):

                self.scream_detected = True
                self.last_detection_time = current_time
                self.stats['total_screams_detected'] += 1

                detection = {
                    'timestamp': datetime.now().isoformat(),
                    'score': scream_score,
                    'volume': self.current_volume,
                    'type': 'scream'
                }

                self.detection_history.append(detection)
                self.detection_queue.put(detection)

                if self.debug:
                    print(f"[ScreamDetector] ⚠ SCREAM DETECTED! Score: {scream_score:.2f}, Volume: {self.current_volume:.2f}")

                # Keep history limited
                if len(self.detection_history) > 1000:
                    self.detection_history = self.detection_history[-500:]
            else:
                self.scream_detected = False

        except Exception as e:
            if self.debug:
                print(f"[ScreamDetector] Analysis error: {e}")

    def _calculate_scream_score(self, audio_array: np.ndarray) -> float:
        """
        Calculate scream likelihood score (0-1)

        Uses multiple features:
        1. Volume (loudness)
        2. Zero-crossing rate (high for screams)
        3. Spectral centroid (screams have high frequencies)
        4. Energy in scream frequency band
        """
        try:
            score = 0.0
            weights = []

            # 1. Volume score (screams are loud)
            rms = np.sqrt(np.mean(audio_array**2))
            volume_score = min(1.0, rms / self.volume_threshold)
            score += volume_score * 0.25
            weights.append(0.25)

            # 2. Zero-crossing rate (screams have high ZCR)
            zcr = np.sum(np.abs(np.diff(np.sign(audio_array)))) / (2 * len(audio_array))
            zcr_score = min(1.0, zcr / 0.1)  # Normalize
            score += zcr_score * 0.2
            weights.append(0.2)

            # 3. Spectral analysis (if librosa available)
            if LIBROSA_AVAILABLE:
                # Spectral centroid
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio_array,
                    sr=self.sample_rate
                )[0]
                avg_centroid = np.mean(spectral_centroid)

                # Screams typically have centroid in 1000-4000 Hz range
                if self.scream_freq_min <= avg_centroid <= self.scream_freq_max:
                    centroid_score = 1.0
                elif avg_centroid > self.scream_freq_max:
                    centroid_score = 0.7
                else:
                    centroid_score = avg_centroid / self.scream_freq_min

                score += centroid_score * 0.25
                weights.append(0.25)

                # 4. Energy in scream frequency band
                fft = np.abs(np.fft.rfft(audio_array))
                freqs = np.fft.rfftfreq(len(audio_array), 1/self.sample_rate)

                # Find indices for scream frequency range
                scream_mask = (freqs >= self.scream_freq_min) & (freqs <= self.scream_freq_max)
                total_energy = np.sum(fft**2)
                scream_energy = np.sum(fft[scream_mask]**2)

                if total_energy > 0:
                    energy_ratio = scream_energy / total_energy
                    energy_score = min(1.0, energy_ratio * 3)  # Screams should have >30% in this band
                    score += energy_score * 0.3
                    weights.append(0.3)
            else:
                # Simple FFT analysis without librosa
                fft = np.abs(np.fft.rfft(audio_array))
                freqs = np.fft.rfftfreq(len(audio_array), 1/self.sample_rate)

                # High frequency energy
                high_freq_mask = freqs > 1000
                total_energy = np.sum(fft**2)
                high_energy = np.sum(fft[high_freq_mask]**2)

                if total_energy > 0:
                    high_ratio = high_energy / total_energy
                    score += min(1.0, high_ratio * 2) * 0.55
                    weights.append(0.55)

            # Normalize score
            total_weight = sum(weights)
            if total_weight > 0:
                score = score / total_weight * sum(weights)

            return min(1.0, max(0.0, score))

        except Exception as e:
            if self.debug:
                print(f"[ScreamDetector] Score calculation error: {e}")
            return 0.0

    def get_current_state(self) -> Dict:
        """Get current detection state"""
        return {
            'is_running': self.is_running,
            'volume': self.current_volume,
            'scream_score': self.current_scream_score,
            'scream_detected': self.scream_detected,
            'timestamp': datetime.now().isoformat()
        }

    def get_detection(self) -> Optional[Dict]:
        """
        Get latest detection from queue (non-blocking)

        Returns:
            Detection dict or None if no new detection
        """
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()

    # ------------------------------------------------------------------
    # Video file mode: pre-load audio and check volume per frame
    # ------------------------------------------------------------------

    def load_video_audio(self, video_path: str) -> bool:
        """
        Pre-load audio from a video file for offline scream detection.
        Simple rule: RMS volume > auto-calibrated threshold  →  scream.
        Works without librosa: uses ffmpeg → raw WAV → numpy RMS.
        """
        import subprocess, tempfile, os, wave

        print(f"[ScreamDetector] Loading audio from {video_path} ...")
        y, sr = None, None

        # --- Attempt 1: librosa direct (if installed) ---
        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(video_path, sr=22050, mono=True)
                print("[ScreamDetector] librosa load OK")
            except Exception as e1:
                print(f"[ScreamDetector] librosa failed ({e1}), trying ffmpeg...")

        # --- Attempt 2: ffmpeg → 16-bit WAV → numpy ---
        if y is None:
            # Prefer bundled ffmpeg from imageio-ffmpeg (no PATH needed)
            ffmpeg_exe = 'ffmpeg'
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass

            tmp_wav = tempfile.mktemp(suffix='.wav')
            try:
                result = subprocess.run(
                    [ffmpeg_exe, '-y', '-i', video_path,
                     '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                     tmp_wav],
                    capture_output=True, timeout=120
                )
                if result.returncode == 0:
                    with wave.open(tmp_wav, 'rb') as wf:
                        sr = wf.getframerate()
                        raw = wf.readframes(wf.getnframes())
                    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    print(f"[ScreamDetector] ffmpeg+wave OK ({len(y)/sr:.1f}s)")
                else:
                    print(f"[ScreamDetector] ffmpeg error: {result.stderr.decode()[-300:]}")
            except Exception as e2:
                print(f"[ScreamDetector] ffmpeg extraction failed: {e2}")
            finally:
                if os.path.exists(tmp_wav):
                    try:
                        os.unlink(tmp_wav)
                    except Exception:
                        pass

        if y is None or sr is None:
            print("[ScreamDetector] ✗ Cannot load audio – scream detection disabled for this file")
            self._video_mode = False
            return False

        try:
            # Compute RMS in 0.25-s windows using pure numpy
            hop = int(sr * 0.25)
            win = int(sr * 0.5)
            n = max(1, (len(y) - win) // hop + 1)
            rms = np.array(
                [float(np.sqrt(np.mean(y[i * hop: i * hop + win] ** 2))) for i in range(n)],
                dtype=np.float32,
            )
            times = np.arange(n, dtype=np.float32) * 0.25  # seconds per window

            # Auto-calibrate: mean + 2×std  (only unusually loud windows trigger)
            mean_rms  = float(rms.mean())
            std_rms   = float(rms.std())
            threshold = max(0.03, mean_rms + 2.0 * std_rms)

            self._video_rms_times  = times
            self._video_rms_values = rms
            self._video_threshold  = threshold
            self._video_mode       = True

            print(f"[ScreamDetector] ✓ Audio ready: {n} windows, "
                  f"duration={times[-1]:.1f}s, mean={mean_rms:.4f}, "
                  f"max={rms.max():.4f}, threshold={threshold:.4f}")
            return True
        except Exception as e:
            print(f"[ScreamDetector] RMS computation failed: {e}")
            self._video_mode = False
            return False

    def process_video_frame(self, pos_sec: float) -> None:
        """
        Call once per video frame with the current playback position.
        Puts a detection event into self.detection_queue if loud enough.

        Args:
            pos_sec: Current video position in seconds
        """
        if not self._video_mode or self._video_rms_times is None:
            return
        idx = int(np.searchsorted(self._video_rms_times, pos_sec))
        if idx >= len(self._video_rms_values):
            return
        rms_val = float(self._video_rms_values[idx])
        current_time = datetime.now().timestamp()
        if rms_val > self._video_threshold and \
                current_time - self.last_detection_time > self.cooldown_seconds:
            self.scream_detected = True
            self.last_detection_time = current_time
            self.stats['total_screams_detected'] += 1
            score = min(1.0, rms_val / max(self._video_threshold, 0.001))
            detection = {
                'timestamp': datetime.now().isoformat(),
                'score': score,
                'volume': rms_val,
                'type': 'scream',
            }
            self.detection_history.append(detection)
            self.detection_queue.put(detection)
            if self.debug:
                print(f"[ScreamDetector] Scream at {pos_sec:.2f}s  rms={rms_val:.4f}")

    # ------------------------------------------------------------------

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_screams_detected': 0,
            'average_volume': 0.0,
            'max_volume': 0.0
        }
        self.detection_history = []
        self._video_mode = False
        self._video_rms_times = None
        self._video_rms_values = None

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
