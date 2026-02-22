import cv2
import threading
import time
import logging

class RealTimeCapture:
    """
    Threaded video capture for low-latency RTSP and webcam streams.
    Always provides the most recent frame, discarding any internal buffer.
    """
    def __init__(self, source, reconnect_delay=5, reconnect=True):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.reconnect = reconnect
        self.cap = cv2.VideoCapture(source)
        
        # Set buffer size to 1 to minimize latency if supported by backend
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame = None
        self.ret = False
        self.running = True
        self.stopped = False
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        logging.info(f"RealTimeCapture: Started for source {source}")

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                if not self.reconnect:
                    self.running = False
                    break
                logging.warning(f"RealTimeCapture: Source {self.source} disconnected. Retrying in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self.cap.open(self.source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = self.cap.read()
            
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                else:
                    # If read fails, the stream might be down
                    self.cap.release()
            
            # Tiny sleep to prevent 100% CPU usage if capture is too fast
            # but small enough to not cause lag
            time.sleep(0.001)

    def read(self):
        """Returns the latest frame captured."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened() or self.running

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
        self.stopped = True
        logging.info("RealTimeCapture: Stopped")

    def get(self, propId):
        """Compatibility with cv2.VideoCapture.get()"""
        return self.cap.get(propId)

    def set(self, propId, value):
        """Compatibility with cv2.VideoCapture.set()"""
        return self.cap.set(propId, value)
