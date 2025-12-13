// Stream control hook for managing video stream state

import { useState } from 'react';
import { streamApi, StartStreamRequest } from '../api/stream';

export function useStreamControl() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startStream = async (request: StartStreamRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await streamApi.start(request);
      if (response.success) {
        setIsStreaming(true);
      } else {
        setError(response.error || 'Failed to start stream');
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const stopStream = async () => {
    setLoading(true);
    try {
      await streamApi.stop();
      setIsStreaming(false);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return { isStreaming, loading, error, startStream, stopStream };
}
