// Server-Sent Events hook for real-time stats streaming

import { useEffect, useState, useRef } from 'react';
import { StatsSnapshot } from '../types/analytics';

export function useSSE(url: string, enabled: boolean = true) {
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setConnected(false);
      return;
    }

    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log('[SSE] Connected to stats stream');
      setConnected(true);
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as StatsSnapshot;
        setStats(data);
      } catch (error) {
        console.error('[SSE] Parse error:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('[SSE] Connection error:', error);
      setConnected(false);
      eventSource.close();

      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (enabled) {
          console.log('[SSE] Attempting to reconnect...');
        }
      }, 3000);
    };

    return () => {
      eventSource.close();
    };
  }, [url, enabled]);

  return { stats, connected };
}
