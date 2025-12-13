// Global stream state context for ToBeLess AI

import { createContext, useContext, ReactNode } from 'react';
import { useStreamControl } from '../hooks/useStreamControl';
import { useSSE } from '../hooks/useSSE';
import { streamApi, StartStreamRequest } from '../api/stream';
import { StatsSnapshot } from '../types/analytics';

interface StreamContextType {
  isStreaming: boolean;
  stats: StatsSnapshot | null;
  connected: boolean;
  startStream: (request: StartStreamRequest) => Promise<void>;
  stopStream: () => Promise<void>;
  loading: boolean;
  error: string | null;
}

const StreamContext = createContext<StreamContextType | undefined>(undefined);

export function StreamProvider({ children }: { children: ReactNode }) {
  const streamControl = useStreamControl();
  const sse = useSSE(streamApi.getStatsStreamUrl(), streamControl.isStreaming);

  return (
    <StreamContext.Provider value={{
      ...streamControl,
      stats: sse.stats,
      connected: sse.connected,
    }}>
      {children}
    </StreamContext.Provider>
  );
}

export function useStream() {
  const context = useContext(StreamContext);
  if (!context) {
    throw new Error('useStream must be used within StreamProvider');
  }
  return context;
}
