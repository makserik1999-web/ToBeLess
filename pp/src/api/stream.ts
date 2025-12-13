// Stream control API for ToBeLess AI

import { apiClient } from './client';

export interface StartStreamRequest {
  source?: string; // '0' for webcam, RTSP URL, or omit for file
  file?: File;     // Video file to upload
}

export interface StreamResponse {
  success: boolean;
  streaming?: boolean;
  stream_url?: string;
  job_id?: string;
  error?: string;
}

export const streamApi = {
  async start(request: StartStreamRequest): Promise<StreamResponse> {
    if (request.file) {
      const formData = new FormData();
      formData.append('file', request.file);
      const response = await apiClient.post<StreamResponse>('/start_stream', formData);
      return response.data || { success: false, error: 'No data returned' };
    } else {
      const response = await apiClient.post<StreamResponse>('/start_stream', {
        source: request.source || '0'
      });
      return response.data || { success: false, error: 'No data returned' };
    }
  },

  async stop(): Promise<{ success: boolean }> {
    const response = await apiClient.post<{ success: boolean }>('/stop_stream');
    return response.data || { success: false };
  },

  getVideoFeedUrl(): string {
    return apiClient.getStreamUrl('/video_feed');
  },

  getStatsStreamUrl(): string {
    return apiClient.getStreamUrl('/stats_stream');
  },
};
