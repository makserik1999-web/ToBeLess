// Analytics API for ToBeLess AI

import { apiClient } from './client';
import { AnalyticsResponse, HotspotsResponse } from '../types/analytics';

export const analyticsApi = {
  async getAnalytics(): Promise<AnalyticsResponse> {
    const response = await apiClient.get<AnalyticsResponse>('/analytics');
    return response.data || {
      success: false,
      streaming: false,
      recent_data: [],
      analytics: { fight_events: [] },
      latest_stats: {
        people: 0,
        fights: 0,
        confidence: 0,
        fps: 0,
        timestamp: new Date().toISOString()
      }
    };
  },

  getHeatmapUrl(): string {
    return apiClient.getImageUrl('/heatmap', true);
  },

  async getHotspots(): Promise<HotspotsResponse> {
    const response = await apiClient.get<HotspotsResponse>('/hotspots');
    return response.data || { success: false, hotspots: [] };
  },
};
