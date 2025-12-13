// Settings API for ToBeLess AI

import { apiClient } from './client';

export interface DetectorSettings {
  body_proximity_threshold?: number;
  limb_proximity_threshold?: number;
  fight_hold_duration?: number;
  min_pose_confidence?: number;
}

export interface SettingsResponse {
  success: boolean;
  error?: string;
}

export const settingsApi = {
  async updateSettings(settings: DetectorSettings): Promise<SettingsResponse> {
    const response = await apiClient.post<SettingsResponse>('/settings', settings);
    return response.data || { success: false, error: 'No data returned' };
  },
};
