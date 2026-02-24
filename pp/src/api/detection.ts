// Detection control API for ToBeLess AI

import { apiClient } from './client';
import { DetectionStatus, ReportResponse, ReportInfo } from '../types/analytics';

export interface ToggleResponse {
  success: boolean;
  weapon_detection_enabled?: boolean;
  fall_detection_enabled?: boolean;
  scream_detection_enabled?: boolean;
  face_recognition_enabled?: boolean;
  face_blur_enabled?: boolean;
}

export type DetectionMode = 'fight' | 'weapon' | 'scream';
export type PrivacyMode = 'off' | 'recognition' | 'blur';

export interface ModeResponse {
  success: boolean;
  mode?: string;
  face_recognition_enabled?: boolean;
  face_blur_enabled?: boolean;
  error?: string;
}

export const detectionApi = {
  // Get detection module status
  async getStatus(): Promise<DetectionStatus> {
    const response = await apiClient.get<DetectionStatus>('/detection_status');
    return response.data || {
      success: false,
      fight_detection: true,
      weapon_detection: false,
      fall_detection: false,
      scream_detection: false,
      statistics: { fights: 0, weapons: 0, falls: 0, screams: 0 }
    };
  },

  // Toggle weapon detection
  async toggleWeaponDetection(): Promise<ToggleResponse> {
    const response = await apiClient.post<ToggleResponse>('/toggle_weapon_detection');
    return response.data || { success: false };
  },

  // Toggle fall detection
  async toggleFallDetection(): Promise<ToggleResponse> {
    const response = await apiClient.post<ToggleResponse>('/toggle_fall_detection');
    return response.data || { success: false };
  },

  // Toggle scream detection
  async toggleScreamDetection(): Promise<ToggleResponse> {
    const response = await apiClient.post<ToggleResponse>('/toggle_scream_detection');
    return response.data || { success: false };
  },

  // Generate report
  async generateReport(format: 'pdf' | 'excel' | 'json'): Promise<ReportResponse> {
    const response = await apiClient.post<ReportResponse>('/generate_report', { format });
    return response.data || { success: false, error: 'No data returned' };
  },

  // List available reports
  async listReports(): Promise<{ success: boolean; reports: ReportInfo[] }> {
    const response = await apiClient.get<{ success: boolean; reports: ReportInfo[] }>('/list_reports');
    return response.data || { success: false, reports: [] };
  },

  // Get download URL for a report
  getReportDownloadUrl(filename: string): string {
    return apiClient.getStreamUrl(`/download_report/${filename}`);
  },

  // Get detection events
  async getEvents(): Promise<{ success: boolean; events: any[]; count: number }> {
    const response = await apiClient.get<{ success: boolean; events: any[]; count: number }>('/detection_events');
    return response.data || { success: false, events: [], count: 0 };
  },

  // Clear detection events
  async clearEvents(): Promise<{ success: boolean }> {
    const response = await apiClient.post<{ success: boolean }>('/clear_events');
    return response.data || { success: false };
  },

  // Detection mode
  async setDetectionMode(mode: DetectionMode): Promise<ModeResponse> {
    const response = await apiClient.post<ModeResponse>('/set_detection_mode', { mode });
    return response.data || { success: false };
  },

  async getDetectionMode(): Promise<ModeResponse> {
    const response = await apiClient.get<ModeResponse>('/get_detection_mode');
    return response.data || { success: false };
  },

  // Privacy mode
  async setPrivacyMode(mode: PrivacyMode): Promise<ModeResponse> {
    const response = await apiClient.post<ModeResponse>('/set_privacy_mode', { mode });
    return response.data || { success: false };
  },

  async getPrivacyMode(): Promise<ModeResponse> {
    const response = await apiClient.get<ModeResponse>('/get_privacy_mode');
    return response.data || { success: false };
  },
};
