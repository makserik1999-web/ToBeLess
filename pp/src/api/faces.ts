// Face management API for ToBeLess AI

import { apiClient } from './client';

export interface AddFaceRequest {
  file: File;
  name: string;
}

export interface AddFaceResponse {
  success: boolean;
  msg?: string;
  error?: string;
}

export interface ReloadFacesResponse {
  success: boolean;
  database?: string[];
  count?: number;
  error?: string;
}

export interface FeatureStatusResponse {
  success: boolean;
  face_blur_enabled?: boolean;
  face_recognition_enabled?: boolean;
  error?: string;
}

export const facesApi = {
  async addFace(request: AddFaceRequest): Promise<AddFaceResponse> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('name', request.name);

    const response = await apiClient.post<AddFaceResponse>('/add_face', formData);
    return response.data || { success: false, error: 'No data returned' };
  },

  async reloadFaces(): Promise<ReloadFacesResponse> {
    const response = await apiClient.post<ReloadFacesResponse>('/reload_faces');
    return response.data || { success: false, error: 'No data returned' };
  },

  async toggleFaceBlur(enabled: boolean): Promise<{ success: boolean; face_blur_enabled?: boolean }> {
    const response = await apiClient.post('/toggle_face_blur', { enabled: enabled.toString() });
    return response.data || { success: false };
  },

  async toggleFaceRecognition(enabled: boolean): Promise<{ success: boolean; face_recognition_enabled?: boolean }> {
    const response = await apiClient.post('/toggle_face_recognition', { enabled: enabled.toString() });
    return response.data || { success: false };
  },

  async getFeatureStatus(): Promise<FeatureStatusResponse> {
    const response = await apiClient.get<FeatureStatusResponse>('/feature_status');
    return response.data || { success: false, error: 'No data returned' };
  },
};
