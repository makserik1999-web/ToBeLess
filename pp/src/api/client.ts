// Base API client for ToBeLess AI

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

interface ApiResponse<T = any> {
  success: boolean;
  error?: string;
  data?: T;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      console.error(`API GET error: ${endpoint}`, error);
      return { success: false, error: (error as Error).message };
    }
  }

  async post<T>(endpoint: string, body?: any): Promise<ApiResponse<T>> {
    try {
      const isFormData = body instanceof FormData;

      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: isFormData ? {} : { 'Content-Type': 'application/json' },
        body: isFormData ? body : JSON.stringify(body),
      });

      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      console.error(`API POST error: ${endpoint}`, error);
      return { success: false, error: (error as Error).message };
    }
  }

  getStreamUrl(endpoint: string): string {
    return `${this.baseUrl}${endpoint}`;
  }

  getImageUrl(endpoint: string, cacheBuster: boolean = true): string {
    const url = `${this.baseUrl}${endpoint}`;
    return cacheBuster ? `${url}?t=${Date.now()}` : url;
  }
}

export const apiClient = new ApiClient();
