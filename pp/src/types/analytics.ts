// TypeScript type definitions for ToBeLess AI Analytics

export interface FightEvent {
  start_time: string;
  end_time?: string;
  duration: number;
  confidence: number;
  frame_idx?: number;
  location?: string;
}

export interface StatsSnapshot {
  people: number;
  fights: number;
  weapons: number;
  falls: number;
  screams: number;
  confidence: number;
  fps: number;
  timestamp: string;
  escalation_warning?: boolean;
  conflict_type?: string;
}

export interface DetectionStatus {
  success: boolean;
  fight_detection: boolean;
  weapon_detection: boolean;
  fall_detection: boolean;
  scream_detection: boolean;
  statistics: {
    fights: number;
    weapons: number;
    falls: number;
    screams: number;
  };
}

export interface ReportResponse {
  success: boolean;
  filepath?: string;
  filename?: string;
  format?: string;
  error?: string;
}

export interface ReportInfo {
  filename: string;
  path: string;
  size: number;
  created: string;
  type: string;
}

export interface Metrics {
  confidence: number;
  people_count: number;
  people_names?: string[];
  body_distances?: number[];
  limb_crossings?: number;
  close_contacts?: number;
}

export interface RecentDataPoint {
  frame: number;
  fight: boolean;
  people: number;
  metrics: Metrics;
  timestamp: string;
}

export interface AnalyticsData {
  fight_events: FightEvent[];
  people_count_history?: Array<{ frame: number; count: number; timestamp: string }>;
  total_frames?: number;
  fight_duration_total?: number;
  total_detections?: number;
  fight_duration_history?: number[];
  detection_confidence_history?: Array<{
    frame: number;
    confidence: number;
    timestamp: string;
  }>;
}

export interface AnalyticsResponse {
  success: boolean;
  streaming: boolean;
  recent_data: RecentDataPoint[];
  analytics: AnalyticsData;
  latest_stats: StatsSnapshot;
}

export interface Hotspot {
  x: number;
  y: number;
  intensity: number;
  events: number;
}

export interface HotspotsResponse {
  success: boolean;
  hotspots: Hotspot[];
}
