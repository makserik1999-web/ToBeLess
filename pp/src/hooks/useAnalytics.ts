// Analytics polling hook

import { useEffect, useState } from 'react';
import { analyticsApi } from '../api/analytics';
import { AnalyticsResponse } from '../types/analytics';

export function useAnalytics(pollingInterval: number = 800, enabled: boolean = true) {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }

    let isCancelled = false;

    const fetchData = async () => {
      try {
        const response = await analyticsApi.getAnalytics();
        if (!isCancelled) {
          setData(response);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (!isCancelled) {
          setError((err as Error).message);
          setLoading(false);
        }
      }
    };

    // Initial fetch
    fetchData();

    // Set up polling
    const interval = setInterval(fetchData, pollingInterval);

    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, [pollingInterval, enabled]);

  return { data, loading, error };
}
