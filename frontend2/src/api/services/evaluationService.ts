import api from '../client';

export interface EvaluateRequest {
  model_type: 'original' | 'compressed' | string;
  dataset_path: string;
}

export interface MetricsResponse {
  message?: string;
  model_type?: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    inference_time: number;
  };
  saved_to?: string;
}

export const evaluationService = {
  async evaluate(req: EvaluateRequest): Promise<MetricsResponse> {
    const res = await api.post('/api/evaluation/evaluate', req);
    return res.data as MetricsResponse;
  },

  async metrics(modelType: 'original' | 'compressed'): Promise<MetricsResponse['metrics']> {
    const res = await api.get(`/api/evaluation/metrics/${modelType}`);
    return res.data as MetricsResponse['metrics'];
  },

  async allMetrics(): Promise<{ original?: MetricsResponse['metrics']; compressed?: MetricsResponse['metrics'] }> {
    const res = await api.get('/api/evaluation/all-metrics');
    return res.data as { original?: MetricsResponse['metrics']; compressed?: MetricsResponse['metrics'] };
  },
};