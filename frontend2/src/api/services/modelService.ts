import api from '../client';

export interface AvailableModelsResponse {
  models: Record<string, any>;
  count: number;
}

export interface ModelSelectionRequest {
  model_type: string;
  task_type: string;
  input_shape?: [number, number, number] | null;
  num_classes?: number | null;
  config?: Record<string, any> | null;
}

export const modelService = {
  async getAvailable(): Promise<AvailableModelsResponse> {
    const res = await api.get('/api/model/available');
    return res.data as AvailableModelsResponse;
  },

  async select(req: ModelSelectionRequest): Promise<{ message: string; selection: any }> {
    const res = await api.post('/api/model/select', req);
    return res.data as { message: string; selection: any };
  },

  async current(): Promise<any> {
    const res = await api.get('/api/model/current');
    return res.data as any;
  },
};