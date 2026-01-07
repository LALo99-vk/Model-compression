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

export interface TrainedModelInfo {
  accuracy: number;
  size_kb: number;
  parameters: number;
  path: string;
  compression_ratio?: number;
  size_reduction?: number;
  method?: string;
}

export interface TrainingSession {
  id: string;
  model_type: 'decision_tree' | 'cnn' | 'rnn' | string;
  dataset_name: string;
  dataset_path: string;
  created_at: string | null;
  training_time: number;
  original: TrainedModelInfo | null;
  compressed: TrainedModelInfo | null;
}

export interface TrainedModelsResponse {
  sessions: TrainingSession[];
  count: number;
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

  async getTrainedModels(): Promise<TrainedModelsResponse> {
    const res = await api.get('/api/model/trained');
    return res.data as TrainedModelsResponse;
  },
};