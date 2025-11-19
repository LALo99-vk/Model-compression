import api from '../client';

export interface StartTrainingRequest {
  dataset_path: string;
  epochs?: number;
  batch_size?: number;
  validation_split?: number;
}

export interface TrainingStatusResponse {
  status: string;
  current_epoch: number;
  total_epochs: number;
  message?: string;
  timestamp: number;
}

export interface TrainingLogsResponse {
  model_type: string;
  epochs: number;
  history?: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    val_accuracy: number;
  }>;
  train_score?: number;
  val_score?: number;
  training_time?: number;
}

export const trainingService = {
  async start(req: StartTrainingRequest): Promise<any> {
    const res = await api.post('/api/training/start', req);
    return res.data as any;
  },

  async status(): Promise<TrainingStatusResponse> {
    const res = await api.get('/api/training/status');
    return res.data as TrainingStatusResponse;
  },

  async logs(): Promise<TrainingLogsResponse> {
    const res = await api.get('/api/training/logs');
    return res.data as TrainingLogsResponse;
  },

  async stop(): Promise<{ message: string }> {
    const res = await api.delete('/api/training/stop');
    return res.data as { message: string };
  },
};