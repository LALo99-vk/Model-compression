export interface Dataset {
  id: string;
  filename: string;
  size: number;
  uploadDate: string;
  status: 'ready' | 'processing' | 'error';
  type: 'csv' | 'image';
}

export interface Model {
  id: string;
  name: string;
  type: 'cnn' | 'rnn' | 'decision_tree';
  description: string;
  bestFor: string;
  defaultConfig: Record<string, any>;
  icon: string;
}

export interface TrainingConfig {
  modelType: string;
  datasetPath: string;
  epochs: number;
  batchSize: number;
  validationSplit: number;
  taskType: 'classification' | 'regression';
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'completed' | 'error';
  currentEpoch: number;
  totalEpochs: number;
  progress: number;
  estimatedTimeRemaining: string;
  metrics: {
    trainingLoss: number;
    validationLoss: number;
    validationAccuracy: number;
  };
  logs: string[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  inferenceTime: number;
  modelSize: number;
}

export interface CompressionMethod {
  id: string;
  name: string;
  description: string;
  typicalReduction: string;
  accuracyImpact: string;
  icon: string;
  parameters: Record<string, any>;
}

export interface ComparisonData {
  original: ModelMetrics & { type: 'Original' };
  compressed: ModelMetrics & { type: 'Compressed' };
  sizeReduction: number;
  accuracyPreserved: number;
  speedImprovement: number;
}