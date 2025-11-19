import { create } from 'zustand';
import { datasetService, modelService, trainingService, evaluationService, compressionService, comparisonService } from '../api/services';

export interface AppState {
  backendConnected: boolean;
  datasets: { filename: string; size: number; path: string }[];
  selectedModel: any | null;
  trainingStatus: {
    status: string;
    current_epoch: number;
    total_epochs: number;
    message?: string;
    timestamp?: number;
  } | null;
  trainingLogs: any | null;
  evaluation: {
    original?: { accuracy: number; precision: number; recall: number; f1_score: number; inference_time: number };
    compressed?: { accuracy: number; precision: number; recall: number; f1_score: number; inference_time: number };
  };
  compressionInfo: any | null;
  comparison: any | null;
  loading: boolean;
  error?: string | null;

  setBackendConnected: (connected: boolean) => void;
  fetchDatasets: () => Promise<void>;
  uploadDatasets: (files: File[]) => Promise<void>;
  deleteDataset: (filename: string) => Promise<void>;
  fetchAvailableModels: () => Promise<any>;
  selectModel: (req: { model_type: string; task_type: string; input_shape?: any; num_classes?: number; config?: Record<string, any> }) => Promise<void>;
  startTraining: (req: { dataset_path: string; epochs?: number; batch_size?: number; validation_split?: number }) => Promise<void>;
  pollTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  evaluateModel: (req: { model_type: string; dataset_path: string }) => Promise<void>;
  fetchMetrics: () => Promise<void>;
  compressModel: (req: { method: string; pruning_amount?: number; quantization_bits?: number; distillation_temperature?: number; distillation_alpha?: number }) => Promise<void>;
  fetchComparison: () => Promise<void>;
}

export const useAppStore = create<AppState>((set, get) => ({
  backendConnected: false,
  datasets: [],
  selectedModel: null,
  trainingStatus: null,
  trainingLogs: null,
  evaluation: {},
  compressionInfo: null,
  comparison: null,
  loading: false,
  error: null,

  setBackendConnected: (connected) => set({ backendConnected: connected }),

  fetchDatasets: async () => {
    set({ loading: true, error: null });
    try {
      const res = await datasetService.list();
      set({ datasets: res.files });
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  uploadDatasets: async (files: File[]) => {
    set({ loading: true, error: null });
    try {
      await datasetService.upload(files);
      await get().fetchDatasets();
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  deleteDataset: async (filename: string) => {
    set({ loading: true, error: null });
    try {
      await datasetService.delete(filename);
      await get().fetchDatasets();
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  fetchAvailableModels: async () => {
    const res = await modelService.getAvailable();
    return res.models;
  },

  selectModel: async (req) => {
    set({ loading: true, error: null });
    try {
      const res = await modelService.select(req);
      set({ selectedModel: res.selection });
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  startTraining: async (req) => {
    set({ loading: true, error: null });
    try {
      await trainingService.start(req);
      await get().pollTraining();
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  pollTraining: async () => {
    try {
      const status = await trainingService.status();
      set({ trainingStatus: status });
      try {
        const logs = await trainingService.logs();
        set({ trainingLogs: logs });
      } catch {}
      const intervalMs = Number(process.env.REACT_APP_POLLING_INTERVAL || '2000');
      if (status.status === 'training') {
        setTimeout(() => get().pollTraining(), intervalMs);
      }
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  stopTraining: async () => {
    await trainingService.stop();
    set({ trainingStatus: null });
  },

  evaluateModel: async (req) => {
    set({ loading: true, error: null });
    try {
      const res = await evaluationService.evaluate(req);
      set({ evaluation: { ...get().evaluation, [req.model_type]: res.metrics } });
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  fetchMetrics: async () => {
    try {
      const all = await evaluationService.allMetrics();
      set({ evaluation: all });
    } catch {}
  },

  compressModel: async (req) => {
    set({ loading: true, error: null });
    try {
      const res = await compressionService.compress(req);
      set({ compressionInfo: res.compression_info || res });
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },

  fetchComparison: async () => {
    set({ loading: true, error: null });
    try {
      const res = await comparisonService.compare();
      set({ comparison: res });
    } catch (e: any) {
      set({ error: e.message });
    } finally {
      set({ loading: false });
    }
  },
}));