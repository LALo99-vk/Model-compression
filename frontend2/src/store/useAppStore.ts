import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { datasetService, modelService, trainingService, evaluationService, compressionService, comparisonService } from '../api/services';
import { TrainingStatusResponse, TrainingLogsResponse } from '../api/services/trainingService';

export interface AppState {
  backendConnected: boolean;
  datasets: { filename: string; size: number; path: string }[];
  selectedDatasetPath: string | null;
  selectedDatasetName: string | null;
  selectedModel: any | null;
  trainingStatus: TrainingStatusResponse | null;
  trainingLogs: TrainingLogsResponse | null;
  evaluation: {
    original?: { accuracy: number; precision: number; recall: number; f1_score: number; inference_time: number };
    compressed?: { accuracy: number; precision: number; recall: number; f1_score: number; inference_time: number };
  };
  compressionInfo: any | null;
  comparison: any | null;
  loading: boolean;
  error?: string | null;

  setBackendConnected: (connected: boolean) => void;
  setSelectedDataset: (dataset: { filename: string; path: string } | null) => void;
  fetchDatasets: () => Promise<void>;
  uploadDatasets: (files: File[]) => Promise<any>;
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

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      backendConnected: false,
      datasets: [],
      selectedDatasetPath: null,
      selectedDatasetName: null,
      selectedModel: null,
      trainingStatus: null,
      trainingLogs: null,
      evaluation: {},
      compressionInfo: null,
      comparison: null,
      loading: false,
      error: null,

      setBackendConnected: (connected) => set({ backendConnected: connected }),

      setSelectedDataset: (dataset) => {
        if (!dataset) {
          set({ selectedDatasetPath: null, selectedDatasetName: null });
        } else {
          set({ selectedDatasetPath: dataset.path, selectedDatasetName: dataset.filename });
        }
      },

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
      const response = await datasetService.upload(files);
      await get().fetchDatasets();
      return response;
    } catch (e: any) {
      set({ error: e.message });
      throw e;
    } finally {
      set({ loading: false });
    }
  },

  deleteDataset: async (filename: string) => {
    set({ loading: true, error: null });
    try {
      await datasetService.delete(filename);
      await get().fetchDatasets();
      const state = get();
      if (state.selectedDatasetName === filename) {
        set({ selectedDatasetName: null, selectedDatasetPath: null });
      }
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
      
      // Always update status first
      set({ trainingStatus: status });
      
      // Try to fetch logs
      try {
        const logs = await trainingService.logs();
        set({ trainingLogs: logs });
      } catch {}
      
      const intervalMs = Number(process.env.REACT_APP_POLLING_INTERVAL || '2000');
      
      // Continue polling if training is in progress
      // Include all statuses that indicate training is ongoing
      const ongoingStatuses = [
        'training', 
        'loading_data', 
        'preprocessing', 
        'validating',
        'normalizing',  // NEW: Universal normalization status
        'normalizing_dataset'  // Alternative status name
      ];
      
      if (ongoingStatuses.includes(status.status)) {
        setTimeout(() => get().pollTraining(), intervalMs);
      } else if (status.status === 'completed' || status.status === 'stopped') {
        // Training completed - ensure we have latest logs and status
        try {
          const finalLogs = await trainingService.logs();
          const finalStatus = await trainingService.status(); // Get status one more time to ensure it's latest
          set({ trainingLogs: finalLogs, trainingStatus: finalStatus });
        } catch (e) {
          // Even if logs fail, ensure status is set
          const finalStatus = await trainingService.status();
          set({ trainingStatus: finalStatus });
        }
        // Stop polling - training is complete
      }
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  stopTraining: async () => {
    await trainingService.stop();
    // Poll once to get the updated "stopped" status from backend
    setTimeout(() => get().pollTraining(), 500);
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
    }),
    {
      name: 'model-compression-store',
      // Only persist the selected dataset and model - not transient state
      partialize: (state) => ({
        selectedDatasetPath: state.selectedDatasetPath,
        selectedDatasetName: state.selectedDatasetName,
        selectedModel: state.selectedModel,
      }),
    }
  )
);