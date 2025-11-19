import { useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useToast } from '../components/ui/ToastContainer';

export const useTraining = () => {
  const { trainingStatus, trainingLogs, startTraining, pollTraining, stopTraining } = useAppStore((s) => ({
    trainingStatus: s.trainingStatus,
    trainingLogs: s.trainingLogs,
    startTraining: s.startTraining,
    pollTraining: s.pollTraining,
    stopTraining: s.stopTraining,
  }));
  const { showSuccess, showError } = useToast();

  const start = useCallback(
    async (req: { dataset_path: string; epochs?: number; batch_size?: number; validation_split?: number }) => {
      try {
        await startTraining(req);
        showSuccess('Training Started', 'Polling every 2s');
      } catch (e: any) {
        showError('Training Failed', e.message);
      }
    },
    [startTraining, showSuccess, showError]
  );

  const poll = useCallback(async () => {
    try {
      await pollTraining();
    } catch (e: any) {
      showError('Polling Error', e.message);
    }
  }, [pollTraining, showError]);

  const stop = useCallback(async () => {
    try {
      await stopTraining();
      showSuccess('Training Stopped');
    } catch (e: any) {
      showError('Stop Failed', e.message);
    }
  }, [stopTraining, showSuccess, showError]);

  return { trainingStatus, trainingLogs, start, poll, stop };
};