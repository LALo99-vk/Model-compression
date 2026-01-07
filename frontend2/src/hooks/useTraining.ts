import { useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';

export const useTraining = () => {
  const { trainingStatus, trainingLogs, startTraining, pollTraining, stopTraining } = useAppStore((s) => ({
    trainingStatus: s.trainingStatus,
    trainingLogs: s.trainingLogs,
    startTraining: s.startTraining,
    pollTraining: s.pollTraining,
    stopTraining: s.stopTraining,
  }));

  const start = useCallback(
    async (req: { dataset_path: string; epochs?: number; batch_size?: number; validation_split?: number }) => {
      await startTraining(req);
      // No toast - UI shows training status
    },
    [startTraining]
  );

  const poll = useCallback(async () => {
    await pollTraining();
    // No toast - silent polling
  }, [pollTraining]);

  const stop = useCallback(async () => {
    await stopTraining();
    // No toast - UI shows stopped status
  }, [stopTraining]);

  return { trainingStatus, trainingLogs, start, poll, stop };
};