import { useEffect, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useToast } from '../components/ui/ToastContainer';

export const useDataset = () => {
  const { datasets, fetchDatasets, uploadDatasets, deleteDataset, selectedDatasetPath, selectedDatasetName, setSelectedDataset } = useAppStore((s) => ({
    datasets: s.datasets,
    fetchDatasets: s.fetchDatasets,
    uploadDatasets: s.uploadDatasets,
    deleteDataset: s.deleteDataset,
    selectedDatasetPath: s.selectedDatasetPath,
    selectedDatasetName: s.selectedDatasetName,
    setSelectedDataset: s.setSelectedDataset,
  }));
  const { showSuccess, showError } = useToast();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      setProgress(detail?.percent || 0);
    };
    window.addEventListener('upload-progress', handler as EventListener);
    return () => window.removeEventListener('upload-progress', handler as EventListener);
  }, []);

  const upload = async (files: File[]) => {
    try {
      const maxSize = Number(process.env.REACT_APP_MAX_FILE_SIZE || '104857600');
      const allowed = ['.csv', '.jpg', '.jpeg', '.png', '.bmp'];
      const invalid: string[] = [];
      const tooLarge: string[] = [];
      files.forEach((f) => {
        const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
        if (!allowed.includes(ext)) invalid.push(f.name);
        if (f.size > maxSize) tooLarge.push(f.name);
      });
      if (invalid.length) {
        showError('Unsupported file type', `Allowed: ${allowed.join(', ')}`);
        return;
      }
      if (tooLarge.length) {
        showError('File too large', `Max size: ${Math.round(maxSize / (1024 * 1024))} MB`);
        return;
      }
      await uploadDatasets(files);
      if (files.length > 0) {
        const f = files[0];
        setSelectedDataset({ filename: f.name, path: `uploads/${f.name}` });
      }
      showSuccess('Upload Complete', 'Files uploaded successfully');
      setProgress(100);
    } catch (e: any) {
      showError('Upload Failed', e.message);
    }
  };

  const remove = async (filename: string) => {
    try {
      await deleteDataset(filename);
      showSuccess('Deleted', filename);
    } catch (e: any) {
      showError('Delete Failed', e.message);
    }
  };

  return { datasets, progress, upload, remove, selectedDatasetPath, selectedDatasetName, setSelectedDataset };
};