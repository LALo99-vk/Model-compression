import { useEffect } from 'react';
import api from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { useToast } from '../components/ui/ToastContainer';

export const useBackendStatus = () => {
  const setConnected = useAppStore((s) => s.setBackendConnected);
  const { showSuccess, showError } = useToast();

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await api.get('/health');
        if (mounted) {
          setConnected(true);
          showSuccess('Backend Connected', 'FastAPI is reachable');
        }
      } catch {
        if (mounted) {
          setConnected(false);
          showError('Backend Disconnected', 'FastAPI is not reachable');
        }
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [setConnected, showSuccess, showError]);
};