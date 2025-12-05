import { useEffect } from 'react';
import api from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { useToast } from '../components/ui/ToastContainer';

export const useBackendStatus = () => {
  const setConnected = useAppStore((s) => s.setBackendConnected);
  const { showError } = useToast();

  useEffect(() => {
    let mounted = true;
    let wasConnected = false;
    
    const check = async () => {
      try {
        await api.get('/health');
        if (mounted) {
          setConnected(true);
          wasConnected = true;
          // Don't show success toast - connection is visible in navbar
        }
      } catch {
        if (mounted) {
          setConnected(false);
          // Only show error if we were previously connected
          if (wasConnected) {
            showError('Backend Disconnected');
            wasConnected = false;
          }
        }
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [setConnected, showError]);
};