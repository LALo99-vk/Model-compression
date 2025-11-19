import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    optimizeDeps: {
      exclude: ['lucide-react'],
    },
    define: {
      'process.env.REACT_APP_API_URL': JSON.stringify(env.REACT_APP_API_URL ?? 'http://localhost:8000'),
      'process.env.REACT_APP_POLLING_INTERVAL': JSON.stringify(env.REACT_APP_POLLING_INTERVAL ?? '2000'),
      'process.env.REACT_APP_MAX_FILE_SIZE': JSON.stringify(env.REACT_APP_MAX_FILE_SIZE ?? '104857600'),
    },
  };
});
