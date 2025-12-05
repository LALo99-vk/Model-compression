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
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          // Don't rewrite the path, just proxy it
          secure: false,
          // Important for large file downloads
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.log('proxy error', err);
            });
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              console.log('Sending Request:', req.method, req.url);
            });
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('Received Response:', proxyRes.statusCode, req.url);
            });
          },
        },
      },
    },
    define: {
      'process.env.REACT_APP_API_URL': JSON.stringify(env.REACT_APP_API_URL ?? 'http://localhost:8000'),
      'process.env.REACT_APP_POLLING_INTERVAL': JSON.stringify(env.REACT_APP_POLLING_INTERVAL ?? '2000'),
      'process.env.REACT_APP_MAX_FILE_SIZE': JSON.stringify(env.REACT_APP_MAX_FILE_SIZE ?? '104857600'),
    },
  };
});
