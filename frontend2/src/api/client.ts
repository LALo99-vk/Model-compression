import axios, { AxiosError } from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
});

api.interceptors.request.use(
  (config) => {
    console.info(`[API] → ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error', error);
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => {
    console.info(`[API] ← ${response.status} ${response.config.url}`);
    return response;
  },
  (error: AxiosError) => {
    const status = error.response?.status;
    const url = error.config?.url;
    const message = (error.response?.data as any)?.detail || error.message;
    console.error(`[API] Error ${status} on ${url}: ${message}`);
    window.dispatchEvent(
      new CustomEvent('global-error', { detail: { status, url, message } })
    );
    return Promise.reject(error);
  }
);

export default api;