const API_BASE_URL = 'http://localhost:8000';

// API client configuration
const apiClient = {
  get: async (endpoint: string) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },
  
  post: async (endpoint: string, data?: any) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },
  
  delete: async (endpoint: string) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  },
  
  upload: async (endpoint: string, formData: FormData) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
};

// Dataset API
export const datasetAPI = {
  upload: (formData: FormData) => apiClient.upload('/api/dataset/upload', formData),
  list: () => apiClient.get('/api/dataset/list'),
  delete: (filename: string) => apiClient.delete(`/api/dataset/delete/${filename}`),
};

// Model API
export const modelAPI = {
  getAvailable: () => apiClient.get('/api/model/available'),
  select: (config: any) => apiClient.post('/api/model/select', config),
  getCurrent: () => apiClient.get('/api/model/current'),
};

// Training API
export const trainingAPI = {
  start: (config: any) => apiClient.post('/api/training/start', config),
  getStatus: () => apiClient.get('/api/training/status'),
  getLogs: () => apiClient.get('/api/training/logs'),
};

// Evaluation API
export const evaluationAPI = {
  evaluate: (modelType: string) => apiClient.post('/api/evaluation/evaluate', { model_type: modelType }),
  getMetrics: (modelType: string) => apiClient.get(`/api/evaluation/metrics/${modelType}`),
  getAllMetrics: () => apiClient.get('/api/evaluation/all-metrics'),
};

// Compression API
export const compressionAPI = {
  compress: (method: string, params: any) => apiClient.post('/api/compression/compress', { method, params }),
  getMethods: () => apiClient.get('/api/compression/methods'),
  getInfo: () => apiClient.get('/api/compression/info'),
};

// Comparison API
export const comparisonAPI = {
  compare: () => apiClient.get('/api/comparison/compare'),
  getSummary: () => apiClient.get('/api/comparison/summary'),
  getTable: () => apiClient.get('/api/comparison/table'),
};