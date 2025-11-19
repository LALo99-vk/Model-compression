import api from '../client';

export const comparisonService = {
  async compare(): Promise<any> {
    const res = await api.get('/api/comparison/compare');
    return res.data as any;
  },

  async summary(): Promise<any> {
    const res = await api.get('/api/comparison/summary');
    return res.data as any;
  },

  async table(): Promise<{ headers: string[]; rows: any[] }> {
    const res = await api.get('/api/comparison/table');
    return res.data as { headers: string[]; rows: any[] };
  },
};