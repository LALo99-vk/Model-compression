import api from '../client';

export interface CompressRequest {
  method: 'pruning' | 'quantization' | 'distillation' | string;
  pruning_amount?: number;
  quantization_bits?: number;
  distillation_temperature?: number;
  distillation_alpha?: number;
}

export const compressionService = {
  async compress(req: CompressRequest): Promise<any> {
    const res = await api.post('/api/compression/compress', req);
    return res.data as any;
  },

  async methods(): Promise<{ methods: Record<string, any> }> {
    const res = await api.get('/api/compression/methods');
    return res.data as { methods: Record<string, any> };
  },

  async info(): Promise<any> {
    const res = await api.get('/api/compression/info');
    return res.data as any;
  },
};