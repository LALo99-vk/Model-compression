import api from '../client';

export interface ValidationRequest {
  dataset_path: string;
  model_type: 'decision_tree' | 'cnn' | 'rnn';
}

export interface ValidationResponse {
  status: 'valid' | 'invalid' | 'error';
  message: string;
  issues?: string[];
  warnings?: string[];
  can_autofix?: boolean;
  fix_suggestions?: string[];
  info?: Record<string, any>;
  report_text?: string;
}

export interface ConditioningRequest {
  dataset_path: string;
  model_type: 'decision_tree' | 'cnn' | 'rnn';
  auto_fix?: boolean;
}

export interface ConditioningResponse {
  status: 'fixed' | 'invalid_after_fix' | 'error' | 'valid';
  message: string;
  new_path?: string;
  backup_path?: string;
  changes_made?: string[];
  issues?: string[];
  warnings?: string[];
}

export const validationService = {
  async validate(req: ValidationRequest): Promise<ValidationResponse> {
    const res = await api.post('/api/validation/validate', req);
    return res.data as ValidationResponse;
  },

  async condition(req: ConditioningRequest): Promise<ConditioningResponse> {
    const res = await api.post('/api/validation/condition', req);
    return res.data as ConditioningResponse;
  },

  async getReport(): Promise<ValidationResponse> {
    const res = await api.get('/api/validation/report');
    return res.data.validation_result as ValidationResponse;
  },
};

