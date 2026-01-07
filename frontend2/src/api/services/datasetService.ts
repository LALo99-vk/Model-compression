import api from '../client';

export interface DatasetFile {
  filename: string;
  size: number;
  path: string;
  type?: 'file' | 'folder';
  image_count?: number;
}

export interface DatasetListResponse {
  files: DatasetFile[];
  count: number;
}

export interface UploadResponse {
  message: string;
  files: DatasetFile[];
  folders?: string[];
  count: number;
}

export interface DatasetPreview {
  filename: string;
  path: string;
  file_type: 'csv' | 'text' | 'image_folder' | string;
  size_bytes: number;
  // CSV specific
  columns?: string[];
  num_columns?: number;
  num_rows?: number;
  dtypes?: Record<string, string>;
  missing_values?: Record<string, number>;
  total_missing?: number;
  target_column?: string;
  unique_targets?: number;
  target_values?: Record<string, number>;
  preview?: string[][];
  // Text specific
  num_lines?: number;
  num_characters?: number;
  num_words?: number;
  vocab_size?: number;
  sample_vocab?: string[];
  format?: 'plain_text' | 'tab_separated';
  detected_labels?: string[];
  // Image folder specific
  classes?: Record<string, number>;
  num_classes?: number;
  total_images?: number;
  sample_images?: Array<{
    class: string;
    filename: string;
    size: string;
    mode: string;
  }>;
  error?: string;
}

export const datasetService = {
  async upload(files: File[]): Promise<UploadResponse> {
    const form = new FormData();
    // Use webkitRelativePath for folder uploads, fallback to name for single files
    files.forEach((file) => {
      const filename = (file as any).webkitRelativePath || file.name;
      form.append('files', file, filename);
    });

    const res = await api.post('/api/dataset/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        const percent = e.total ? Math.round((e.loaded / e.total) * 100) : 0;
        window.dispatchEvent(
          new CustomEvent('upload-progress', {
            detail: { loaded: e.loaded, total: e.total, percent },
          })
        );
      },
    });
    return res.data as UploadResponse;
  },

  async list(): Promise<DatasetListResponse> {
    const res = await api.get('/api/dataset/list');
    return res.data as DatasetListResponse;
  },

  async delete(filename: string): Promise<{ message: string }> {
    const res = await api.delete(`/api/dataset/delete/${encodeURIComponent(filename)}`);
    return res.data as { message: string };
  },

  async preview(filename: string, rows: number = 10): Promise<DatasetPreview> {
    const res = await api.get(`/api/dataset/preview/${encodeURIComponent(filename)}`, {
      params: { rows }
    });
    return res.data as DatasetPreview;
  },
};