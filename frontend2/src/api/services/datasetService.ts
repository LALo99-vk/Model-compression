import api from '../client';

export interface DatasetFile {
  filename: string;
  size: number;
  path: string;
}

export interface DatasetListResponse {
  files: DatasetFile[];
  count: number;
}

export interface UploadResponse {
  message: string;
  files: DatasetFile[];
  count: number;
}

export const datasetService = {
  async upload(files: File[]): Promise<UploadResponse> {
    const form = new FormData();
    files.forEach((file) => form.append('files', file, file.name));

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
};