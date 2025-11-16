import React, { useState, useCallback } from 'react';
import { Upload, FileText, Image, Download, Trash2, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { Dataset } from '../../types';

const DatasetUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([
    {
      id: '1',
      filename: 'customer_data.csv',
      size: 2500000,
      uploadDate: '2024-01-15',
      status: 'ready',
      type: 'csv'
    },
    {
      id: '2',
      filename: 'product_images.zip',
      size: 15600000,
      uploadDate: '2024-01-14',
      status: 'processing',
      type: 'image'
    }
  ]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleFiles = (files: FileList) => {
    setIsUploading(true);
    setUploadProgress(0);
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          // Add new dataset to list
          const newDataset: Dataset = {
            id: Date.now().toString(),
            filename: files[0].name,
            size: files[0].size,
            uploadDate: new Date().toISOString().split('T')[0],
            status: 'ready',
            type: files[0].name.endsWith('.csv') ? 'csv' : 'image'
          };
          setDatasets(prev => [newDataset, ...prev]);
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: Dataset['status']) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-5 h-5 text-[#00FFA0]" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-[#FFB84D] animate-spin" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-[#FF3B6B]" />;
    }
  };

  const getFileIcon = (type: Dataset['type']) => {
    return type === 'csv' ? 
      <FileText className="w-6 h-6 text-[#00F3FF]" /> : 
      <Image className="w-6 h-6 text-[#FF00D0]" />;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Upload Your Training Data
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Support for CSV files and image datasets
        </p>
      </div>

      {/* Upload Zone */}
      <div className="max-w-2xl mx-auto">
        <div
          className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
            dragActive
              ? 'border-[#00F3FF] bg-[rgba(0,243,255,0.05)] shadow-[0_0_30px_rgba(0,243,255,0.2)]'
              : 'border-[#122033] hover:border-[#00F3FF]/50'
          } ${isUploading ? 'pointer-events-none opacity-75' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {/* Background Glow */}
          <div className="absolute inset-0 bg-gradient-to-br from-[rgba(0,243,255,0.02)] to-[rgba(255,0,208,0.02)] rounded-xl"></div>
          
          <div className="relative z-10 space-y-6">
            {/* Upload Icon */}
            <div className="mx-auto w-16 h-16 relative">
              <Upload className={`w-full h-full text-[#00F3FF] ${isUploading ? 'animate-bounce' : ''}`} />
              <div className="absolute inset-0 bg-[#00F3FF] blur-xl opacity-20 rounded-full"></div>
            </div>

            {/* Upload Text */}
            {isUploading ? (
              <div className="space-y-4">
                <p className="text-xl text-[#E6FBFF] font-semibold">Uploading...</p>
                <div className="max-w-xs mx-auto">
                  <div className="bg-[#121628] rounded-full h-2 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] h-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-[#9BD8FF] mt-2">{uploadProgress}% complete</p>
                </div>
              </div>
            ) : (
              <>
                <div>
                  <p className="text-xl text-[#E6FBFF] font-semibold mb-2">
                    Drag & drop files here or click to browse
                  </p>
                  <p className="text-[#9BD8FF]">Max 100MB per file</p>
                </div>

                {/* File Types */}
                <div className="flex justify-center space-x-4">
                  <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-sm text-[#00F3FF] font-medium">
                    CSV
                  </span>
                  <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-sm text-[#FF00D0] font-medium">
                    JPG
                  </span>
                  <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-sm text-[#00FFA0] font-medium">
                    PNG
                  </span>
                </div>
              </>
            )}
          </div>

          {/* Hidden File Input */}
          <input
            type="file"
            multiple
            accept=".csv,.jpg,.jpeg,.png,.zip"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isUploading}
          />
        </div>
      </div>

      {/* Uploaded Files List */}
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-[#E6FBFF]">
            Uploaded Datasets ({datasets.length})
          </h2>
        </div>

        {datasets.length > 0 ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 hover:border-[#00F3FF]/30 hover:shadow-[0_0_20px_rgba(0,243,255,0.1)] transition-all duration-300 group"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getFileIcon(dataset.type)}
                    {getStatusIcon(dataset.status)}
                  </div>
                  <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-1 text-[#9BD8FF] hover:text-[#00F3FF] transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                    <button className="p-1 text-[#9BD8FF] hover:text-[#FF3B6B] transition-colors">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="space-y-2">
                  <h3 className="text-[#E6FBFF] font-semibold truncate" title={dataset.filename}>
                    {dataset.filename}
                  </h3>
                  <div className="flex justify-between text-sm text-[#9BD8FF]">
                    <span>{formatFileSize(dataset.size)}</span>
                    <span>{dataset.uploadDate}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      dataset.status === 'ready' 
                        ? 'bg-[#00FFA0]/20 text-[#00FFA0]' 
                        : dataset.status === 'processing'
                        ? 'bg-[#FFB84D]/20 text-[#FFB84D]'
                        : 'bg-[#FF3B6B]/20 text-[#FF3B6B]'
                    }`}>
                      {dataset.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="mx-auto w-24 h-24 mb-6 relative">
              <Upload className="w-full h-full text-[#9BD8FF]/30" />
              <div className="absolute inset-0 bg-[#9BD8FF] blur-xl opacity-10 rounded-full"></div>
            </div>
            <h3 className="text-xl text-[#9BD8FF] font-semibold mb-2">No datasets uploaded yet</h3>
            <p className="text-[#9BD8FF]/70 mb-6">Start by uploading your first dataset above</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetUpload;