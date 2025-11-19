import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileText, Image, Download, Trash2, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { useDataset } from '../../hooks/useDataset';

const DatasetUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const { datasets, progress, upload, remove, selectedDatasetPath, selectedDatasetName, setSelectedDataset } = useDataset();
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  const handleFiles = async (files: FileList) => {
    setIsUploading(true);
    await upload(Array.from(files));
    setIsUploading(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: 'ready' | 'processing' | 'error') => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-5 h-5 text-[#00FFA0]" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-[#FFB84D] animate-spin" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-[#FF3B6B]" />;
    }
  };

  const getFileIcon = (type: 'csv' | 'image') => {
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
        <div className="mt-3 flex items-center justify-center gap-2 text-sm">
          {selectedDatasetName ? (
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md" title={selectedDatasetName}>
              Dataset: {selectedDatasetName}
            </span>
          ) : (
            <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-[#9BD8FF]">No dataset selected</span>
          )}
        </div>
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
          onClick={() => {
            if (!isUploading) fileInputRef.current?.click();
          }}
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
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-[#9BD8FF] mt-2">{progress}% complete</p>
                </div>
              </div>
            ) : (
              <>
                <div>
                  <p className="text-xl text-[#E6FBFF] font-semibold mb-2">
                    Drag & drop files here or click to browse
                  </p>
                  <p className="text-[#9BD8FF]">Max {Math.round(Number(process.env.REACT_APP_MAX_FILE_SIZE || '104857600') / (1024 * 1024))}MB per file · Allowed: CSV, JPG, JPEG, PNG, BMP</p>
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
            accept=".csv,.jpg,.jpeg,.png,.bmp"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
            ref={fileInputRef}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
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
              <button
                type="button"
                key={dataset.filename}
                onClick={() => setSelectedDataset({ filename: dataset.filename, path: dataset.path })}
                className={`text-left bg-[#121628]/50 border rounded-xl p-6 hover:shadow-[0_0_20px_rgba(0,243,255,0.1)] transition-all duration-300 group w-full ${
                  selectedDatasetPath === dataset.path
                    ? 'border-[#00F3FF] shadow-[0_0_20px_rgba(0,243,255,0.3)]'
                    : 'border-[#122033] hover:border-[#00F3FF]/30'
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getFileIcon(dataset.filename.endsWith('.csv') ? 'csv' : 'image')}
                    {getStatusIcon('ready')}
                  </div>
                  <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-1 text-[#9BD8FF] hover:text-[#00F3FF] transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); remove(dataset.filename); }} className="p-1 text-[#9BD8FF] hover:text-[#FF3B6B] transition-colors">
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
                    <span>uploaded</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${
                        selectedDatasetPath === dataset.path
                          ? 'bg-[#00F3FF]/20 text-[#00F3FF]'
                          : 'bg-[#00FFA0]/20 text-[#00FFA0]'
                      }`}
                    >
                      {selectedDatasetPath === dataset.path ? 'SELECTED' : 'READY'}
                    </span>
                  </div>
                </div>
              </button>
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