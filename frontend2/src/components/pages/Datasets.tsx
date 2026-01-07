import React, { useState, useEffect } from 'react';
import {
  Database,
  FileText,
  Image,
  Folder,
  Trash2,
  RefreshCw,
  Search,
  Eye,
  BarChart3,
  Table,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  Rows3,
  Columns3,
  Upload,
  X,
  FileSpreadsheet,
  FileType,
  Tag,
  Hash,
  Type,
  ListTree,
  LayoutGrid,
} from 'lucide-react';
import { datasetService, DatasetFile, DatasetPreview } from '../../api/services/datasetService';

const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<DatasetFile | null>(null);
  const [previewData, setPreviewData] = useState<DatasetPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const response = await datasetService.list();
      setDatasets(response.files || []);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (filename: string) => {
    if (!confirm(`Delete "${filename}"?`)) return;
    try {
      await datasetService.delete(filename);
      loadDatasets();
      if (selectedDataset?.filename === filename) {
        setSelectedDataset(null);
        setPreviewData(null);
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const loadDatasetPreview = async (dataset: DatasetFile) => {
    setSelectedDataset(dataset);
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewData(null);
    
    try {
      const preview = await datasetService.preview(dataset.filename, 8);
      setPreviewData(preview);
    } catch (error: any) {
      console.error('Failed to load preview:', error);
      setPreviewError(error.message || 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const getFileIcon = (filename: string, type?: string) => {
    if (type === 'folder') return <Folder className="w-5 h-5 text-amber-400" />;
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'csv':
        return <FileSpreadsheet className="w-5 h-5 text-emerald-400" />;
      case 'txt':
        return <FileText className="w-5 h-5 text-blue-400" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'bmp':
        return <Image className="w-5 h-5 text-purple-400" />;
      default:
        return <FileType className="w-5 h-5 text-gray-400" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
    if (bytes >= 1024) {
      return `${(bytes / 1024).toFixed(2)} KB`;
    }
    return `${bytes} B`;
  };

  const getFileTypeLabel = (filename: string, type?: string) => {
    if (type === 'folder') return 'Image Folder';
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'csv':
        return 'CSV Dataset';
      case 'txt':
        return 'Text File';
      case 'jpg':
      case 'jpeg':
      case 'png':
        return 'Image';
      default:
        return 'File';
    }
  };

  const filteredDatasets = datasets.filter((d) =>
    d.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalSize = datasets.reduce((acc, d) => acc + (d.size || 0), 0);

  // Render CSV Preview
  const renderCsvPreview = () => {
    if (!previewData || previewData.file_type !== 'csv') return null;
    
    return (
      <>
        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Rows3 className="w-3 h-3" />
              Rows
            </div>
            <p className="text-lg font-semibold text-white">{previewData.num_rows?.toLocaleString()}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Columns3 className="w-3 h-3" />
              Columns
            </div>
            <p className="text-lg font-semibold text-white">{previewData.num_columns}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Tag className="w-3 h-3" />
              Target Classes
            </div>
            <p className="text-lg font-semibold text-white">{previewData.unique_targets}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <AlertCircle className="w-3 h-3" />
              Missing Values
            </div>
            <p className={`text-lg font-semibold ${previewData.total_missing === 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {previewData.total_missing}
            </p>
          </div>
        </div>

        {/* Data Quality Bar */}
        <div className="bg-[#0b1220]/50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-[#9BD8FF]/60">Data Quality</span>
            {previewData.total_missing === 0 ? (
              <span className="flex items-center gap-1 text-xs text-emerald-400">
                <CheckCircle2 className="w-3 h-3" /> Complete
              </span>
            ) : (
              <span className="flex items-center gap-1 text-xs text-amber-400">
                <AlertCircle className="w-3 h-3" /> {previewData.total_missing} missing values
              </span>
            )}
          </div>
          <div className="w-full bg-[#1e3a5f] rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${
                previewData.total_missing === 0 ? 'bg-emerald-500' : 'bg-amber-500'
              }`}
              style={{
                width: `${Math.max(5, 100 - ((previewData.total_missing || 0) / (previewData.num_rows || 1)) * 100)}%`,
              }}
            />
          </div>
        </div>

        {/* Target Column Info */}
        {previewData.target_column && previewData.target_values && (
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-2">
              <ListTree className="w-3 h-3" />
              Target: <span className="text-[#00F3FF]">{previewData.target_column}</span>
            </div>
            <div className="space-y-1">
              {Object.entries(previewData.target_values).slice(0, 5).map(([label, count]) => (
                <div key={label} className="flex items-center justify-between text-xs">
                  <span className="text-white/80 truncate max-w-[150px]">{String(label)}</span>
                  <span className="text-[#9BD8FF]/60">{count}</span>
                </div>
              ))}
              {Object.keys(previewData.target_values).length > 5 && (
                <p className="text-xs text-[#9BD8FF]/40 text-center">
                  +{Object.keys(previewData.target_values).length - 5} more classes
                </p>
              )}
            </div>
          </div>
        )}

        {/* Data Preview Table */}
        {previewData.preview && previewData.preview.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-white mb-2">Data Preview</h4>
            <div className="bg-[#0b1220]/50 rounded-lg overflow-x-auto max-h-64">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-[#0b1220]">
                  <tr className="border-b border-[#1e3a5f]">
                    {previewData.preview[0].map((header, i) => (
                      <th key={i} className="px-2 py-1.5 text-left text-[#00F3FF] font-medium whitespace-nowrap">
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.preview.slice(1).map((row, i) => (
                    <tr key={i} className="border-b border-[#1e3a5f]/30 hover:bg-[#1e3a5f]/20">
                      {row.map((cell, j) => (
                        <td key={j} className="px-2 py-1.5 text-white/80 whitespace-nowrap max-w-[100px] truncate">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </>
    );
  };

  // Render Text Preview
  const renderTextPreview = () => {
    if (!previewData || previewData.file_type !== 'text') return null;
    
    return (
      <>
        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Rows3 className="w-3 h-3" />
              Lines
            </div>
            <p className="text-lg font-semibold text-white">{previewData.num_lines?.toLocaleString()}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Type className="w-3 h-3" />
              Characters
            </div>
            <p className="text-lg font-semibold text-white">{previewData.num_characters?.toLocaleString()}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <FileText className="w-3 h-3" />
              Words
            </div>
            <p className="text-lg font-semibold text-white">{previewData.num_words?.toLocaleString()}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Hash className="w-3 h-3" />
              Vocab Size
            </div>
            <p className="text-lg font-semibold text-white">{previewData.vocab_size}</p>
          </div>
        </div>

        {/* Format Info */}
        <div className="bg-[#0b1220]/50 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-[#9BD8FF]/60">Format</span>
            <span className={`text-xs px-2 py-0.5 rounded ${
              previewData.format === 'tab_separated' ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'
            }`}>
              {previewData.format === 'tab_separated' ? 'Classification (Tab-Separated)' : 'Plain Text'}
            </span>
          </div>
        </div>

        {/* Detected Labels */}
        {previewData.detected_labels && previewData.detected_labels.length > 0 && (
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-2">
              <Tag className="w-3 h-3" />
              Detected Labels
            </div>
            <div className="flex flex-wrap gap-1">
              {previewData.detected_labels.map((label, i) => (
                <span key={i} className="text-xs px-2 py-0.5 bg-[#1e3a5f] text-white/80 rounded">
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Sample Vocabulary */}
        {previewData.sample_vocab && (
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-2">
              <Hash className="w-3 h-3" />
              Character Vocabulary Sample
            </div>
            <p className="text-xs text-white/60 font-mono break-all">
              {previewData.sample_vocab.map(c => c === '\n' ? '\\n' : c === '\t' ? '\\t' : c === ' ' ? '␣' : c).join('')}
            </p>
          </div>
        )}

        {/* Text Preview */}
        {previewData.preview && previewData.preview.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-white mb-2">Content Preview</h4>
            <div className="bg-[#0b1220]/50 rounded-lg p-3 max-h-64 overflow-y-auto">
              <pre className="text-xs text-white/80 whitespace-pre-wrap font-mono">
                {previewData.preview.map(row => row[0]).join('\n')}
              </pre>
            </div>
          </div>
        )}
      </>
    );
  };

  // Render Image Folder Preview
  const renderImageFolderPreview = () => {
    if (!previewData || previewData.file_type !== 'image_folder') return null;
    
    return (
      <>
        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <Image className="w-3 h-3" />
              Total Images
            </div>
            <p className="text-lg font-semibold text-white">{previewData.total_images?.toLocaleString()}</p>
          </div>
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-1">
              <LayoutGrid className="w-3 h-3" />
              Classes
            </div>
            <p className="text-lg font-semibold text-white">{previewData.num_classes}</p>
          </div>
        </div>

        {/* Class Distribution */}
        {previewData.classes && Object.keys(previewData.classes).length > 0 && (
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-3">
              <Tag className="w-3 h-3" />
              Class Distribution
            </div>
            <div className="space-y-2">
              {Object.entries(previewData.classes).map(([className, count]) => {
                const percentage = ((count as number) / (previewData.total_images || 1)) * 100;
                return (
                  <div key={className}>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-white/80 truncate max-w-[120px]">{className}</span>
                      <span className="text-[#9BD8FF]/60">{count as number} ({percentage.toFixed(1)}%)</span>
                    </div>
                    <div className="w-full bg-[#1e3a5f] rounded-full h-1.5">
                      <div
                        className="h-1.5 rounded-full bg-gradient-to-r from-[#00F3FF] to-[#FF00D0]"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Sample Images Info */}
        {previewData.sample_images && previewData.sample_images.length > 0 && (
          <div className="bg-[#0b1220]/50 rounded-lg p-3">
            <div className="flex items-center gap-2 text-[#9BD8FF]/60 text-xs mb-2">
              <Image className="w-3 h-3" />
              Sample Images
            </div>
            <div className="space-y-2">
              {previewData.sample_images.map((img, i) => (
                <div key={i} className="flex items-center justify-between text-xs p-2 bg-[#1e3a5f]/30 rounded">
                  <div className="flex items-center gap-2">
                    <Image className="w-4 h-4 text-purple-400" />
                    <span className="text-white/80 truncate max-w-[120px]">{img.filename}</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#9BD8FF]/60">
                    <span>{img.size}</span>
                    <span className="text-xs px-1.5 py-0.5 bg-[#0b1220] rounded">{img.mode}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </>
    );
  };

  return (
    <div className="flex h-[calc(100vh-180px)]">
      {/* Main Content */}
      <div className={`flex-1 space-y-6 ${selectedDataset ? 'pr-6' : ''} overflow-y-auto`}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
              Datasets
            </h1>
            <p className="text-[#9BD8FF]/70 mt-1">
              Browse and manage your uploaded datasets
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'upload' }))}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg text-white font-medium hover:opacity-90 transition-all"
            >
              <Upload className="w-4 h-4" />
              Upload New
            </button>
            <button
              onClick={loadDatasets}
              className="flex items-center gap-2 px-4 py-2 bg-[#0f1a2e] border border-[#1e3a5f] rounded-lg text-[#9BD8FF] hover:bg-[#152238] hover:border-[#00F3FF]/50 transition-all"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-cyan-500/20 rounded-lg">
                <Database className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{datasets.length}</p>
                <p className="text-xs text-[#9BD8FF]/60">Total Datasets</p>
              </div>
            </div>
          </div>
          <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-500/20 rounded-lg">
                <HardDrive className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{formatFileSize(totalSize)}</p>
                <p className="text-xs text-[#9BD8FF]/60">Total Storage</p>
              </div>
            </div>
          </div>
          <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Folder className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">
                  {datasets.filter((d) => d.type === 'folder').length}
                </p>
                <p className="text-xs text-[#9BD8FF]/60">Image Folders</p>
              </div>
            </div>
          </div>
        </div>

        {/* Search and View Toggle */}
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#9BD8FF]/50" />
            <input
              type="text"
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-[#0b1220] border border-[#1e3a5f] rounded-lg text-white placeholder-[#9BD8FF]/40 focus:outline-none focus:border-[#00F3FF]/50"
            />
          </div>
          <div className="flex bg-[#0b1220] border border-[#1e3a5f] rounded-lg p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded ${viewMode === 'grid' ? 'bg-[#1e3a5f] text-[#00F3FF]' : 'text-[#9BD8FF]/60'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded ${viewMode === 'list' ? 'bg-[#1e3a5f] text-[#00F3FF]' : 'text-[#9BD8FF]/60'}`}
            >
              <Table className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Dataset List */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#00F3FF]"></div>
          </div>
        ) : filteredDatasets.length === 0 ? (
          <div className="text-center py-20 bg-[#0b1220]/50 border border-[#1e3a5f] rounded-xl">
            <Database className="w-16 h-16 mx-auto text-[#9BD8FF]/30 mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Datasets Found</h3>
            <p className="text-[#9BD8FF]/60 mb-4">
              {searchQuery ? 'Try a different search term' : 'Upload a dataset to get started'}
            </p>
            <button
              onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'upload' }))}
              className="px-4 py-2 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg text-white font-medium"
            >
              Upload Dataset
            </button>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredDatasets.map((dataset) => (
              <div
                key={dataset.filename}
                onClick={() => loadDatasetPreview(dataset)}
                className={`bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border rounded-xl p-4 cursor-pointer transition-all hover:border-[#00F3FF]/50 ${
                  selectedDataset?.filename === dataset.filename
                    ? 'border-[#00F3FF] ring-1 ring-[#00F3FF]/30'
                    : 'border-[#1e3a5f]'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-[#0b1220] rounded-lg">
                      {getFileIcon(dataset.filename, dataset.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-white truncate">{dataset.filename}</h3>
                      <p className="text-xs text-[#9BD8FF]/60">{getFileTypeLabel(dataset.filename, dataset.type)}</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[#9BD8FF]/60">{formatFileSize(dataset.size)}</span>
                  {dataset.image_count && (
                    <span className="text-[#9BD8FF]/60">{dataset.image_count} images</span>
                  )}
                </div>
                <div className="mt-3 pt-3 border-t border-[#1e3a5f]/50 flex justify-end gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      loadDatasetPreview(dataset);
                    }}
                    className="p-2 text-[#9BD8FF]/60 hover:text-[#00F3FF] hover:bg-[#1e3a5f]/50 rounded transition-all"
                    title="Preview"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(dataset.filename);
                    }}
                    className="p-2 text-[#9BD8FF]/60 hover:text-red-400 hover:bg-red-500/10 rounded transition-all"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-[#0b1220]/50 border border-[#1e3a5f] rounded-xl overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[#1e3a5f]">
                  <th className="text-left px-4 py-3 text-xs font-medium text-[#9BD8FF]/70">Name</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-[#9BD8FF]/70">Type</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-[#9BD8FF]/70">Size</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-[#9BD8FF]/70">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredDatasets.map((dataset) => (
                  <tr
                    key={dataset.filename}
                    onClick={() => loadDatasetPreview(dataset)}
                    className={`border-b border-[#1e3a5f]/30 cursor-pointer hover:bg-[#1e3a5f]/20 ${
                      selectedDataset?.filename === dataset.filename ? 'bg-[#1e3a5f]/30' : ''
                    }`}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        {getFileIcon(dataset.filename, dataset.type)}
                        <span className="text-white">{dataset.filename}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[#9BD8FF]/70">{getFileTypeLabel(dataset.filename, dataset.type)}</td>
                    <td className="px-4 py-3 text-[#9BD8FF]/70">{formatFileSize(dataset.size)}</td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            loadDatasetPreview(dataset);
                          }}
                          className="p-1.5 text-[#9BD8FF]/60 hover:text-[#00F3FF] hover:bg-[#1e3a5f]/50 rounded"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(dataset.filename);
                          }}
                          className="p-1.5 text-[#9BD8FF]/60 hover:text-red-400 hover:bg-red-500/10 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Preview Sidebar */}
      {selectedDataset && (
        <div className="w-[420px] bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl overflow-hidden flex flex-col">
          {/* Sidebar Header */}
          <div className="p-4 border-b border-[#1e3a5f] flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getFileIcon(selectedDataset.filename, selectedDataset.type)}
              <div>
                <h3 className="font-medium text-white truncate max-w-[220px]">{selectedDataset.filename}</h3>
                <p className="text-xs text-[#9BD8FF]/60">{formatFileSize(selectedDataset.size)}</p>
              </div>
            </div>
            <button
              onClick={() => {
                setSelectedDataset(null);
                setPreviewData(null);
                setPreviewError(null);
              }}
              className="p-1.5 text-[#9BD8FF]/60 hover:text-white hover:bg-[#1e3a5f] rounded"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Sidebar Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {previewLoading ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#00F3FF]"></div>
              </div>
            ) : previewError ? (
              <div className="text-center py-10">
                <AlertCircle className="w-10 h-10 text-red-400 mx-auto mb-3" />
                <p className="text-red-400 text-sm">{previewError}</p>
                <button
                  onClick={() => loadDatasetPreview(selectedDataset)}
                  className="mt-3 px-4 py-2 bg-[#1e3a5f] text-white rounded-lg text-sm hover:bg-[#2a4a6f]"
                >
                  Retry
                </button>
              </div>
            ) : previewData ? (
              <>
                {/* Render based on file type */}
                {previewData.file_type === 'csv' && renderCsvPreview()}
                {previewData.file_type === 'text' && renderTextPreview()}
                {previewData.file_type === 'image_folder' && renderImageFolderPreview()}

                {/* Actions */}
                <div className="space-y-2 pt-2">
                  <button
                    onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'validation' }))}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-[#00F3FF]/20 to-[#FF00D0]/20 border border-[#00F3FF]/30 rounded-lg text-[#00F3FF] hover:border-[#00F3FF] transition-all"
                  >
                    <CheckCircle2 className="w-4 h-4" />
                    Validate Dataset
                  </button>
                  <button
                    onClick={() => handleDelete(selectedDataset.filename)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 hover:border-red-500 transition-all"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Dataset
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};

export default Datasets;
