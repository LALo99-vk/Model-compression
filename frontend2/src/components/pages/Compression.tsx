import { useState, useEffect } from 'react';
import { 
  Scissors, 
  Zap, 
  CheckCircle, 
  Clock, 
  ArrowRight, 
  AlertCircle,
  Loader,
  GitBranch,
  Layers,
  BarChart,
  Download
} from 'lucide-react';
import { compressionService } from '../../api/services/compressionService';
import { trainingService } from '../../api/services/trainingService';
import { useToast } from '../ui/ToastContainer';
import { useAppStore } from '../../store/useAppStore';

interface ModelInfo {
  model_path: string;
  model_type: string;
  file_size_mb: number;
  parameters: number;
  architecture: string;
}

interface CompressionResult {
  status: string;
  original: {
    size_mb: number;
    parameters: number;
    architecture: string;
  };
  compressed?: {
    size_mb: number;
    parameters: number;
    architecture: string;
  };
  reduction_percent?: number;
  success: boolean;
  failure_reason?: string;
}

const Compression = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isCompressing, setIsCompressing] = useState(false);
  const [compressionResult, setCompressionResult] = useState<CompressionResult | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [autoStarted, setAutoStarted] = useState(false);
  
  const { showSuccess, showError, showInfo } = useToast();
  const selectedDatasetName = useAppStore((s) => s.selectedDatasetName);
  const selectedModelConfig = useAppStore((s) => s.selectedModel);

  // STEP 5: Auto-detect trained model and load info
  // Reset state when component mounts to avoid showing stale data
  useEffect(() => {
    // Clear any previous compression results
    setCompressionResult(null);
    setModelInfo(null);
    setValidationError(null);
    setAutoStarted(false);
    
    // Load fresh model info
    loadTrainedModelInfo();
  }, []);

  // Show model info - NO auto-start
  useEffect(() => {
    if (modelInfo && !autoStarted) {
      showInfo('Model Loaded', 'Ready to compress. Click "Start Compression" below.');
      setAutoStarted(true);
    }
  }, [modelInfo, autoStarted]);

  // Show success message - NO auto-navigation
  useEffect(() => {
    if (compressionResult && compressionResult.success) {
      showSuccess('Compression Complete', 'Model compressed successfully! Click "View Results" to see comparison.');
      // REMOVED: Auto-navigation - user will click button manually
    }
  }, [compressionResult]);

  const loadTrainedModelInfo = async () => {
    try {
      // STEP 5: Automatic handoff - get trained model info
      const trainingLogs = await trainingService.logs();
      
      if (!trainingLogs) {
        setValidationError('No trained model found. Please train a model first.');
        showError('No Model', 'No trained model found. Please complete training first.');
        return;
      }

      const modelType = selectedModelConfig?.model_type || 'decision_tree';
      const modelSizeMB = trainingLogs.model_size_mb || 0;
      const totalParams = trainingLogs.total_parameters || 
                         trainingLogs.num_parameters || 
                         trainingLogs.trainable_parameters || 0;

      // GLOBAL HARD RULE: Validate trained model before compression
      if (modelSizeMB <= 0) {
        setValidationError('Invalid model: File size is 0 MB');
        showError('Invalid Model', 'Trained model has invalid file size (0 MB). Please retrain.');
        return;
      }

      if (totalParams <= 0) {
        setValidationError('Invalid model: Parameter count is 0');
        showError('Invalid Model', 'Trained model has 0 parameters. Please retrain.');
        return;
      }

      const architecture = modelType === 'cnn' ? 'CNN' : 
                          modelType === 'rnn' ? 'RNN' : 
                          modelType === 'decision_tree' ? 'Decision Tree' : 
                          'Unknown';

      const info: ModelInfo = {
        model_path: 'models/original_model.pkl', // Backend handles actual path
        model_type: modelType,
        file_size_mb: modelSizeMB,
        parameters: totalParams,
        architecture: architecture
      };

      setModelInfo(info);
      setValidationError(null);
      
      console.log('✅ Loaded model info for compression:', {
        type: modelType,
        architecture: architecture,
        size_mb: modelSizeMB,
        parameters: totalParams
      });
      
      showInfo('Model Loaded', `Found trained ${architecture} model: ${modelSizeMB.toFixed(3)} MB, ${totalParams.toLocaleString()} params`);
      
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to load model info';
      setValidationError(errorMsg);
      showError('Load Failed', errorMsg);
    }
  };

  const startCompression = async () => {
    if (!modelInfo) {
      showError('No Model', 'No model available for compression');
      return;
    }

    setIsCompressing(true);
    setCompressionResult(null);

    try {
      showInfo('Compressing', `Applying ${getCompressionMethodsForModel(modelInfo.model_type).join(', ')}...`);
      
      // Call comprehensive compression endpoint
      const result = await compressionService.compressComprehensive({
        pruning_amount: 0.35,
        quantization_bits: 8,
        distillation_temperature: 3.0,
        distillation_alpha: 0.5
      });

      // STEP 6: Validate compression result
      const validation = validateCompressionResult(result);
      
      if (!validation.valid) {
        setCompressionResult({
          status: 'failed',
          original: {
            size_mb: modelInfo.file_size_mb,
            parameters: modelInfo.parameters,
            architecture: modelInfo.architecture
          },
          success: false,
          failure_reason: validation.reason
        });
        showError('Compression Failed', validation.reason || 'Compression did not meet validation criteria');
        return;
      }

      // Extract result from backend response
      const comparisonReport = result.comparison_report || result;
      
      setCompressionResult({
        status: 'completed',
        original: {
          size_mb: comparisonReport.original?.size_mb || modelInfo.file_size_mb,
          parameters: comparisonReport.original?.parameters || modelInfo.parameters,
          architecture: comparisonReport.original?.architecture || modelInfo.architecture
        },
        compressed: {
          size_mb: comparisonReport.compressed?.size_mb || 0,
          parameters: comparisonReport.compressed?.parameters || 0,
          architecture: comparisonReport.compressed?.architecture || modelInfo.architecture + ' (Compressed)'
        },
        reduction_percent: comparisonReport.reduction_percent || 0,
        success: comparisonReport.success || false,
        failure_reason: comparisonReport.failure_reason
      });

      if (comparisonReport.success) {
        showSuccess(
          'Compression Successful',
          `Reduced size by ${comparisonReport.reduction_percent?.toFixed(1)}%`
        );
      } else {
        showError('Compression Failed', comparisonReport.failure_reason || 'Unknown error');
      }

    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Compression failed';
      showError('Compression Failed', errorMsg);
      setCompressionResult({
        status: 'failed',
        original: {
          size_mb: modelInfo.file_size_mb,
          parameters: modelInfo.parameters,
          architecture: modelInfo.architecture
        },
        success: false,
        failure_reason: errorMsg
      });
    } finally {
      setIsCompressing(false);
    }
  };

  // STEP 6: Get valid compression methods for each model type
  const getCompressionMethodsForModel = (modelType: string): string[] => {
    switch (modelType) {
      case 'decision_tree':
        return [
          'Cost-Complexity Pruning',
          'Depth/Node Reduction',
          'Tree Distillation',
          'Joblib/Gzip Compression'
        ];
      case 'cnn':
        return [
          'Weight Pruning',
          'INT8 Quantization',
          'CNN Distillation'
        ];
      case 'rnn':
        return [
          'Weight Pruning',
          'Dynamic INT8 Quantization',
          'RNN Distillation'
        ];
      default:
        return ['Auto-detected methods'];
    }
  };

  // STEP 6: Validate compression result
  const validateCompressionResult = (result: any): { valid: boolean; reason?: string } => {
    const comparisonReport = result.comparison_report || result;
    
    // GLOBAL HARD RULE: Compressed model must be valid
    if (!comparisonReport.compressed) {
      return { valid: false, reason: 'No compressed model generated' };
    }

    const compressedSizeMB = comparisonReport.compressed.size_mb || 0;
    const compressedParams = comparisonReport.compressed.parameters || 0;
    const originalSizeMB = comparisonReport.original?.size_mb || modelInfo?.file_size_mb || 0;
    const originalParams = comparisonReport.original?.parameters || modelInfo?.parameters || 0;

    // GLOBAL HARD RULE: Never output 0 MB models
    if (compressedSizeMB <= 0) {
      return { valid: false, reason: 'Compressed model has 0 MB file size (invalid)' };
    }

    // GLOBAL HARD RULE: Never output 0 parameters
    if (compressedParams <= 0) {
      return { valid: false, reason: 'Compressed model has 0 parameters (invalid)' };
    }

    // GLOBAL HARD RULE: Compressed must be smaller than original
    if (compressedSizeMB >= originalSizeMB) {
      return { valid: false, reason: `Compressed model (${compressedSizeMB.toFixed(2)} MB) is not smaller than original (${originalSizeMB.toFixed(2)} MB)` };
    }

    if (compressedParams >= originalParams) {
      return { valid: false, reason: `Compressed parameters (${compressedParams}) are not less than original (${originalParams})` };
    }

    // GLOBAL HARD RULE: Must not be empty
    if (!comparisonReport.success) {
      return { valid: false, reason: comparisonReport.failure_reason || 'Compression failed' };
    }

    return { valid: true };
  };

  const getModelIcon = () => {
    if (!modelInfo) return <Layers className="w-12 h-12 text-[#9BD8FF]" />;
    
    switch (modelInfo.model_type) {
      case 'cnn':
        return <Layers className="w-12 h-12 text-[#00F3FF]" />;
      case 'rnn':
        return <BarChart className="w-12 h-12 text-[#FF00D0]" />;
      case 'decision_tree':
        return <GitBranch className="w-12 h-12 text-[#00FFA0]" />;
      default:
        return <Layers className="w-12 h-12 text-[#9BD8FF]" />;
    }
  };

  // If validation error, show error state
  if (validationError) {
    return (
      <div className="space-y-8">
        <div className="text-center space-y-3">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Model Compression
          </h1>
          <p className="text-lg text-[#9BD8FF]">
            Automatic Model-Type-Aware Compression
          </p>
        </div>

        <div className="max-w-2xl mx-auto bg-[#121628]/50 border border-[#FF3B6B]/30 rounded-xl p-8 text-center">
          <AlertCircle className="w-16 h-16 text-[#FF3B6B] mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-[#E6FBFF] mb-2">Validation Error</h3>
          <p className="text-[#9BD8FF] mb-6">{validationError}</p>
          <div className="flex flex-col gap-3">
            <p className="text-sm text-[#9BD8FF]">Please ensure:</p>
            <ul className="text-left text-sm text-[#9BD8FF] list-disc list-inside space-y-1">
              <li>Dataset validation passed</li>
              <li>Model selection completed</li>
              <li>Training completed successfully</li>
              <li>Model file size {'>'} 0 MB</li>
              <li>Model parameters {'>'} 0</li>
            </ul>
          </div>
          <button
            onClick={loadTrainedModelInfo}
            className="mt-6 px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105"
          >
            Retry Loading Model
          </button>
        </div>
      </div>
    );
  }

  // If no model info yet, show loading
  if (!modelInfo) {
    return (
      <div className="space-y-8">
        <div className="text-center space-y-3">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Model Compression
          </h1>
          <p className="text-lg text-[#9BD8FF]">
            Loading trained model...
          </p>
        </div>

        <div className="flex justify-center">
          <Loader className="w-16 h-16 text-[#00F3FF] animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Model Compression
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Automatic Model-Type-Aware Compression
        </p>
        {selectedDatasetName && (
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md">
              Dataset: {selectedDatasetName}
            </span>
          </div>
        )}
      </div>

      {/* Model Info Card */}
      <div className="max-w-4xl mx-auto bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <div className="flex items-start gap-6">
          <div className="p-4 bg-gradient-to-br from-[#00F3FF]/20 to-transparent rounded-xl border border-[#122033]">
            {getModelIcon()}
          </div>
          <div className="flex-1">
            <h3 className="text-2xl font-bold text-[#E6FBFF] mb-2">{modelInfo.architecture}</h3>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div>
                <p className="text-sm text-[#9BD8FF]">File Size</p>
                <p className="text-lg font-semibold text-[#E6FBFF]">{modelInfo.file_size_mb.toFixed(2)} MB</p>
              </div>
              <div>
                <p className="text-sm text-[#9BD8FF]">Parameters</p>
                <p className="text-lg font-semibold text-[#E6FBFF]">{modelInfo.parameters.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Download Original Model Button - Show if model is available */}
      {modelInfo && !isCompressing && (
        <div className="max-w-4xl mx-auto">
          <button
            onClick={async () => {
              try {
                const modelType = modelInfo.model_type || 'unknown';
                let fileExtension = '.pkl'; // Default for Decision Trees
                
                if (modelType === 'rnn' || modelType === 'cnn' || modelType === 'lstm' || modelType === 'gru') {
                  fileExtension = '.pt'; // PyTorch models
                }
                
                console.log(`📥 Downloading ${modelType} model as: original_model${fileExtension}`);
                
                const response = await fetch('http://localhost:8000/api/model/download/original');
                
                if (!response.ok) {
                  throw new Error(`Download failed: ${response.statusText}`);
                }
                
                const blob = await response.blob();
                const sizeMB = (blob.size / (1024 * 1024)).toFixed(4);
                console.log(`✅ Downloaded ${blob.size.toLocaleString()} bytes (${sizeMB} MB)`);
                
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `original_model${fileExtension}`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                
                showSuccess('Download Complete', `Original model downloaded (${sizeMB} MB)`);
              } catch (error: any) {
                console.error('Download failed:', error);
                showError('Download Failed', 'Could not download original model');
              }
            }}
            className="w-full px-8 py-3 bg-gradient-to-r from-[#00F3FF] to-[#0088FF] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
          >
            <Download className="w-5 h-5" />
            Download Original Model
          </button>
        </div>
      )}

      {/* Start Compression Button - Only show if not compressing and no result yet */}
      {!isCompressing && !compressionResult && (
        <div className="max-w-4xl mx-auto">
          <button
            onClick={startCompression}
            className="w-full px-8 py-4 bg-gradient-to-r from-[#00FFA0] to-[#00D67F] rounded-lg font-semibold text-white text-lg shadow-lg hover:shadow-[0_0_30px_rgba(0,255,160,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-3"
          >
            <Zap className="w-6 h-6" />
            Start Compression
            <ArrowRight className="w-6 h-6" />
          </button>
        </div>
      )}

      {/* Compression Methods Card */}
      <div className="max-w-4xl mx-auto bg-[#121628]/50 border border-[#00FFA0]/30 rounded-xl p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-[#00FFA0]/10 rounded-lg">
            <Zap className="w-8 h-8 text-[#00FFA0]" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-semibold text-[#E6FBFF] mb-2">
              Model-Specific Compression Techniques
            </h3>
            <p className="text-[#9BD8FF] mb-4">
              Applying {modelInfo.model_type === 'decision_tree' ? 'tree-specific' : 'neural network'} compression methods:
            </p>
            <ul className="space-y-2">
              {getCompressionMethodsForModel(modelInfo.model_type).map((method, idx) => (
                <li key={idx} className="flex items-center gap-2 text-[#E6FBFF]">
                  <CheckCircle className="w-4 h-4 text-[#00FFA0]" />
                  {method}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Compression Status */}
      {isCompressing && (
        <div className="max-w-4xl mx-auto bg-[#121628]/50 border border-[#00F3FF]/30 rounded-xl p-8 text-center">
          <Loader className="w-16 h-16 text-[#00F3FF] animate-spin mx-auto mb-4" />
          <h3 className="text-2xl font-bold text-[#E6FBFF] mb-2">Compressing Model...</h3>
          <p className="text-[#9BD8FF]">
            Applying optimizations and validating results...
          </p>
        </div>
      )}

      {/* Compression Result */}
      {compressionResult && !isCompressing && (
        <div className={`max-w-4xl mx-auto bg-[#121628]/50 border rounded-xl p-8 ${
          compressionResult.success
            ? 'border-[#00FFA0]/30'
            : 'border-[#FF3B6B]/30'
        }`}>
          <div className="text-center mb-6">
            {compressionResult.success ? (
              <CheckCircle className="w-16 h-16 text-[#00FFA0] mx-auto mb-4" />
            ) : (
              <AlertCircle className="w-16 h-16 text-[#FF3B6B] mx-auto mb-4" />
            )}
            <h3 className="text-2xl font-bold text-[#E6FBFF] mb-2">
              {compressionResult.success ? 'Compression Successful!' : 'Compression Failed'}
            </h3>
            {compressionResult.failure_reason && (
              <p className="text-[#FF3B6B]">{compressionResult.failure_reason}</p>
            )}
          </div>

          {compressionResult.success && compressionResult.compressed && (
            <div className="grid grid-cols-3 gap-6 mb-6">
              <div className="bg-[#0b1220]/50 border border-[#122033] rounded-lg p-4 text-center">
                <p className="text-sm text-[#9BD8FF] mb-2">Original Size</p>
                <p className="text-2xl font-bold text-[#E6FBFF]">{compressionResult.original.size_mb.toFixed(2)} MB</p>
              </div>
              <div className="bg-[#0b1220]/50 border border-[#122033] rounded-lg p-4 text-center">
                <p className="text-sm text-[#9BD8FF] mb-2">Compressed Size</p>
                <p className="text-2xl font-bold text-[#00FFA0]">{compressionResult.compressed.size_mb.toFixed(2)} MB</p>
              </div>
              <div className="bg-[#0b1220]/50 border border-[#122033] rounded-lg p-4 text-center">
                <p className="text-sm text-[#9BD8FF] mb-2">Reduction</p>
                <p className="text-2xl font-bold text-[#00F3FF]">{compressionResult.reduction_percent?.toFixed(1)}%</p>
              </div>
            </div>
          )}

          {compressionResult.success && (
            <div className="flex flex-col gap-3">
              {/* Download Buttons */}
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={async () => {
                    try {
                      const modelType = modelInfo?.model_type || 'unknown';
                      let fileExtension = '.pkl';
                      
                      if (modelType === 'rnn' || modelType === 'cnn' || modelType === 'lstm' || modelType === 'gru') {
                        fileExtension = '.pt';
                      }
                      
                      console.log(`📥 Downloading original model as: original_model${fileExtension}`);
                      
                      const response = await fetch('http://localhost:8000/api/model/download/original');
                      
                      if (!response.ok) {
                        throw new Error(`Download failed: ${response.statusText}`);
                      }
                      
                      const blob = await response.blob();
                      const sizeMB = (blob.size / (1024 * 1024)).toFixed(4);
                      console.log(`✅ Downloaded ${blob.size.toLocaleString()} bytes (${sizeMB} MB)`);
                      
                      const url = window.URL.createObjectURL(blob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = `original_model${fileExtension}`;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                      window.URL.revokeObjectURL(url);
                      
                      showSuccess('Download Complete', `Original model downloaded (${sizeMB} MB)`);
                    } catch (error: any) {
                      console.error('Download failed:', error);
                      showError('Download Failed', 'Could not download original model');
                    }
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#0088FF] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Original
                </button>
                <button
                  onClick={async () => {
                    try {
                      // Detect model type for correct file extension
                      const modelType = modelInfo?.model_type || 'unknown';
                      let fileExtension = '.pkl'; // Default for Decision Trees
                      
                      if (modelType === 'rnn' || modelType === 'cnn' || modelType === 'lstm' || modelType === 'gru') {
                        fileExtension = '.pt'; // PyTorch models
                      }
                      
                      console.log(`📥 Downloading compressed ${modelType} model as: compressed_model${fileExtension}`);
                      
                      const response = await fetch('http://localhost:8000/api/model/download/compressed');
                      
                      if (!response.ok) {
                        throw new Error(`Download failed: ${response.statusText}`);
                      }
                      
                      const blob = await response.blob();
                      const sizeMB = (blob.size / (1024 * 1024)).toFixed(4);
                      console.log(`✅ Downloaded ${blob.size.toLocaleString()} bytes (${sizeMB} MB)`);
                      
                      // Get filename from Content-Disposition header (backend returns correct extension)
                      const disposition = response.headers.get('content-disposition');
                      let filename = `compressed_model${fileExtension}`; // Fallback based on model type
                      if (disposition && disposition.includes('filename=')) {
                        filename = disposition.split('filename=')[1].replace(/"/g, '');
                      }
                      
                      const url = window.URL.createObjectURL(blob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = filename;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                      window.URL.revokeObjectURL(url);
                      
                      showSuccess('Download Complete', `Compressed model downloaded (${sizeMB} MB)`);
                    } catch (error: any) {
                      console.error('Download failed:', error);
                      showError('Download Failed', 'Could not download compressed model');
                    }
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-[#00FFA0] to-[#00D67F] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,255,160,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Compressed
                </button>
              </div>
              
              {/* View Results Button */}
              <button
                onClick={() => {
                  const event = new CustomEvent('navigate-to', { detail: 'results' });
                  window.dispatchEvent(event);
                }}
                className="w-full px-8 py-4 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white text-lg shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
              >
                <CheckCircle className="w-6 h-6" />
                View Results
                <ArrowRight className="w-6 h-6" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Compression;
