import { useState, useEffect } from 'react';
import { Scissors, Compass as Compress, FlaskRound as Flask, Play, CheckCircle, Clock, ArrowRight, Zap, AlertCircle } from 'lucide-react';
import { compressionService } from '../../api/services/compressionService';
import { useToast } from '../ui/ToastContainer';
import { useAppStore } from '../../store/useAppStore';

interface CompressionMethod {
  id: string;
  name: string;
  description: string;
  typicalReduction: string;
  accuracyImpact: string;
  icon: string;
  parameters: Record<string, any>;
}

interface CompressionStatus {
  method: string | null;
  status: 'idle' | 'compressing' | 'completed' | 'failed';
  progress: number;
  timeElapsed: number;
  error?: string;
}

interface ModelInfo {
  original: {
    size: number;
    parameters: number;
    architecture: string;
  };
  compressed?: {
    size: number;
    parameters: number;
    reduction: number;
  };
}

const Compression = () => {
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const [compressionStatus, setCompressionStatus] = useState<CompressionStatus>({
    method: null,
    status: 'idle',
    progress: 0,
    timeElapsed: 0
  });
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const { showSuccess, showError } = useToast();
  const selectedDatasetName = useAppStore((s) => s.selectedDatasetName);
  const selectedModelConfig = useAppStore((s) => s.selectedModel);

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      // Try to get compression info if compression was done
      const compressionInfo = await compressionService.info().catch(() => null);
      
      // Get model config to determine architecture
      const modelType = selectedModelConfig?.model_type || 'unknown';
      const architecture = modelType === 'cnn' ? 'CNN' : modelType === 'rnn' ? 'RNN' : modelType === 'decision_tree' ? 'Decision Tree' : 'Unknown';
      
      // Try to get file size from backend (we'll need to add an endpoint or calculate from compression info)
      let originalSize = 0;
      let originalParams = 0;
      
      if (compressionInfo?.result) {
        originalSize = compressionInfo.result.original_size_mb || 0;
        originalParams = compressionInfo.result.original_parameters || 0;
      }
      
      setModelInfo({
        original: {
          size: originalSize,
          parameters: originalParams,
          architecture: architecture
        },
        compressed: compressionInfo?.result ? {
          size: compressionInfo.result.compressed_size_mb || 0,
          parameters: compressionInfo.result.compressed_parameters || 0,
          reduction: compressionInfo.result.size_reduction_percent || 0
        } : undefined
      });
    } catch (error) {
      // If no compression info, just set basic info
      const modelType = selectedModelConfig?.model_type || 'unknown';
      const architecture = modelType === 'cnn' ? 'CNN' : modelType === 'rnn' ? 'RNN' : modelType === 'decision_tree' ? 'Decision Tree' : 'Unknown';
      setModelInfo({
        original: {
          size: 0,
          parameters: 0,
          architecture: architecture
        }
      });
    }
  };

  const methods: CompressionMethod[] = [
    {
      id: 'pruning',
      name: 'Weight Pruning',
      description: 'Remove low-magnitude weights',
      typicalReduction: '40-60%',
      accuracyImpact: '1-3% drop',
      icon: 'scissors',
      parameters: {
        pruningAmount: 0.5
      }
    },
    {
      id: 'quantization',
      name: 'Quantization',
      description: 'Reduce weight precision',
      typicalReduction: '60-75%',
      accuracyImpact: '0.5-2% drop',
      icon: 'compress',
      parameters: {
        bitWidth: 8
      }
    },
    {
      id: 'distillation',
      name: 'Knowledge Distillation',
      description: 'Train smaller student model',
      typicalReduction: '50-70%',
      accuracyImpact: '2-4% drop',
      icon: 'flask',
      parameters: {
        temperature: 5,
        alpha: 0.7
      }
    }
  ];

  useEffect(() => {
    if (compressionStatus.status === 'compressing') {
      const interval = setInterval(() => {
        setCompressionStatus(prev => {
          const newProgress = Math.min(prev.progress + 2, 95); // Don't go to 100, wait for backend
          const newTimeElapsed = prev.timeElapsed + 1;
          
          return {
            ...prev,
            progress: newProgress,
            timeElapsed: newTimeElapsed
          };
        });
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [compressionStatus.status]);

  useEffect(() => {
    if (compressionStatus.status === 'completed') {
      const methodName = methods.find(m => m.id === compressionStatus.method)?.name || 'Compression';
      const detail = modelInfo?.compressed?.reduction ? `-${modelInfo.compressed.reduction}% size` : undefined;
      showSuccess(`${methodName} Completed`, detail);
    }
  }, [compressionStatus.status, compressionStatus.method, modelInfo?.compressed, showSuccess]);

  const getMethodIcon = (icon: string) => {
    switch (icon) {
      case 'scissors':
        return <Scissors className="w-8 h-8" />;
      case 'compress':
        return <Compress className="w-8 h-8" />;
      case 'flask':
        return <Flask className="w-8 h-8" />;
      default:
        return <Compress className="w-8 h-8" />;
    }
  };

  const handleMethodSelect = (methodId: string) => {
    const method = methods.find(m => m.id === methodId);
    if (method) {
      setSelectedMethod(methodId);
      setParameters(method.parameters);
    }
  };

  const handleParameterChange = (key: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const startCompression = async () => {
    if (!selectedMethod) return;
    
    setCompressionStatus({
      method: selectedMethod,
      status: 'compressing',
      progress: 0,
      timeElapsed: 0
    });

    try {
      let result;
      if (selectedMethod === 'pruning') {
        result = await compressionService.compress({ method: 'pruning', pruning_amount: parameters.pruningAmount ?? 0.3 });
      } else if (selectedMethod === 'quantization') {
        result = await compressionService.compress({ method: 'quantization', quantization_bits: parameters.bitWidth ?? 8 });
      } else if (selectedMethod === 'distillation') {
        result = await compressionService.compress({ method: 'distillation', distillation_temperature: parameters.temperature ?? 3, distillation_alpha: parameters.alpha ?? 0.5 });
      }
      
      // Update compression status to completed
      setCompressionStatus(prev => ({
        ...prev,
        status: 'completed',
        progress: 100
      }));
      
      // Reload model info to get compression results
      await loadModelInfo();
      
      showSuccess('Compression Completed', 'Model successfully compressed');
    } catch (error) {
      setCompressionStatus(prev => ({
        ...prev,
        status: 'failed',
        error: 'Compression failed'
      }));
      showError('Compression Failed', (error as any)?.message ?? String(error));
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const renderParameterControls = () => {
    if (!selectedMethod) return null;

    const method = methods.find(m => m.id === selectedMethod);
    if (!method) return null;

    return (
      <div className="bg-[#0b1220]/50 rounded-lg p-4 space-y-4">
        <h4 className="text-sm font-medium text-[#E6FBFF] mb-3">Parameters</h4>
        
        {selectedMethod === 'pruning' && (
          <div className="space-y-2">
            <label className="block text-sm text-[#9BD8FF]">
              Pruning Amount: {(parameters.pruningAmount * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.1"
              value={parameters.pruningAmount}
              onChange={(e) => handleParameterChange('pruningAmount', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        )}

        {selectedMethod === 'quantization' && (
          <div className="space-y-3">
            <label className="block text-sm text-[#9BD8FF]">Bit Width</label>
            <div className="flex space-x-3">
              {[4, 8, 16].map((bits) => (
                <button
                  key={bits}
                  onClick={() => handleParameterChange('bitWidth', bits)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    parameters.bitWidth === bits
                      ? 'bg-gradient-to-r from-[#FF00D0] to-[#FF3B6B] text-white'
                      : 'bg-[#121628] border border-[#122033] text-[#9BD8FF] hover:border-[#FF00D0]/50'
                  }`}
                >
                  {bits}-bit
                </button>
              ))}
            </div>
          </div>
        )}

        {selectedMethod === 'distillation' && (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="block text-sm text-[#9BD8FF]">
                Temperature: {parameters.temperature}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={parameters.temperature}
                onChange={(e) => handleParameterChange('temperature', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm text-[#9BD8FF]">
                Alpha: {parameters.alpha}
              </label>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.1"
                value={parameters.alpha}
                onChange={(e) => handleParameterChange('alpha', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCompressionStatus = () => {
    if (compressionStatus.status === 'idle') return null;

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-8">
        <div className="text-center space-y-6">
          {compressionStatus.status === 'compressing' && (
            <>
              <div className="mx-auto w-16 h-16 relative">
                <div className="absolute inset-0 border-4 border-[#00F3FF]/20 rounded-full"></div>
                <div 
                  className="absolute inset-0 border-4 border-[#00F3FF] rounded-full border-t-transparent animate-spin"
                ></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Zap className="w-6 h-6 text-[#00F3FF]" />
                </div>
              </div>
              
              <div>
                <h3 className="text-2xl font-bold text-[#E6FBFF] mb-2">Compressing Model...</h3>
                <p className="text-[#9BD8FF]">Using {methods.find(m => m.id === compressionStatus.method)?.name}</p>
              </div>

              <div className="max-w-md mx-auto space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-[#9BD8FF]">Progress</span>
                  <span className="text-[#E6FBFF] font-semibold">{compressionStatus.progress}%</span>
                </div>
                <div className="bg-[#0b1220] rounded-full h-3 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] h-full transition-all duration-500"
                    style={{ width: `${compressionStatus.progress}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-[#9BD8FF]">
                  <span>Time elapsed: {formatTime(compressionStatus.timeElapsed)}</span>
                  <span>Est. remaining: {formatTime(Math.max(0, 60 - compressionStatus.timeElapsed))}</span>
                </div>
              </div>
            </>
          )}

          {compressionStatus.status === 'completed' && (
            <>
              <div className="mx-auto w-16 h-16 text-[#00FFA0] relative">
                <CheckCircle className="w-full h-full" />
                <div className="absolute inset-0 bg-[#00FFA0] blur-xl opacity-30 rounded-full animate-pulse"></div>
              </div>
              
              <div>
                <h3 className="text-2xl font-bold text-[#00FFA0] mb-2">Compression Completed!</h3>
                <p className="text-[#9BD8FF]">Model successfully compressed in {formatTime(compressionStatus.timeElapsed)}</p>
              </div>
            </>
          )}

          {compressionStatus.status === 'failed' && (
            <>
              <div className="mx-auto w-16 h-16 text-[#FF3B6B]">
                <AlertCircle className="w-full h-full" />
              </div>
              
              <div>
                <h3 className="text-2xl font-bold text-[#FF3B6B] mb-2">Compression Failed</h3>
                <p className="text-[#9BD8FF]">{compressionStatus.error || 'An error occurred during compression'}</p>
              </div>

              <button
                onClick={() => setCompressionStatus({ method: null, status: 'idle', progress: 0, timeElapsed: 0 })}
                className="px-6 py-3 bg-gradient-to-r from-[#FF3B6B] to-[#FF0040] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_20px_rgba(255,59,107,0.3)] transition-all duration-300 hover:scale-105"
              >
                Try Again
              </button>
            </>
          )}
        </div>
      </div>
    );
  };

  const renderBeforeAfterComparison = () => {
    if (!modelInfo || !modelInfo.compressed) return null;

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-8">
        <h2 className="text-2xl font-semibold text-[#E6FBFF] mb-6 text-center">Before vs After</h2>
        
        <div className="grid md:grid-cols-2 gap-8">
          {/* Original Model */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-[#9BD8FF] text-center">Original Model</h3>
            <div className="bg-[#0b1220]/50 rounded-lg p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-[#9BD8FF]">File Size</span>
                <span className="text-[#E6FBFF] font-semibold">{modelInfo.original.size} MB</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#9BD8FF]">Parameters</span>
                <span className="text-[#E6FBFF] font-semibold">{modelInfo.original.parameters.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#9BD8FF]">Architecture</span>
                <span className="text-[#E6FBFF] font-semibold">{modelInfo.original.architecture}</span>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div className="hidden md:flex items-center justify-center">
            <ArrowRight className="w-8 h-8 text-[#00F3FF]" />
          </div>

          {/* Compressed Model */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-[#00FFA0] text-center">Compressed Model</h3>
            <div className="bg-[#0b1220]/50 rounded-lg p-6 space-y-4 border border-[#00FFA0]/20">
              <div className="flex items-center justify-between">
                <span className="text-[#9BD8FF]">File Size</span>
                <div className="flex items-center gap-2">
                  <span className="text-[#00FFA0] font-semibold">{modelInfo.compressed.size.toFixed(1)} MB</span>
                  <span className="px-2 py-1 bg-[#00FFA0]/20 text-[#00FFA0] text-xs rounded-full font-medium">
                    -{modelInfo.compressed.reduction}%
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#9BD8FF]">Parameters</span>
                <span className="text-[#00FFA0] font-semibold">{modelInfo.compressed.parameters.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#9BD8FF]">Architecture</span>
                <span className="text-[#00FFA0] font-semibold">{modelInfo.original.architecture} (Compressed)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Size Visualization */}
        <div className="mt-8 space-y-4">
          <h4 className="text-center text-[#E6FBFF] font-semibold">Size Comparison</h4>
          <div className="flex items-end justify-center gap-8">
            <div className="text-center">
              <div 
                className="bg-[#9BD8FF]/20 border border-[#9BD8FF] rounded-lg mb-2"
                style={{ width: '120px', height: '80px' }}
              ></div>
              <span className="text-sm text-[#9BD8FF]">Original</span>
              <div className="text-xs text-[#E6FBFF]">{modelInfo.original.size} MB</div>
            </div>
            <ArrowRight className="w-6 h-6 text-[#00F3FF] mb-8" />
            <div className="text-center">
              <div 
                className="bg-[#00FFA0]/20 border border-[#00FFA0] rounded-lg mb-2"
                style={{ width: '42px', height: '28px' }}
              ></div>
              <span className="text-sm text-[#00FFA0]">Compressed</span>
              <div className="text-xs text-[#E6FBFF]">{modelInfo.compressed.size.toFixed(1)} MB</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Model Compression
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Reduce model size while maintaining accuracy
        </p>
        {selectedDatasetName && (
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md" title={`Dataset: ${selectedDatasetName} | Model: ${modelInfo?.original.architecture || 'Unknown'}`}>
              Dataset: {selectedDatasetName} | Model: {modelInfo?.original.architecture || 'Unknown'}
            </span>
          </div>
        )}
      </div>

      {/* Method Selection */}
      <div className="grid md:grid-cols-3 gap-8">
        {methods.map((method) => (
          <div
            key={method.id}
            className={`relative bg-[#121628]/50 border rounded-xl p-6 cursor-pointer transition-all duration-300 group hover:scale-105 ${
              selectedMethod === method.id
                ? 'border-[#00F3FF] shadow-[0_0_20px_rgba(0,243,255,0.3)]'
                : 'border-[#122033] hover:border-[#00F3FF]/50 hover:shadow-[0_0_30px_rgba(0,243,255,0.2)]'
            }`}
            onClick={() => handleMethodSelect(method.id)}
          >
            {/* Background Glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-[rgba(0,243,255,0.02)] to-[rgba(255,0,208,0.02)] rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
            
            <div className="relative z-10 space-y-4">
              {/* Icon */}
              <div className={`mb-4 relative ${
                method.id === 'pruning' ? 'text-[#00F3FF]' :
                method.id === 'quantization' ? 'text-[#FF00D0]' :
                'text-[#00FFA0]'
              }`}>
                {getMethodIcon(method.icon)}
                <div className={`absolute inset-0 blur-lg opacity-20 rounded-full ${
                  method.id === 'pruning' ? 'bg-[#00F3FF]' :
                  method.id === 'quantization' ? 'bg-[#FF00D0]' :
                  'bg-[#00FFA0]'
                }`}></div>
              </div>

              {/* Method Info */}
              <div>
                <h3 className="text-xl font-bold text-[#E6FBFF] mb-2">{method.name}</h3>
                <p className="text-[#9BD8FF] text-sm mb-4">{method.description}</p>
                
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-[#9BD8FF]">Typical Reduction:</span>
                    <span className="text-[#00FFA0] font-medium">{method.typicalReduction}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#9BD8FF]">Accuracy Impact:</span>
                    <span className="text-[#FFB84D] font-medium">{method.accuracyImpact}</span>
                  </div>
                </div>
              </div>

              {/* Parameters Preview */}
              {selectedMethod === method.id && renderParameterControls()}

              {/* Apply Button */}
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  startCompression();
                }}
                disabled={compressionStatus.status === 'compressing'}
                className={`w-full py-3 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
                  selectedMethod === method.id
                    ? `bg-gradient-to-r ${
                        method.id === 'pruning' ? 'from-[#00F3FF] to-[#00B3CC]' :
                        method.id === 'quantization' ? 'from-[#FF00D0] to-[#CC0099]' :
                        'from-[#00FFA0] to-[#00CC80]'
                      } text-white shadow-lg hover:scale-105`
                    : 'bg-[#0b1220] border border-[#122033] text-[#9BD8FF] hover:border-[#00F3FF]/50'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {compressionStatus.status === 'compressing' && compressionStatus.method === method.id ? (
                  <>
                    <Clock className="w-4 h-4 animate-spin" />
                    Compressing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Apply {method.name}
                  </>
                )}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Compression Status */}
      {renderCompressionStatus()}

      {/* Before/After Comparison */}
      {renderBeforeAfterComparison()}
    </div>
  );
};

export default Compression;