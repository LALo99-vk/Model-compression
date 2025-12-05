import { useState, useEffect } from 'react';
import { 
  Play, 
  Square, 
  Activity, 
  TrendingUp, 
  Clock, 
  CheckCircle,
  Terminal,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { useTraining } from '../../hooks/useTraining';
import { useAppStore } from '../../store/useAppStore';
import { useToast } from '../ui/ToastContainer';
import { TrainingLogsResponse } from '../../api/services/trainingService';

const Training = () => {
  const { trainingStatus, trainingLogs, start, stop, poll } = useTraining();
  const datasets = useAppStore((s) => s.datasets);
  const selectedDatasetPath = useAppStore((s) => s.selectedDatasetPath);
  const selectedDatasetName = useAppStore((s) => s.selectedDatasetName);
  const setSelectedDataset = useAppStore((s) => s.setSelectedDataset);
  const selectedModelConfig = useAppStore((s) => s.selectedModel);
  const { showError, showSuccess } = useToast();
  
  const [config, setConfig] = useState({
    epochs: 20,
    batchSize: 32,
    validationSplit: 0.2,
    datasetPath: selectedDatasetPath || (datasets[0]?.path || '')
  });
  
  const [lastTrainingParams, setLastTrainingParams] = useState<{
    datasetPath: string;
    epochs: number;
    batchSize: number;
    validationSplit: number;
  } | null>(null);
  
  // Define isTraining BEFORE using it in useEffect
  const isTraining = trainingStatus?.status === 'training' || 
                     trainingStatus?.status === 'loading_data' || 
                     trainingStatus?.status === 'preprocessing' || 
                     trainingStatus?.status === 'validating' ||
                     trainingStatus?.status === 'normalizing' ||
                     trainingStatus?.status === 'normalizing_dataset';
  
  // Check status on mount to catch any completed training
  useEffect(() => {
    poll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount to check if training completed while page wasn't open
  
  // Continuous polling while training is active
  useEffect(() => {
    if (!isTraining) return;
    
    const pollInterval = setInterval(() => {
      poll();
    }, 1000); // Poll every 1 second for faster updates
    
    return () => clearInterval(pollInterval);
  }, [isTraining, poll]);

  useEffect(() => {
    if (selectedDatasetPath && config.datasetPath !== selectedDatasetPath) {
      setConfig((prev) => ({ ...prev, datasetPath: selectedDatasetPath }));
    }
  }, [selectedDatasetPath, config.datasetPath]);

  // Listen for auto-start training event from Dataset Validation page
  useEffect(() => {
    const handleAutoStartTraining = (event: Event) => {
      const customEvent = event as CustomEvent<{
        dataset_path: string;
        epochs?: number;
        batch_size?: number;
        validation_split?: number;
      }>;
      
      const { dataset_path, epochs = 20, batch_size = 32, validation_split = 0.2 } = customEvent.detail;
      
      // Update config with the dataset from validation
      setConfig((prev) => ({
        ...prev,
        datasetPath: dataset_path,
        epochs: epochs,
        batchSize: batch_size,
        validationSplit: validation_split
      }));
      
      // Reset last training params to allow auto-start even if same params
      setLastTrainingParams(null);
      
      // Small delay to ensure config is updated, then start training
      setTimeout(() => {
        start({ 
          dataset_path: dataset_path, 
          epochs: epochs, 
          batch_size: batch_size, 
          validation_split: validation_split 
        });
      }, 200);
    };
    
    window.addEventListener('auto-start-training', handleAutoStartTraining);
    
    return () => {
      window.removeEventListener('auto-start-training', handleAutoStartTraining);
    };
  }, [start]);

  const [showLogs, setShowLogs] = useState(false);
  const trainingLogsTyped = trainingLogs as TrainingLogsResponse | null;
  const history = trainingLogsTyped?.history;

  // Auto-show logs when training starts or when logs are available
  useEffect(() => {
    if (isTraining || (trainingStatus?.status === 'completed' && history && history.length > 0)) {
      setShowLogs(true);
    }
  }, [isTraining, trainingStatus?.status, history]);

  // Model verification - NO auto-navigation
  useEffect(() => {
    if (trainingStatus?.status === 'completed' && trainingLogsTyped) {
      // Verify trained model
      const modelValid = verifyTrainedModel(trainingLogsTyped);
      
      if (modelValid) {
        showSuccess('Training Complete', 'Model verified successfully! Click "Compress Model" to continue.');
        // REMOVED: Auto-navigation - user will click button manually
      } else {
        showError('Training Invalid', 'Model verification failed. Please check the logs and retrain.');
      }
    } else if (trainingStatus?.status === 'stopped') {
      showSuccess('Training Stopped', trainingStatus.message || 'Training was stopped by user');
    } else if (trainingStatus?.status === 'error' || trainingStatus?.status === 'failed') {
      showError('Training Failed', trainingStatus.message || 'Training encountered an error');
    }
  }, [trainingStatus?.status, trainingLogsTyped]);

  // Model verification function
  const verifyTrainedModel = (logs: TrainingLogsResponse): boolean => {
    // HARD RULE: Verify model after training
    // 1. File exists (checked by backend)
    // 2. File size > 0
    // 3. Parameters > 0
    // 4. Model loads correctly (checked by backend)
    
    const modelSizeMB = logs.model_size_mb || 0;
    const totalParams = logs.total_parameters || logs.num_parameters || logs.trainable_parameters || 0;
    
    if (modelSizeMB <= 0) {
      console.error('❌ Model verification failed: File size is 0 MB');
      return false;
    }
    
    if (totalParams <= 0) {
      console.error('❌ Model verification failed: Parameter count is 0');
      return false;
    }
    
    console.log('✅ Model verification passed:', {
      size_mb: modelSizeMB,
      parameters: totalParams,
      model_type: selectedModelConfig?.model_type
    });
    
    return true;
  };

  const computedProgress = trainingStatus ? Math.round((trainingStatus.current_epoch / trainingStatus.total_epochs) * 100) : 0;

  const modelLabel = selectedModelConfig?.model_type
    ? (selectedModelConfig.model_type === 'cnn'
        ? 'CNN Model'
        : selectedModelConfig.model_type === 'rnn'
        ? 'RNN Model'
        : selectedModelConfig.model_type === 'decision_tree'
        ? 'Decision Tree'
        : String(selectedModelConfig.model_type))
    : 'No Model Selected';

  const trainingTimeSeconds = trainingLogsTyped?.training_time;
  const formattedTrainingTime = typeof trainingTimeSeconds === 'number'
    ? `${Math.floor(trainingTimeSeconds / 60)}m ${Math.floor(trainingTimeSeconds % 60)}s`
    : 'N/A';

  const startTraining = () => {
    // Check if parameters are the same as last training
    if (lastTrainingParams) {
      const paramsMatch = 
        lastTrainingParams.datasetPath === config.datasetPath &&
        lastTrainingParams.epochs === config.epochs &&
        lastTrainingParams.batchSize === config.batchSize &&
        lastTrainingParams.validationSplit === config.validationSplit;
      
      if (paramsMatch) {
        showError('Duplicate Training', 'Training parameters are the same as the last training. Please change at least one parameter (epochs, batch size, validation split, or dataset) before training again.');
        return;
      }
    }
    
    // Save current parameters as last training params
    setLastTrainingParams({
      datasetPath: config.datasetPath,
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationSplit: config.validationSplit
    });
    
    start({ dataset_path: config.datasetPath, epochs: config.epochs, batch_size: config.batchSize, validation_split: config.validationSplit });
  };

  const stopTraining = () => {
    stop();
  };

  const renderTrainingChart = () => {
    if (!history || history.length === 0) return null;

    const maxLoss = Math.max(...history.map(d => Math.max(d.train_loss, d.val_loss)));
    const minLoss = Math.min(...history.map(d => Math.min(d.train_loss, d.val_loss)));

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h3 className="text-xl font-semibold text-[#E6FBFF] mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-[#00F3FF]" />
          Training Progress
        </h3>
        
        <div className="relative h-64 bg-[#0b1220] rounded-lg p-4">
          <svg width="100%" height="100%" viewBox="0 0 400 200" className="overflow-visible">
            {/* Grid lines */}
            <defs>
              <pattern id="grid" width="40" height="20" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#122033" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            
            {/* Training Loss Line */}
            <polyline
              fill="none"
              stroke="#00F3FF"
              strokeWidth="2"
              points={history.map((d, i) => 
                `${(i / (history.length - 1)) * 380 + 10},${190 - ((d.train_loss - minLoss) / (maxLoss - minLoss)) * 170}`
              ).join(' ')}
            />
            
            {/* Validation Loss Line */}
            <polyline
              fill="none"
              stroke="#FF00D0"
              strokeWidth="2"
              points={history.map((d, i) => 
                `${(i / (history.length - 1)) * 380 + 10},${190 - ((d.val_loss - minLoss) / (maxLoss - minLoss)) * 170}`
              ).join(' ')}
            />

            {/* Accuracy Line (scaled) */}
            <polyline
              fill="none"
              stroke="#00FFA0"
              strokeWidth="2"
              points={history.map((d, i) => 
                `${(i / (history.length - 1)) * 380 + 10},${190 - d.val_accuracy * 170}`
              ).join(' ')}
            />
          </svg>
          
          {/* Legend */}
          <div className="absolute top-4 right-4 space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-0.5 bg-[#00F3FF]"></div>
              <span className="text-[#00F3FF]">Training Loss</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-0.5 bg-[#FF00D0]"></div>
              <span className="text-[#FF00D0]">Validation Loss</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-0.5 bg-[#00FFA0]"></div>
              <span className="text-[#00FFA0]">Accuracy</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Model Training
          </h1>
          <div className="flex items-center gap-4 mt-2">
            {selectedDatasetName && (
              <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-sm text-[#E6FBFF] font-medium" title={`Dataset: ${selectedDatasetName} | Model: ${modelLabel}`}>
                Dataset: {selectedDatasetName} | Model: {modelLabel}
              </span>
            )}
            {!selectedDatasetName && (
              <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-sm text-[#00F3FF] font-medium">
                Model: {modelLabel}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h2 className="text-xl font-semibold text-[#E6FBFF] mb-4">Training Configuration</h2>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[#9BD8FF]">Epochs</label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => setConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
              className="w-full px-3 py-2 bg-[#0b1220] border border-[#122033] rounded-lg text-[#E6FBFF] focus:border-[#00F3FF] focus:outline-none"
              disabled={trainingStatus?.status === 'training'}
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[#9BD8FF]">Dataset</label>
            <select
              value={config.datasetPath}
              onChange={(e) => {
                const value = e.target.value;
                setConfig(prev => ({ ...prev, datasetPath: value }));
                const ds = datasets.find(d => d.path === value);
                if (ds) {
                  setSelectedDataset({ filename: ds.filename, path: ds.path });
                }
              }}
              className="w-full px-3 py-2 bg-[#0b1220] border border-[#122033] rounded-lg text-[#E6FBFF] focus:border-[#00F3FF] focus:outline-none"
              disabled={isTraining}
            >
              {datasets.length === 0 && (
                <option value={config.datasetPath}>Select a dataset from Upload page</option>
              )}
              {datasets.map((d) => (
                <option key={d.filename} value={d.path}>{d.filename}</option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[#9BD8FF]">Batch Size</label>
            <select
              value={config.batchSize}
              onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
              className="w-full px-3 py-2 bg-[#0b1220] border border-[#122033] rounded-lg text-[#E6FBFF] focus:border-[#00F3FF] focus:outline-none"
              disabled={isTraining}
            >
              <option value={8}>8</option>
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={64}>64</option>
              <option value={128}>128</option>
            </select>
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[#9BD8FF]">Validation Split</label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.validationSplit}
              onChange={(e) => setConfig(prev => ({ ...prev, validationSplit: parseFloat(e.target.value) }))}
              className="w-full"
              disabled={isTraining}
            />
            <div className="text-xs text-[#9BD8FF] text-center">{config.validationSplit}</div>
          </div>
          <div className="flex items-end">
            {!isTraining ? (
              <button
                onClick={startTraining}
                className="w-full py-2 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_20px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                Start Training
              </button>
            ) : (
              <button
                onClick={stopTraining}
                className="w-full py-2 bg-gradient-to-r from-[#FF3B6B] to-[#FF0040] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_20px_rgba(255,59,107,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
              >
                <Square className="w-4 h-4" />
                Stop Training
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Training Status Card */}
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-8">
        {(!trainingStatus || trainingStatus.status === 'not_started' || trainingStatus.status === 'idle') && (
          <div className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 text-[#9BD8FF]/50">
              <Play className="w-full h-full" />
            </div>
            <h3 className="text-xl font-semibold text-[#E6FBFF]">Ready to Train</h3>
            <p className="text-[#9BD8FF]">Configure your parameters and start training</p>
          </div>
        )}

        {/* Normalizing Status */}
        {(trainingStatus && (trainingStatus.status === 'normalizing' || trainingStatus.status === 'normalizing_dataset')) && (
          <div className="space-y-6">
            <div className="flex justify-center mb-6">
              <div className="relative w-32 h-32">
                {[...Array(3)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute inset-0 border-2 border-[#00F3FF] rounded-full animate-ping opacity-20"
                    style={{
                      animationDelay: `${i * 0.5}s`,
                      animationDuration: '2s'
                    }}
                  ></div>
                ))}
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-[#00F3FF]" />
                </div>
              </div>
            </div>
            <div className="text-center space-y-4">
              <h3 className="text-xl font-semibold text-[#E6FBFF]">Normalizing Dataset</h3>
              <p className="text-[#9BD8FF]">{trainingStatus.message || "Processing and standardizing dataset..."}</p>
              <div className="flex items-center justify-center gap-2 text-sm text-[#9BD8FF]">
                <div className="w-2 h-2 bg-[#00F3FF] rounded-full animate-pulse"></div>
                <span>Auto-detecting schema, mapping columns, preprocessing...</span>
              </div>
            </div>
          </div>
        )}

        {/* Validating Status */}
        {trainingStatus && trainingStatus.status === 'validating' && (
          <div className="space-y-6">
            <div className="text-center space-y-4">
              <div className="mx-auto w-16 h-16 text-[#00F3FF]">
                <Activity className="w-full h-full animate-pulse" />
              </div>
              <h3 className="text-xl font-semibold text-[#E6FBFF]">Validating Dataset</h3>
              <p className="text-[#9BD8FF]">{trainingStatus.message || "Checking dataset format and structure..."}</p>
            </div>
          </div>
        )}

        {/* Preprocessing Status */}
        {trainingStatus && trainingStatus.status === 'preprocessing' && (
          <div className="space-y-6">
            <div className="text-center space-y-4">
              <div className="mx-auto w-16 h-16 text-[#00F3FF]">
                <Activity className="w-full h-full animate-pulse" />
              </div>
              <h3 className="text-xl font-semibold text-[#E6FBFF]">Preprocessing Data</h3>
              <p className="text-[#9BD8FF]">{trainingStatus.message || "Applying model-specific preprocessing..."}</p>
            </div>
          </div>
        )}

        {/* Loading Data Status */}
        {trainingStatus && trainingStatus.status === 'loading_data' && (
          <div className="space-y-6">
            <div className="text-center space-y-4">
              <div className="mx-auto w-16 h-16 text-[#00F3FF]">
                <Activity className="w-full h-full animate-pulse" />
              </div>
              <h3 className="text-xl font-semibold text-[#E6FBFF]">Loading Data</h3>
              <p className="text-[#9BD8FF]">{trainingStatus.message || "Preparing data for training..."}</p>
            </div>
          </div>
        )}

        {trainingStatus && trainingStatus.status === 'training' && (
          <div className="space-y-6">
            {/* Neural Network Animation */}
            <div className="flex justify-center mb-6">
              <div className="relative w-32 h-32">
                {[...Array(3)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute inset-0 border-2 border-[#00F3FF] rounded-full animate-ping opacity-20"
                    style={{
                      animationDelay: `${i * 0.5}s`,
                      animationDuration: '2s'
                    }}
                  ></div>
                ))}
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-[#00F3FF]" />
                </div>
              </div>
            </div>

            {/* Progress */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-xl font-semibold text-[#E6FBFF]">Training in Progress</h3>
                <div className="flex items-center gap-4 text-sm text-[#9BD8FF]">
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                  <span>{trainingStatus.message}</span>
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-[#9BD8FF]">Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs}</span>
                  <span className="text-[#E6FBFF] font-semibold">{computedProgress}%</span>
                </div>
                <div className="bg-[#0b1220] rounded-full h-3 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] h-full transition-all duration-500 relative"
                    style={{ width: `${computedProgress}%` }}
                  >
                    <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                  </div>
                </div>
              </div>

              {/* Live Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <div className="bg-[#0b1220] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#00F3FF] rounded-full"></div>
                    <span className="text-sm text-[#9BD8FF]">Training Loss</span>
                  </div>
                  <div className="text-xl font-bold text-[#E6FBFF]">
                    {history && history.length > 0 ? history[history.length - 1].train_loss.toFixed(4) : (trainingLogsTyped?.train_score ?? 0).toFixed(4)}
                  </div>
                </div>
                <div className="bg-[#0b1220] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#FF00D0] rounded-full"></div>
                    <span className="text-sm text-[#9BD8FF]">Val Loss</span>
                  </div>
                  <div className="text-xl font-bold text-[#E6FBFF]">
                    {history && history.length > 0 ? history[history.length - 1].val_loss.toFixed(4) : (trainingLogsTyped?.val_score ?? 0).toFixed(4)}
                  </div>
                </div>
                <div className="bg-[#0b1220] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#00FFA0] rounded-full"></div>
                    <span className="text-sm text-[#9BD8FF]">Val Accuracy</span>
                  </div>
                  <div className="text-xl font-bold text-[#E6FBFF]">
                    {history && history.length > 0 ? (history[history.length - 1].val_accuracy * 100).toFixed(2) : (trainingLogsTyped?.val_score ? (trainingLogsTyped.val_score * 100).toFixed(2) : 'N/A')}%
                  </div>
                </div>
                {(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb) && (
                  <div className="bg-[#0b1220] rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-2 h-2 bg-[#FFD700] rounded-full"></div>
                      <span className="text-sm text-[#9BD8FF]">Model Size</span>
                    </div>
                    <div className="text-xl font-bold text-[#E6FBFF]">
                      {(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb ?? 0).toFixed(3)} MB
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {trainingStatus && trainingStatus.status === 'completed' && (
          <div className="text-center space-y-6">
            <div className="mx-auto w-16 h-16 text-[#00FFA0] relative">
              <CheckCircle className="w-full h-full" />
              <div className="absolute inset-0 bg-[#00FFA0] blur-xl opacity-30 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h3 className="text-2xl font-bold text-[#00FFA0] mb-2">Training Completed Successfully!</h3>
              <p className="text-[#9BD8FF]">Model training finished. All {trainingStatus.total_epochs} epochs completed.</p>
              {trainingStatus.message && (
                <p className="text-[#9BD8FF] text-sm mt-2">{trainingStatus.message}</p>
              )}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto text-sm">
              <div>
                <span className="text-[#9BD8FF] block mb-1">Final Accuracy</span>
                <div className="text-xl font-bold text-[#E6FBFF]">
                  {history && history.length > 0 ? (history[history.length - 1].val_accuracy * 100).toFixed(2) : (trainingLogsTyped?.val_score ? (trainingLogsTyped.val_score * 100).toFixed(2) : 'N/A')}%
                </div>
              </div>
              <div>
                <span className="text-[#9BD8FF] block mb-1">Training Time</span>
                <div className="text-xl font-bold text-[#E6FBFF]">
                  {formattedTrainingTime}
                </div>
              </div>
              <div>
                <span className="text-[#9BD8FF] block mb-1">Model Size</span>
                <div className="text-xl font-bold text-[#E6FBFF]">
                  {(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb) 
                    ? `${(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb ?? 0).toFixed(3)} MB`
                    : (trainingStatus?.model_size_kb ?? trainingLogsTyped?.model_size_kb)
                    ? `${(trainingStatus?.model_size_kb ?? trainingLogsTyped?.model_size_kb ?? 0).toFixed(2)} KB`
                    : 'N/A'}
                </div>
              </div>
              <div>
                <span className="text-[#9BD8FF] block mb-1">Total Epochs</span>
                <div className="text-xl font-bold text-[#E6FBFF]">
                  {trainingStatus.total_epochs}
                </div>
              </div>
            </div>
            {/* Model Parameters Count */}
            {(trainingStatus?.total_parameters ?? trainingStatus?.num_parameters ?? trainingLogsTyped?.total_parameters ?? trainingLogsTyped?.num_parameters) && (
              <div className="bg-[#0b1220] rounded-lg p-4 max-w-md mx-auto">
                <div className="text-center">
                  <span className="text-[#9BD8FF] block mb-1">Model Parameters</span>
                  <div className="text-lg font-bold text-[#E6FBFF]">
                    {(trainingStatus?.total_parameters ?? trainingStatus?.num_parameters ?? trainingLogsTyped?.total_parameters ?? trainingLogsTyped?.num_parameters)?.toLocaleString()}
                    {(trainingStatus?.trainable_parameters ?? trainingLogsTyped?.trainable_parameters) && (
                      <span className="text-sm text-[#9BD8FF] ml-2">
                        ({(trainingStatus?.trainable_parameters ?? trainingLogsTyped?.trainable_parameters ?? 0).toLocaleString()} trainable)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
            {history && history.length > 0 && (
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto text-sm mt-4">
                <div>
                  <span className="text-[#9BD8FF] block mb-1">Final Train Loss</span>
                  <div className="text-lg font-semibold text-[#E6FBFF]">
                    {history[history.length - 1].train_loss.toFixed(4)}
                  </div>
                </div>
                <div>
                  <span className="text-[#9BD8FF] block mb-1">Final Val Loss</span>
                  <div className="text-lg font-semibold text-[#E6FBFF]">
                    {history[history.length - 1].val_loss.toFixed(4)}
                  </div>
                </div>
              </div>
            )}
            <div className="flex gap-4">
              <button 
                onClick={async () => {
                  try {
                    // Detect correct file extension based on model type
                    const modelType = trainingLogsTyped?.model_type || 'unknown';
                    let fileExtension = '.pkl'; // Default for Decision Trees
                    
                    if (modelType === 'rnn' || modelType === 'cnn' || modelType === 'lstm' || modelType === 'gru') {
                      fileExtension = '.pt'; // PyTorch models
                    }
                    
                    console.log(`📥 Downloading ${modelType} model as: original_model${fileExtension}`);
                    
                    // Fetch the file as a blob (ensures full download)
                    // Use full URL to bypass any Vite proxy issues
                    const response = await fetch('http://localhost:8000/api/model/download/original');
                    
                    if (!response.ok) {
                      throw new Error(`Download failed: ${response.statusText}`);
                    }
                    
                    // Get the blob and size
                    const blob = await response.blob();
                    const sizeMB = (blob.size / (1024 * 1024)).toFixed(4);
                    console.log(`✅ Downloaded ${blob.size.toLocaleString()} bytes (${sizeMB} MB)`);
                    
                    // Create download link from blob
                    const url = window.URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `original_model${fileExtension}`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    window.URL.revokeObjectURL(url);
                    
                    showSuccess(`Model downloaded successfully (${sizeMB} MB)`);
                  } catch (error) {
                    console.error('Download failed:', error);
                    showError('Download failed. Please try again.');
                  }
                }}
                className="px-8 py-3 bg-gradient-to-r from-[#00F3FF] to-[#0088FF] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Original Model
              </button>
              <button 
                onClick={() => {
                  const event = new CustomEvent('navigate-to', { detail: 'compression' });
                  window.dispatchEvent(event);
                }}
                className="px-8 py-3 bg-gradient-to-r from-[#00FFA0] to-[#00D67F] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,255,160,0.3)] transition-all duration-300 hover:scale-105 flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Compress Model
              </button>
            </div>
          </div>
        )}

        {trainingStatus && trainingStatus.status === 'stopped' && (
          <div className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 text-[#FFB84D] relative">
              <Square className="w-full h-full" />
              <div className="absolute inset-0 bg-[#FFB84D] blur-xl opacity-30 rounded-full"></div>
            </div>
            <div>
              <h3 className="text-2xl font-bold text-[#FFB84D] mb-2">Training Stopped</h3>
              <p className="text-[#9BD8FF]">{trainingStatus.message || 'Training was stopped by user'}</p>
              {trainingLogsTyped && trainingLogsTyped.model_size_mb && (
                <p className="text-[#00F3FF] mt-2">Best model saved: {trainingLogsTyped.model_size_mb.toFixed(3)} MB</p>
              )}
            </div>
          </div>
        )}

        {trainingStatus && trainingStatus.status === 'error' && (
          <div className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 text-red-500 relative">
              <Activity className="w-full h-full" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-red-500 mb-2">Training Error</h3>
              <p className="text-[#9BD8FF]">{trainingStatus.message || 'An error occurred during training'}</p>
            </div>
          </div>
        )}
      </div>

      {/* Live Backend Process Logs */}
      {(isTraining || trainingStatus?.status === 'completed' || trainingStatus?.status === 'stopped') && (
        <div className="bg-[#121628]/50 border border-[#00FFA0]/30 rounded-xl p-6 mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Terminal className="w-5 h-5 text-[#00FFA0]" />
            <h3 className="text-xl font-semibold text-[#E6FBFF]">
              🔄 Live Process Logs
            </h3>
            {isTraining && (
              <span className="px-2 py-1 bg-[#00FFA0]/10 border border-[#00FFA0]/30 rounded text-xs text-[#00FFA0] animate-pulse">
                LIVE
              </span>
            )}
          </div>

          <div className="bg-[#0b1220] rounded-lg p-4 font-mono text-xs max-h-[400px] overflow-y-auto space-y-1">
            {/* Status Message */}
            {trainingStatus?.message && (
              <div className="text-[#00FFA0] font-semibold">
                <span className="text-[#9BD8FF]">[{new Date(trainingStatus.timestamp * 1000).toLocaleTimeString()}]</span> {trainingStatus.message}
              </div>
            )}
            
            {/* Real Backend Logs */}
            {trainingLogsTyped && trainingStatus && trainingStatus.status !== 'not_started' && (
              <div className="space-y-1 mt-2">
                {/* Model & Task Info */}
                {trainingLogsTyped.model_type && (
                  <div className="text-[#00F3FF]">
                    <span className="text-[#9BD8FF]">[INFO]</span> 🏗️ Model: {trainingLogsTyped.model_type.toUpperCase()}
                    {trainingLogsTyped.task_type && ` | Task: ${trainingLogsTyped.task_type.toUpperCase()}`}
                    {trainingLogsTyped.num_classes && ` | Classes: ${trainingLogsTyped.num_classes}`}
                  </div>
                )}
                
                {/* Optimizations Info */}
                {trainingLogsTyped.optimizations && (
                  <div className="text-[#FFB84D]">
                    <span className="text-[#9BD8FF]">[INFO]</span> ⚡ Optimizations: Early Stopping ✅, LR Scheduler ✅, Mixed Precision ✅
                  </div>
                )}
                
                {/* Training Progress */}
                {history && history.length > 0 && (
                  <>
                    <div className="text-[#00FFA0]">
                      <span className="text-[#9BD8FF]">[PROGRESS]</span> 📈 Epoch {history.length}/{trainingStatus.total_epochs}
                    </div>
                    <div className="text-[#00F3FF]">
                      <span className="text-[#9BD8FF]">[METRICS]</span> Loss: {history[history.length - 1].train_loss.toFixed(4)} | 
                      Val Loss: {history[history.length - 1].val_loss.toFixed(4)} | 
                      Val Acc: {(history[history.length - 1].val_accuracy * 100).toFixed(2)}%
                    </div>
                    {history[history.length - 1].current_lr && (
                      <div className="text-[#9BD8FF]/80">
                        <span className="text-[#9BD8FF]">[INFO]</span> Learning Rate: {history[history.length - 1].current_lr.toFixed(6)}
                      </div>
                    )}
                  </>
                )}
                
                {/* Stopped Status */}
                {trainingStatus.status === 'stopped' && (
                  <div className="text-[#FFB84D] mt-2 font-semibold border-t border-[#FFB84D]/20 pt-2">
                    <span className="text-[#9BD8FF]">[STOPPED]</span> ⏸️ {trainingStatus.message || 'Training stopped by user'}
                  </div>
                )}
                
                {/* Completion Status */}
                {trainingStatus.status === 'completed' && (
                  <>
                    <div className="text-[#00FFA0] mt-2 font-semibold border-t border-[#00FFA0]/20 pt-2">
                      <span className="text-[#9BD8FF]">[SUCCESS]</span> ✅ Training completed successfully!
                    </div>
                    {trainingLogsTyped.training_time && (
                      <div className="text-[#00F3FF]">
                        <span className="text-[#9BD8FF]">[TIME]</span> ⏱️ Total: {(trainingLogsTyped.training_time / 60).toFixed(2)} minutes
                      </div>
                    )}
                    {trainingLogsTyped.model_size_mb && (
                      <div className="text-[#00F3FF]">
                        <span className="text-[#9BD8FF]">[MODEL]</span> 💾 Size: {trainingLogsTyped.model_size_mb.toFixed(4)} MB
                        {trainingLogsTyped.total_parameters && ` | Params: ${trainingLogsTyped.total_parameters.toLocaleString()}`}
                      </div>
                    )}
                    {trainingLogsTyped.best_val_loss !== undefined && (
                      <div className="text-[#00FFA0]">
                        <span className="text-[#9BD8FF]">[BEST]</span> 🎯 Best Val Loss: {trainingLogsTyped.best_val_loss.toFixed(4)}
                      </div>
                    )}
                    {trainingLogsTyped.inference_speed_ms && (
                      <div className="text-[#FFB84D]">
                        <span className="text-[#9BD8FF]">[PERF]</span> ⚡ Inference: {trainingLogsTyped.inference_speed_ms.toFixed(2)} ms/sample
                      </div>
                    )}
                  </>
                )}
                
                {/* Error Status */}
                {(trainingStatus.status === 'error' || trainingStatus.status === 'failed') && (
                  <div className="text-red-400 mt-2 font-semibold border-t border-red-400/20 pt-2">
                    <span className="text-[#9BD8FF]">[ERROR]</span> ❌ {trainingStatus.message || 'Training failed'}
                  </div>
                )}
              </div>
            )}
            
            {/* No Training Status */}
            {(!trainingStatus || trainingStatus.status === 'not_started' || trainingStatus.status === 'idle') && (
              <div className="text-[#9BD8FF]/60 italic text-center py-4">
                💤 Waiting for training to start... Click "Start Training" to begin.
              </div>
            )}
            
            {/* Loading State */}
            {trainingStatus && trainingStatus.status !== 'not_started' && !trainingLogsTyped && (
              <div className="text-[#FFB84D] italic text-center py-4 animate-pulse">
                📡 Loading training logs...
              </div>
            )}
          </div>
          
          <div className="mt-3 p-3 bg-[#0b1220]/50 rounded-lg">
            <div className="flex items-start gap-2 text-xs text-[#9BD8FF]/80">
              <span>💡</span>
              <div>
                <strong className="text-[#E6FBFF]">Pipeline:</strong> Auto-detection → Schema analysis → Task type (regression/classification) → 
                Categorical encoding → Safe splitting → Training → Model verification
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Training Graph */}
        {history && history.length > 0 && renderTrainingChart()}

      {/* Training Logs */}
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl">
        <button
          onClick={() => setShowLogs(!showLogs)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-[#0b1220]/30 transition-colors rounded-t-xl"
        >
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-[#00F3FF]" />
            <span className="font-semibold text-[#E6FBFF]">Training Logs</span>
            <span className="text-xs text-[#9BD8FF] bg-[#0b1220] px-2 py-1 rounded-full">
              {(history?.length ?? 0)} entries
            </span>
          </div>
          {showLogs ? <ChevronUp className="w-5 h-5 text-[#9BD8FF]" /> : <ChevronDown className="w-5 h-5 text-[#9BD8FF]" />}
        </button>
        
        {showLogs && (
          <div className="p-4 border-t border-[#122033] space-y-4">
            {/* Model Size Summary */}
            {(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb ?? trainingStatus?.model_size_kb ?? trainingLogsTyped?.model_size_kb) && (
              <div className="bg-gradient-to-r from-[#00F3FF]/10 to-[#FF00D0]/10 border border-[#00F3FF]/30 rounded-lg p-4">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-[#00F3FF]/20 rounded-lg flex items-center justify-center">
                      <Activity className="w-5 h-5 text-[#00F3FF]" />
                    </div>
                    <div>
                      <div className="text-sm text-[#9BD8FF]">Model Size</div>
                      <div className="text-xl font-bold text-[#E6FBFF]">
                        {(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb) 
                          ? `${(trainingStatus?.model_size_mb ?? trainingLogsTyped?.model_size_mb ?? 0).toFixed(3)} MB`
                          : (trainingStatus?.model_size_kb ?? trainingLogsTyped?.model_size_kb)
                          ? `${(trainingStatus?.model_size_kb ?? trainingLogsTyped?.model_size_kb ?? 0).toFixed(2)} KB`
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                  {(trainingStatus?.total_parameters ?? trainingStatus?.num_parameters ?? trainingLogsTyped?.total_parameters ?? trainingLogsTyped?.num_parameters) && (
                    <div className="text-right">
                      <div className="text-sm text-[#9BD8FF]">Parameters</div>
                      <div className="text-lg font-semibold text-[#E6FBFF]">
                        {(trainingStatus?.total_parameters ?? trainingStatus?.num_parameters ?? trainingLogsTyped?.total_parameters ?? trainingLogsTyped?.num_parameters)?.toLocaleString()}
                      </div>
                    </div>
                  )}
                  {trainingStatus?.model_path && (
                    <div className="text-xs text-[#9BD8FF]/70 font-mono">
                      {trainingStatus.model_path.split('/').pop()}
                    </div>
                  )}
                </div>
              </div>
            )}
            <div className="bg-[#0b0820] rounded-lg p-4 font-mono text-sm max-h-64 overflow-y-auto space-y-1">
              {history && history.length > 0 ? (
                history.map((h) => (
                  <div key={h.epoch} className="text-[#9BD8FF]">
                    {`Epoch ${h.epoch} - loss: ${h.train_loss.toFixed(4)} - val_loss: ${h.val_loss.toFixed(4)} - val_accuracy: ${h.val_accuracy.toFixed(4)}`}
                  </div>
                ))
              ) : (
                <div className="text-[#9BD8FF]/50">No logs available</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Training;