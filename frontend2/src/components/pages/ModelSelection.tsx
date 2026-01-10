import { useState, useEffect } from 'react';
import { 
  Layers, 
  BarChart, 
  GitBranch, 
  Settings, 
  CheckCircle, 
  Info,
  ArrowRight,
  Zap,
  AlertTriangle,
  FileText,
  Image as ImageIcon,
  Type
} from 'lucide-react';
import { Model } from '../../types';
import { useToast } from '../ui/ToastContainer';
import { useAppStore } from '../../store/useAppStore';

const ModelSelection = () => {
  const [mode, setMode] = useState<'auto' | 'manual'>('auto');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [showConfig, setShowConfig] = useState(false);
  const [autoSelectedModel, setAutoSelectedModel] = useState<Model | null>(null);
  const [datasetType, setDatasetType] = useState<'tabular' | 'image' | 'text' | null>(null);
  const { showError } = useToast();
  const selectModel = useAppStore((s) => s.selectModel);
  const selectedDatasetName = useAppStore((s) => s.selectedDatasetName);
  const selectedDatasetPath = useAppStore((s) => s.selectedDatasetPath);
  const selectedModelConfig = useAppStore((s) => s.selectedModel);
  const datasets = useAppStore((s) => s.datasets);
  
  const getModelDisplayName = () => {
    if (!selectedModelConfig) return 'No Model Selected';
    const modelType = selectedModelConfig.model_type;
    if (modelType === 'cnn') return 'CNN Model';
    if (modelType === 'rnn') return 'RNN Model';
    if (modelType === 'decision_tree') return 'Decision Tree';
    return String(modelType);
  };

  const models: Model[] = [
    {
      id: 'decision_tree',
      name: 'Decision Tree',
      type: 'decision_tree',
      description: 'Fast and interpretable for tabular data',
      bestFor: 'Tabular Data, Classification/Regression',
      defaultConfig: {
        maxDepth: 10,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
        criterion: 'gini'
      },
      icon: 'git-branch',
      compatibleWith: ['tabular']
    },
    {
      id: 'cnn',
      name: 'CNN',
      type: 'cnn',
      description: 'Perfect for image classification tasks',
      bestFor: 'Images, Vision Tasks',
      defaultConfig: {
        convLayers: 3,
        filters: [32, 64, 128],
        learningRate: 0.001,
        dropout: 0.5,
        epochs: 20
      },
      icon: 'layers',
      compatibleWith: ['image']
    },
    {
      id: 'rnn',
      name: 'RNN/LSTM',
      type: 'rnn',
      description: 'Ideal for sequential and time-series data',
      bestFor: 'Sequences, Text, Time Series',
      defaultConfig: {
        hiddenUnits: 128,
        layers: 2,
        learningRate: 0.001,
        dropout: 0.3,
        epochs: 50
      },
      icon: 'bar-chart',
      compatibleWith: ['text', 'tabular']
    }
  ];

  // AUTO MODE: Auto-detect dataset type and recommend model
  useEffect(() => {
    if (mode === 'auto' && selectedDatasetName) {
      detectDatasetTypeAndRecommendModel();
    }
  }, [mode, selectedDatasetName]);

  const detectDatasetTypeAndRecommendModel = () => {
    if (!selectedDatasetName) {
      setDatasetType(null);
      setAutoSelectedModel(null);
      return;
    }

    const filename = selectedDatasetName.toLowerCase();
    
    // Find the selected dataset in the datasets list
    const selectedDataset = datasets.find(d => d.filename === selectedDatasetName);
    
    // Check if it's a folder with images (from backend metadata)
    if (selectedDataset && (selectedDataset as any).type === 'folder' && (selectedDataset as any).image_count > 0) {
      // Folder with images → CNN
      setDatasetType('image');
      const recommended = models.find(m => m.id === 'cnn');
      setAutoSelectedModel(recommended || null);
      return;
    }
    
    // Detect dataset type based on file extension/name (priority order)
    if (filename.endsWith('.txt')) {
      // .txt files → RNN (text data)
      setDatasetType('text');
      const recommended = models.find(m => m.id === 'rnn');
      setAutoSelectedModel(recommended || null);
    } else if (filename.endsWith('.csv') || filename.endsWith('.xlsx') || filename.endsWith('.xls')) {
      setDatasetType('tabular');
      // Tabular → Decision Tree
      const recommended = models.find(m => m.id === 'decision_tree');
      setAutoSelectedModel(recommended || null);
    } else if (filename.includes('image') || filename.includes('img') || filename.includes('pic')) {
      setDatasetType('image');
      // Images → CNN
      const recommended = models.find(m => m.id === 'cnn');
      setAutoSelectedModel(recommended || null);
    } else if (filename.includes('text') || filename.includes('nlp')) {
      setDatasetType('text');
      // Text → RNN
      const recommended = models.find(m => m.id === 'rnn');
      setAutoSelectedModel(recommended || null);
    } else {
      // Default to tabular for unknown
      setDatasetType('tabular');
      const recommended = models.find(m => m.id === 'decision_tree');
      setAutoSelectedModel(recommended || null);
    }
  };

  const getModelIcon = (icon: string) => {
    switch (icon) {
      case 'layers':
        return <Layers className="w-8 h-8" />;
      case 'bar-chart':
        return <BarChart className="w-8 h-8" />;
      case 'git-branch':
        return <GitBranch className="w-8 h-8" />;
      default:
        return <Layers className="w-8 h-8" />;
    }
  };

  const getDatasetTypeIcon = () => {
    if (datasetType === 'tabular') return <FileText className="w-5 h-5 text-[#00F3FF]" />;
    if (datasetType === 'image') return <ImageIcon className="w-5 h-5 text-[#FF00D0]" />;
    if (datasetType === 'text') return <Type className="w-5 h-5 text-[#00FFA0]" />;
    return <FileText className="w-5 h-5 text-[#9BD8FF]" />;
  };

  // Validate dataset-model compatibility
  const validateCompatibility = (model: Model): { compatible: boolean; reason?: string } => {
    if (!datasetType) {
      return { compatible: false, reason: 'Dataset type not detected. Please upload a valid dataset first.' };
    }

    const modelCompatibility = model.compatibleWith || [];
    
    if (!modelCompatibility.includes(datasetType)) {
      let reason = '';
      if (model.id === 'cnn' && datasetType !== 'image') {
        reason = 'CNN models require image datasets. Your dataset appears to be ' + datasetType + '.';
      } else if (model.id === 'decision_tree' && datasetType !== 'tabular') {
        reason = 'Decision Tree models require tabular data (CSV). Your dataset appears to be ' + datasetType + '.';
      } else if (model.id === 'rnn' && !['text', 'tabular'].includes(datasetType)) {
        reason = 'RNN models require text or sequential data. Your dataset appears to be ' + datasetType + '.';
      } else {
        reason = `This model is not compatible with ${datasetType} data.`;
      }
      return { compatible: false, reason };
    }

    return { compatible: true };
  };

  const handleModelSelect = (model: Model) => {
    // MANUAL MODE: Validate compatibility
    if (mode === 'manual') {
      const validation = validateCompatibility(model);
      
      if (!validation.compatible) {
        showError('Incompatible', validation.reason?.slice(0, 40) || 'Model not compatible');
        return;
      }
    }

    setSelectedModel(model);
    setShowConfig(true);
  };

  const confirmSelection = async () => {
    try {
      const modelToConfirm = mode === 'auto' ? autoSelectedModel : selectedModel;
      
      if (!modelToConfirm) return;

      // Final compatibility check
      const validation = validateCompatibility(modelToConfirm);
      if (!validation.compatible) {
        showError('Incompatible', validation.reason?.slice(0, 40) || 'Model not compatible');
        return;
      }

      const modelConfig = {
        model_type: modelToConfirm.type,
        task_type: taskType,
        config: modelToConfirm.defaultConfig,
        dataset_path: selectedDatasetPath || ''
      };

      selectModel(modelConfig);
      
      // STEP 2 → STEP 3: Auto-navigate to validation after model selection
      // No toast - just navigate (UI shows selection)
      setTimeout(() => {
        const event = new CustomEvent('navigate-to', { detail: 'validation' });
        window.dispatchEvent(event);
      }, 500);  // Reduced delay
      
    } catch (error: any) {
      showError('Error', 'Failed to select model');
    }
  };

  const confirmAutoSelection = () => {
    if (!autoSelectedModel) {
      showError('No Model', 'No model was auto-selected.');
      return;
    }
    setSelectedModel(autoSelectedModel);
    confirmSelection();
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Model Selection
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Choose how to select your AI model
        </p>
        {selectedDatasetName && (
        <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md" title={selectedDatasetName}>
              Dataset: {selectedDatasetName}
            </span>
            {datasetType && (
              <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-[#E6FBFF] font-medium flex items-center gap-2">
                {getDatasetTypeIcon()}
                Type: {datasetType.charAt(0).toUpperCase() + datasetType.slice(1)}
              </span>
          )}
        </div>
        )}
        {/* Show auto-selected model in auto mode, or stored model in manual mode */}
        {(mode === 'auto' && autoSelectedModel) && (
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00FFA0]/60 rounded-full text-[#00FFA0] font-medium">
              ✓ Recommended: {autoSelectedModel.name}
            </span>
          </div>
        )}
        {(mode === 'manual' && selectedModelConfig) && (
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00FFA0]/60 rounded-full text-[#00FFA0] font-medium">
              ✓ Selected: {getModelDisplayName()}
            </span>
          </div>
        )}
      </div>

      {/* Mode Selection */}
      <div className="flex justify-center">
        <div className="flex space-x-2 bg-[#0b1220]/50 p-2 rounded-lg border border-[#122033]">
          <button
            onClick={() => {
              setMode('auto');
              setSelectedModel(null);
              setShowConfig(false);
            }}
            className={`px-8 py-3 rounded-lg font-semibold transition-all duration-200 flex items-center gap-2 ${
              mode === 'auto'
                ? 'bg-gradient-to-r from-[#00FFA0] to-[#00D67F] text-white shadow-lg'
                : 'text-[#9BD8FF] hover:text-[#00FFA0] hover:bg-[#121628]'
            }`}
          >
            <Zap className="w-5 h-5" />
            AUTO MODE (Recommended)
          </button>
          <button
            onClick={() => {
              setMode('manual');
              setSelectedModel(null);
              setShowConfig(false);
            }}
            className={`px-8 py-3 rounded-lg font-semibold transition-all duration-200 flex items-center gap-2 ${
              mode === 'manual'
                ? 'bg-gradient-to-r from-[#00F3FF] to-[#0088FF] text-white shadow-lg'
                : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
            }`}
          >
            <Settings className="w-5 h-5" />
            MANUAL MODE (Advanced)
          </button>
        </div>
      </div>
            
      {/* AUTO MODE Content */}
      {mode === 'auto' && (
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="bg-[#121628]/50 border border-[#00FFA0]/30 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-[#00FFA0]/10 rounded-lg">
                <Zap className="w-8 h-8 text-[#00FFA0]" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-[#E6FBFF] mb-2">Automatic Model Selection</h3>
                <p className="text-[#9BD8FF] mb-4">
                  We've analyzed your dataset and automatically selected the best model for you.
                </p>
                
                {autoSelectedModel ? (
                  <div className="bg-[#0b1220]/50 border border-[#122033] rounded-lg p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="p-4 bg-gradient-to-br from-[#00FFA0]/20 to-transparent rounded-xl border border-[#00FFA0]/30">
                          {getModelIcon(autoSelectedModel.icon)}
              </div>
              <div>
                          <h4 className="text-2xl font-bold text-[#E6FBFF]">{autoSelectedModel.name}</h4>
                          <p className="text-[#9BD8FF] mt-1">{autoSelectedModel.description}</p>
                </div>
                      </div>
                      <CheckCircle className="w-12 h-12 text-[#00FFA0]" />
              </div>

                    <div className="flex items-center gap-2 text-sm">
                      <Info className="w-4 h-4 text-[#00F3FF]" />
                      <span className="text-[#9BD8FF]">Best for: {autoSelectedModel.bestFor}</span>
                    </div>

                    <div className="pt-4 border-t border-[#122033]">
                      <label className="block text-sm font-medium text-[#9BD8FF] mb-2">
                        Task Type
                      </label>
                      <div className="flex gap-3">
                        <button
                          onClick={() => setTaskType('classification')}
                          className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                            taskType === 'classification'
                              ? 'bg-[#00F3FF]/10 border-[#00F3FF] text-[#00F3FF]'
                              : 'border-[#122033] text-[#9BD8FF] hover:border-[#00F3FF]/50'
                          }`}
                        >
                          Classification
                        </button>
                        <button
                          onClick={() => setTaskType('regression')}
                          className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                            taskType === 'regression'
                              ? 'bg-[#00F3FF]/10 border-[#00F3FF] text-[#00F3FF]'
                              : 'border-[#122033] text-[#9BD8FF] hover:border-[#00F3FF]/50'
                          }`}
                        >
                          Regression
                        </button>
                </div>
                    </div>

                    <button
                      onClick={confirmAutoSelection}
                      className="w-full px-6 py-3 bg-gradient-to-r from-[#00FFA0] to-[#00D67F] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,255,160,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
                    >
                      <CheckCircle className="w-5 h-5" />
                      Confirm Auto Selection
                      <ArrowRight className="w-5 h-5" />
                    </button>
                  </div>
                ) : (
                  <div className="bg-[#0b1220]/50 border border-[#FF3B6B]/30 rounded-lg p-6 text-center">
                    <AlertTriangle className="w-12 h-12 text-[#FFB84D] mx-auto mb-3" />
                    <p className="text-[#9BD8FF]">
                      No dataset detected. Please upload a dataset first to enable auto selection.
                    </p>
                  </div>
                )}
              </div>
                </div>
              </div>

          {/* Info about Auto Mode */}
          <div className="bg-[#0b1220]/30 border border-[#122033] rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-[#00F3FF] mt-0.5" />
              <div className="text-sm text-[#9BD8FF]">
                <p className="font-medium text-[#E6FBFF] mb-1">How Auto Mode Works:</p>
                <ul className="space-y-1 list-disc list-inside">
                  <li>Tabular Data (CSV) → Decision Tree</li>
                  <li>Image Data → CNN</li>
                  <li>Text/Sequential Data → RNN</li>
                </ul>
              </div>
            </div>
          </div>
      </div>
      )}

      {/* MANUAL MODE Content */}
      {mode === 'manual' && (
        <div className="max-w-6xl mx-auto space-y-6">
          <div className="bg-[#121628]/50 border border-[#00F3FF]/30 rounded-xl p-6 mb-6">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-[#00F3FF]/10 rounded-lg">
                <Settings className="w-8 h-8 text-[#00F3FF]" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-[#E6FBFF] mb-2">Manual Model Selection</h3>
                <p className="text-[#9BD8FF]">
                  Choose a model manually. The system will validate compatibility with your dataset.
                </p>
              </div>
            </div>
          </div>

          {/* Model Grid */}
          <div className="grid md:grid-cols-3 gap-6">
            {models.map((model) => {
              const validation = validateCompatibility(model);
              const isCompatible = validation.compatible;

              return (
                <div
                  key={model.id}
                  className={`bg-[#121628]/50 border rounded-xl p-6 transition-all duration-300 cursor-pointer ${
                    selectedModel?.id === model.id
                      ? 'border-[#00F3FF] shadow-[0_0_30px_rgba(0,243,255,0.2)] scale-105'
                      : isCompatible
                      ? 'border-[#122033] hover:border-[#00F3FF]/50 hover:scale-105'
                      : 'border-[#FF3B6B]/30 opacity-60 cursor-not-allowed'
                  }`}
                  onClick={() => isCompatible && handleModelSelect(model)}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-4 rounded-xl border ${
                      selectedModel?.id === model.id
                        ? 'bg-[#00F3FF]/10 border-[#00F3FF]'
                        : isCompatible
                        ? 'bg-gradient-to-br from-[#00F3FF]/10 to-transparent border-[#122033]'
                        : 'bg-[#FF3B6B]/10 border-[#FF3B6B]/30'
                    }`}>
                      {getModelIcon(model.icon)}
                    </div>
                    {!isCompatible && (
                      <AlertTriangle className="w-6 h-6 text-[#FFB84D]" />
                    )}
                    {selectedModel?.id === model.id && (
                      <CheckCircle className="w-8 h-8 text-[#00F3FF]" />
                    )}
                  </div>

                  <h3 className="text-xl font-bold text-[#E6FBFF] mb-2">{model.name}</h3>
                  <p className="text-[#9BD8FF] text-sm mb-4">{model.description}</p>
                  
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs">
                      <div className={`w-2 h-2 rounded-full ${isCompatible ? 'bg-[#00FFA0]' : 'bg-[#FF3B6B]'}`}></div>
                      <span className="text-[#9BD8FF]">Best for: {model.bestFor}</span>
                    </div>
                    {!isCompatible && (
                      <div className="mt-3 p-2 bg-[#FF3B6B]/10 border border-[#FF3B6B]/30 rounded text-xs text-[#FFB84D]">
                        {validation.reason}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Configuration Panel */}
          {selectedModel && showConfig && (
            <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 space-y-6">
              <h3 className="text-xl font-semibold text-[#E6FBFF]">Configure {selectedModel.name}</h3>
              
              <div>
                <label className="block text-sm font-medium text-[#9BD8FF] mb-2">
                  Task Type
                </label>
                <div className="flex gap-3">
                  <button
                    onClick={() => setTaskType('classification')}
                    className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                      taskType === 'classification'
                        ? 'bg-[#00F3FF]/10 border-[#00F3FF] text-[#00F3FF]'
                        : 'border-[#122033] text-[#9BD8FF] hover:border-[#00F3FF]/50'
                    }`}
                  >
                    Classification
                  </button>
                  <button
                    onClick={() => setTaskType('regression')}
                    className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                      taskType === 'regression'
                        ? 'bg-[#00F3FF]/10 border-[#00F3FF] text-[#00F3FF]'
                        : 'border-[#122033] text-[#9BD8FF] hover:border-[#00F3FF]/50'
                    }`}
                  >
                    Regression
                  </button>
                </div>
            </div>

            <button
              onClick={confirmSelection}
                className="w-full px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
            >
                <CheckCircle className="w-5 h-5" />
              Confirm Selection
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelSelection;
