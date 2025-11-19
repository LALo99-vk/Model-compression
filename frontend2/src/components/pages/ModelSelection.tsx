import { useState } from 'react';
import { 
  Layers, 
  BarChart, 
  GitBranch, 
  Settings, 
  CheckCircle, 
  Info,
  ArrowRight 
} from 'lucide-react';
import { Model } from '../../types';
import { useToast } from '../ui/ToastContainer';
import { useAppStore } from '../../store/useAppStore';

const ModelSelection = () => {
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [showConfig, setShowConfig] = useState(false);
  const { showSuccess, showError } = useToast();
  const selectModel = useAppStore((s) => s.selectModel);
  const selectedDatasetName = useAppStore((s) => s.selectedDatasetName);
  const selectedModelConfig = useAppStore((s) => s.selectedModel);
  
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
      icon: 'layers'
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
      icon: 'bar-chart'
    },
    {
      id: 'decision_tree',
      name: 'Decision Tree',
      type: 'decision_tree',
      description: 'Fast and interpretable for tabular data',
      bestFor: 'Tabular Data, Classification',
      defaultConfig: {
        maxDepth: 10,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
        criterion: 'gini'
      },
      icon: 'git-branch'
    }
  ];

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

  const handleModelSelect = (model: Model) => {
    setSelectedModel(model);
    setShowConfig(true);
  };

  const confirmSelection = async () => {
    try {
      if (!selectedModel) return;
      await selectModel({ model_type: selectedModel.type, task_type: taskType, config: {} });
      showSuccess('Model Selected', selectedModel.name);
    } catch (e: any) {
      showError('Selection Failed', e.message);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Choose Your Model Architecture
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Select the best model for your data type
        </p>
        <div className="mt-2 flex items-center justify-center gap-2 text-sm">
          {selectedDatasetName ? (
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md" title={`Dataset: ${selectedDatasetName} | Model: ${getModelDisplayName()}`}>
              Dataset: {selectedDatasetName} | Model: {getModelDisplayName()}
            </span>
          ) : (
            <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-[#9BD8FF]">No dataset selected (choose on Upload page)</span>
          )}
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-8">
        {models.map((model) => (
          <div
            key={model.id}
            className={`relative bg-[#121628]/50 border rounded-xl p-6 cursor-pointer transition-all duration-300 group hover:scale-105 hover:shadow-[0_0_30px_rgba(0,243,255,0.2)] ${
              selectedModel?.id === model.id
                ? 'border-[#00F3FF] shadow-[0_0_20px_rgba(0,243,255,0.3)]'
                : 'border-[#122033] hover:border-[#00F3FF]/50'
            }`}
            onClick={() => handleModelSelect(model)}
          >
            {/* Background Glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-[rgba(0,243,255,0.02)] to-[rgba(255,0,208,0.02)] rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
            
            {/* Selection Indicator */}
            {selectedModel?.id === model.id && (
              <div className="absolute top-4 right-4">
                <CheckCircle className="w-6 h-6 text-[#00FFA0]" />
              </div>
            )}

            <div className="relative z-10 space-y-4">
              {/* Icon */}
              <div className="text-[#00F3FF] mb-4 relative">
                {getModelIcon(model.icon)}
                <div className="absolute inset-0 bg-[#00F3FF] blur-lg opacity-20 rounded-full"></div>
              </div>

              {/* Model Info */}
              <div>
                <h3 className="text-xl font-bold text-[#E6FBFF] mb-2">{model.name}</h3>
                <p className="text-[#9BD8FF] text-sm mb-4">{model.description}</p>
                <div className="text-xs text-[#00F3FF] font-medium mb-4">
                  Best for: {model.bestFor}
                </div>
              </div>

              {/* Architecture Preview */}
              <div className="bg-[#0b1220]/50 rounded-lg p-3 space-y-2">
                <div className="flex items-center gap-2 mb-2">
                  <Settings className="w-4 h-4 text-[#9BD8FF]" />
                  <span className="text-xs text-[#9BD8FF] font-medium">Default Config</span>
                </div>
                <div className="space-y-1">
                  {Object.entries(model.defaultConfig).slice(0, 3).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-[#9BD8FF]/70 capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                      <span className="text-[#E6FBFF]">{Array.isArray(value) ? value.join(', ') : String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Select Button */}
              <button className={`w-full py-3 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
                selectedModel?.id === model.id
                  ? 'bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] text-white shadow-lg'
                  : 'bg-[#0b1220] border border-[#122033] text-[#00F3FF] hover:border-[#00F3FF] hover:shadow-[0_0_15px_rgba(0,243,255,0.2)]'
              }`}>
                {selectedModel?.id === model.id ? 'Selected' : 'Select Model'}
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Configuration Panel */}
      {showConfig && selectedModel && (
        <div className="max-w-2xl mx-auto bg-[#121628]/50 border border-[#122033] rounded-xl p-8">
          <div className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <Settings className="w-6 h-6 text-[#00F3FF]" />
              <h2 className="text-2xl font-bold text-[#E6FBFF]">Configure {selectedModel.name}</h2>
            </div>

            {/* Task Type Selection */}
            <div className="space-y-3">
              <label className="block text-sm font-medium text-[#E6FBFF]">Task Type</label>
              <div className="flex space-x-4">
                {(['classification', 'regression'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setTaskType(type)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                      taskType === type
                        ? 'bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] text-white'
                        : 'bg-[#0b1220] border border-[#122033] text-[#9BD8FF] hover:border-[#00F3FF]/50'
                    }`}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="grid md:grid-cols-2 gap-6">
              {Object.entries(selectedModel.defaultConfig).map(([key, value]) => (
                <div key={key} className="space-y-2">
                  <label className="flex items-center gap-2 text-sm font-medium text-[#E6FBFF]">
                    {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                    <Info className="w-4 h-4 text-[#9BD8FF] cursor-help" />
                  </label>
                  {typeof value === 'number' ? (
                    <div className="space-y-2">
                      <input
                        type="range"
                        min={0}
                        max={key === 'learningRate' ? 0.1 : 200}
                        step={key === 'learningRate' ? 0.001 : 1}
                        defaultValue={value}
                        className="w-full h-2 bg-[#0b1220] rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="text-xs text-[#9BD8FF] text-right">{value}</div>
                    </div>
                  ) : (
                    <select className="w-full px-3 py-2 bg-[#0b1220] border border-[#122033] rounded-lg text-[#E6FBFF] focus:border-[#00F3FF] focus:outline-none">
                      <option value={String(value)}>{String(value)}</option>
                    </select>
                  )}
                </div>
              ))}
            </div>

            {/* Confirm Button */}
            <button
              onClick={confirmSelection}
              className="w-full py-4 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
            >
              Confirm Selection
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #00F3FF, #FF00D0);
          cursor: pointer;
          box-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
        }
        
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #00F3FF, #FF00D0);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
        }
      `}</style>
    </div>
  );
};

export default ModelSelection;