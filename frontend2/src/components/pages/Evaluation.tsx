import { useState, useEffect } from 'react';
import { 
  Target, 
  Crosshair, 
  Layers, 
  Activity, 
  Download, 
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Radar
} from 'lucide-react';
import { evaluationService } from '../../api/services/evaluationService';
import { modelService } from '../../api/services/modelService';
import { useToast } from '../ui/ToastContainer';
import { useAppStore } from '../../store/useAppStore';

interface MetricData {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  inferenceTime: number;
  modelSize: number;
  confusionMatrix?: number[][];
}

const Evaluation = () => {
  const [selectedModel, setSelectedModel] = useState<'original' | 'compressed'>('original');
  const [metrics, setMetrics] = useState<MetricData | null>(null);
  const [originalMetrics, setOriginalMetrics] = useState<MetricData | null>(null);
  const [compressedMetrics, setCompressedMetrics] = useState<MetricData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState<string | null>(null);
  const { showError, showSuccess } = useToast();
  const { selectedDatasetName, selectedDatasetPath, setSelectedDataset } = useAppStore((s) => ({
    selectedDatasetName: s.selectedDatasetName,
    selectedDatasetPath: s.selectedDatasetPath,
    setSelectedDataset: s.setSelectedDataset,
  }));

  useEffect(() => {
    loadMetrics();
  }, [selectedModel]);

  useEffect(() => {
    // Load both original and compressed metrics for comparison
    (async () => {
      try {
        const original = await evaluationService.metrics('original').catch(() => null);
        const compressed = await evaluationService.metrics('compressed').catch(() => null);
        
        if (original) {
          const mapped: MetricData = {
            accuracy: original.accuracy,
            precision: original.precision,
            recall: original.recall,
            f1Score: original.f1_score,
            inferenceTime: original.inference_time * 1000,
            modelSize: 0,
          };
          setOriginalMetrics(mapped);
        }
        
        if (compressed) {
          const mapped: MetricData = {
            accuracy: compressed.accuracy,
            precision: compressed.precision,
            recall: compressed.recall,
            f1Score: compressed.f1_score,
            inferenceTime: compressed.inference_time * 1000,
            modelSize: 0,
          };
          setCompressedMetrics(mapped);
        }
      } catch {}
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const current = await modelService.current();
        if (current?.dataset_path) {
          setDatasetPath(current.dataset_path);
          const filename = current.dataset_path.split(/[/\\]/).pop() || current.dataset_path;
          setSelectedDataset({ filename, path: current.dataset_path });
        } else if (selectedDatasetPath) {
          setDatasetPath(selectedDatasetPath);
        }
      } catch {}
    })();
  }, []);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const raw = await evaluationService.metrics(selectedModel);
      const mapped: MetricData = {
        accuracy: raw.accuracy,
        precision: raw.precision,
        recall: raw.recall,
        f1Score: raw.f1_score,
        inferenceTime: raw.inference_time * 1000,
        modelSize: 0,
      };
      setMetrics(mapped);
    } catch (err) {
      setError('Failed to load metrics');
      showError('Metrics Error', (err as any)?.message ?? String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);
    try {
      if (!datasetPath) {
        throw new Error('No dataset selected. Start training or select a dataset.');
      }
      await evaluationService.evaluate({ model_type: selectedModel, dataset_path: datasetPath });
      await loadMetrics();
      showSuccess('Evaluation Completed');
    } catch (err) {
      setError('Evaluation failed');
      showError('Evaluation Failed', (err as any)?.message ?? String(err));
    } finally {
      setLoading(false);
    }
  };

  const calculateChange = (current: number, previous: number | null): number => {
    if (!previous || previous === 0) return 0;
    return ((current - previous) / previous) * 100;
  };

  const formatChange = (value: number) => {
    if (value > 0) return `+${value.toFixed(2)}%`;
    if (value < 0) return `${value.toFixed(2)}%`;
    return '0%';
  };

  const getTrendIcon = (value: number, isImprovement: boolean = true) => {
    // For accuracy/precision/recall/f1: higher is better
    // For inference time: lower is better
    const isPositive = isImprovement ? value > 0 : value < 0;
    if (isPositive) return <TrendingUp className="w-4 h-4 text-[#00FFA0]" />;
    if (!isPositive && value !== 0) return <TrendingDown className="w-4 h-4 text-[#FF3B6B]" />;
    return <Minus className="w-4 h-4 text-[#9BD8FF]" />;
  };

  const getChangeForMetric = (metricName: keyof MetricData): number => {
    if (!metrics || !originalMetrics) return 0;
    
    if (selectedModel === 'original') {
      // Compare original to compressed (if available)
      if (!compressedMetrics) return 0;
      return calculateChange(metrics[metricName], compressedMetrics[metricName]);
    } else {
      // Compare compressed to original
      return calculateChange(metrics[metricName], originalMetrics[metricName]);
    }
  };

  const renderConfusionMatrix = () => {
    if (!metrics?.confusionMatrix) return null;

    const matrix = metrics.confusionMatrix;
    const maxValue = Math.max(...matrix.flat());

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h3 className="text-xl font-semibold text-[#E6FBFF] mb-4">Confusion Matrix</h3>
        <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto">
          {matrix.map((row, i) =>
            row.map((value, j) => (
              <div
                key={`${i}-${j}`}
                className="aspect-square flex items-center justify-center rounded-lg text-sm font-semibold relative group cursor-pointer"
                style={{
                  backgroundColor: `rgba(0, 243, 255, ${value / maxValue * 0.8})`,
                  color: value / maxValue > 0.5 ? '#0b1220' : '#E6FBFF'
                }}
              >
                {value}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-[#0b1220] text-[#E6FBFF] text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                  Predicted: {j}, Actual: {i}
                </div>
              </div>
            ))
          )}
        </div>
        <div className="flex justify-between text-xs text-[#9BD8FF] mt-4">
          <span>Predicted</span>
          <span>Actual</span>
        </div>
      </div>
    );
  };

  const renderMetricsChart = () => {
    if (!metrics) return null;

    const chartMetrics = [
      { name: 'Accuracy', value: metrics.accuracy * 100, color: '#00F3FF' },
      { name: 'Precision', value: metrics.precision * 100, color: '#FF00D0' },
      { name: 'Recall', value: metrics.recall * 100, color: '#00FFA0' },
      { name: 'F1-Score', value: metrics.f1Score * 100, color: '#FFB84D' },
    ];

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h3 className="text-xl font-semibold text-[#E6FBFF] mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-[#00F3FF]" />
          Metrics Comparison
        </h3>
        <div className="space-y-4">
          {chartMetrics.map((metric) => (
            <div key={metric.name} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-[#9BD8FF]">{metric.name}</span>
                <span className="text-[#E6FBFF] font-semibold">{metric.value.toFixed(2)}%</span>
              </div>
              <div className="bg-[#0b1220] rounded-full h-2 overflow-hidden">
                <div 
                  className="h-full transition-all duration-1000 ease-out"
                  style={{ 
                    width: `${metric.value}%`,
                    backgroundColor: metric.color,
                    boxShadow: `0 0 10px ${metric.color}40`
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderRadarChart = () => {
    if (!metrics) return null;

    const radarMetrics = [
      { name: 'Accuracy', value: metrics.accuracy },
      { name: 'Precision', value: metrics.precision },
      { name: 'Recall', value: metrics.recall },
      { name: 'F1-Score', value: metrics.f1Score },
    ];

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h3 className="text-xl font-semibold text-[#E6FBFF] mb-4 flex items-center gap-2">
          <Radar className="w-5 h-5 text-[#00F3FF]" />
          Performance Overview
        </h3>
        <div className="relative w-64 h-64 mx-auto">
          <svg width="100%" height="100%" viewBox="0 0 200 200" className="overflow-visible">
            {/* Grid circles */}
            {[0.2, 0.4, 0.6, 0.8, 1.0].map((radius, i) => (
              <circle
                key={i}
                cx="100"
                cy="100"
                r={radius * 80}
                fill="none"
                stroke="#122033"
                strokeWidth="1"
              />
            ))}
            
            {/* Axis lines */}
            {radarMetrics.map((_, i) => {
              const angle = (i * 2 * Math.PI) / radarMetrics.length - Math.PI / 2;
              const x = 100 + Math.cos(angle) * 80;
              const y = 100 + Math.sin(angle) * 80;
              return (
                <line
                  key={i}
                  x1="100"
                  y1="100"
                  x2={x}
                  y2={y}
                  stroke="#122033"
                  strokeWidth="1"
                />
              );
            })}
            
            {/* Data polygon */}
            <polygon
              points={radarMetrics.map((metric, i) => {
                const angle = (i * 2 * Math.PI) / radarMetrics.length - Math.PI / 2;
                const x = 100 + Math.cos(angle) * metric.value * 80;
                const y = 100 + Math.sin(angle) * metric.value * 80;
                return `${x},${y}`;
              }).join(' ')}
              fill="rgba(0, 243, 255, 0.2)"
              stroke="#00F3FF"
              strokeWidth="2"
            />
            
            {/* Data points */}
            {radarMetrics.map((metric, i) => {
              const angle = (i * 2 * Math.PI) / radarMetrics.length - Math.PI / 2;
              const x = 100 + Math.cos(angle) * metric.value * 80;
              const y = 100 + Math.sin(angle) * metric.value * 80;
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#00F3FF"
                  stroke="#0b1220"
                  strokeWidth="2"
                />
              );
            })}
            
            {/* Labels */}
            {radarMetrics.map((metric, i) => {
              const angle = (i * 2 * Math.PI) / radarMetrics.length - Math.PI / 2;
              const x = 100 + Math.cos(angle) * 95;
              const y = 100 + Math.sin(angle) * 95;
              return (
                <text
                  key={i}
                  x={x}
                  y={y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs fill-[#9BD8FF]"
                >
                  {metric.name}
                </text>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  if (loading && !metrics) {
    return (
      <div className="space-y-8">
        <div className="text-center">
          <div className="mx-auto w-16 h-16 text-[#00F3FF] animate-spin">
            <RefreshCw className="w-full h-full" />
          </div>
          <h2 className="text-2xl font-bold text-[#E6FBFF] mt-4">Loading Evaluation...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Model Evaluation
          </h1>
          <p className="text-lg text-[#9BD8FF] mt-2">
            Analyze your model's performance metrics
          </p>
          {selectedDatasetName && (
            <div className="mt-2 flex items-center gap-2 text-sm">
              <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md" title={`Dataset: ${selectedDatasetName} | Model: ${selectedModel === 'original' ? 'Original' : 'Compressed'}`}>
                Dataset: {selectedDatasetName} | Model: {selectedModel === 'original' ? 'Original' : 'Compressed'}
              </span>
            </div>
          )}
        </div>
        
        {/* Model Type Selector */}
        <div className="flex space-x-2 bg-[#0b1220]/50 p-2 rounded-lg border border-[#122033]">
          {(['original', 'compressed'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setSelectedModel(type)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                selectedModel === type
                  ? 'bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] text-white shadow-lg'
                  : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)} Model
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-[#FF3B6B]/20 border border-[#FF3B6B] rounded-lg p-4 text-[#FF3B6B]">
          {error}
        </div>
      )}

      {/* Metrics Overview */}
      {metrics && (
        <div className="grid md:grid-cols-4 gap-6">
          <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 hover:border-[#00F3FF]/30 hover:shadow-[0_0_20px_rgba(0,243,255,0.1)] transition-all duration-300 group">
            <div className="flex items-center justify-between mb-4">
              <Target className="w-8 h-8 text-[#00F3FF]" />
              <div className="flex items-center gap-1">
                {getTrendIcon(getChangeForMetric('accuracy'), true)}
                <span className={`text-xs ${getChangeForMetric('accuracy') > 0 ? 'text-[#00FFA0]' : getChangeForMetric('accuracy') < 0 ? 'text-[#FF3B6B]' : 'text-[#9BD8FF]'}`}>
                  {formatChange(getChangeForMetric('accuracy'))}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-[#9BD8FF]">Accuracy</h3>
              <div className="text-3xl font-bold text-[#E6FBFF] group-hover:text-[#00F3FF] transition-colors">
                {(metrics.accuracy * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 hover:border-[#00F3FF]/30 hover:shadow-[0_0_20px_rgba(0,243,255,0.1)] transition-all duration-300 group">
            <div className="flex items-center justify-between mb-4">
              <Crosshair className="w-8 h-8 text-[#00F3FF]" />
              <div className="flex items-center gap-1">
                {getTrendIcon(getChangeForMetric('precision'), true)}
                <span className={`text-xs ${getChangeForMetric('precision') > 0 ? 'text-[#00FFA0]' : getChangeForMetric('precision') < 0 ? 'text-[#FF3B6B]' : 'text-[#9BD8FF]'}`}>
                  {formatChange(getChangeForMetric('precision'))}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-[#9BD8FF]">Precision</h3>
              <div className="text-3xl font-bold text-[#E6FBFF] group-hover:text-[#00F3FF] transition-colors">
                {metrics.precision.toFixed(4)}
              </div>
            </div>
          </div>

          <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 hover:border-[#FF00D0]/30 hover:shadow-[0_0_20px_rgba(255,0,208,0.1)] transition-all duration-300 group">
            <div className="flex items-center justify-between mb-4">
              <Layers className="w-8 h-8 text-[#FF00D0]" />
              <div className="flex items-center gap-1">
                {getTrendIcon(getChangeForMetric('recall'), true)}
                <span className={`text-xs ${getChangeForMetric('recall') > 0 ? 'text-[#00FFA0]' : getChangeForMetric('recall') < 0 ? 'text-[#FF3B6B]' : 'text-[#9BD8FF]'}`}>
                  {formatChange(getChangeForMetric('recall'))}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-[#9BD8FF]">Recall</h3>
              <div className="text-3xl font-bold text-[#E6FBFF] group-hover:text-[#FF00D0] transition-colors">
                {metrics.recall.toFixed(4)}
              </div>
            </div>
          </div>

          <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 hover:border-[#00FFA0]/30 hover:shadow-[0_0_20px_rgba(0,255,160,0.1)] transition-all duration-300 group">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-[#00FFA0]" />
              <div className="flex items-center gap-1">
                {getTrendIcon(getChangeForMetric('f1Score'), true)}
                <span className={`text-xs ${getChangeForMetric('f1Score') > 0 ? 'text-[#00FFA0]' : getChangeForMetric('f1Score') < 0 ? 'text-[#FF3B6B]' : 'text-[#9BD8FF]'}`}>
                  {formatChange(getChangeForMetric('f1Score'))}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-[#9BD8FF]">F1-Score</h3>
              <div className="text-3xl font-bold text-[#E6FBFF] group-hover:text-[#00FFA0] transition-colors">
                {metrics.f1Score.toFixed(4)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Metrics Table */}
      {metrics && (
        <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
          <h2 className="text-2xl font-semibold text-[#E6FBFF] mb-6">Detailed Metrics</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[#122033]">
                  <th className="text-left py-3 px-4 text-[#9BD8FF] font-medium">Metric</th>
                  <th className="text-left py-3 px-4 text-[#9BD8FF] font-medium">Value</th>
                  <th className="text-left py-3 px-4 text-[#9BD8FF] font-medium">Change</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: 'Accuracy', value: `${(metrics.accuracy * 100).toFixed(2)}%`, change: getChangeForMetric('accuracy'), icon: Target, isImprovement: true },
                  { name: 'Precision', value: metrics.precision.toFixed(4), change: getChangeForMetric('precision'), icon: Crosshair, isImprovement: true },
                  { name: 'Recall', value: metrics.recall.toFixed(4), change: getChangeForMetric('recall'), icon: Layers, isImprovement: true },
                  { name: 'F1-Score', value: metrics.f1Score.toFixed(4), change: getChangeForMetric('f1Score'), icon: Activity, isImprovement: true },
                  { name: 'Inference Time', value: `${metrics.inferenceTime.toFixed(2)} ms`, change: getChangeForMetric('inferenceTime'), icon: Activity, isImprovement: false },
                  { name: 'Model Size', value: `${metrics.modelSize.toFixed(1)} MB`, change: 0, icon: Activity, isImprovement: false },
                ].map((metric, index) => {
                  const IconComponent = metric.icon;
                  return (
                    <tr key={metric.name} className={`hover:bg-[#0b1220]/30 transition-colors ${index % 2 === 0 ? 'bg-[#0b1220]/20' : ''}`}>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <IconComponent className="w-4 h-4 text-[#00F3FF]" />
                          <span className="text-[#E6FBFF]">{metric.name}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-[#E6FBFF] font-semibold">{metric.value}</td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          {getTrendIcon(metric.change, metric.isImprovement)}
                          <span className={`text-sm font-medium ${
                            (metric.isImprovement && metric.change > 0) || (!metric.isImprovement && metric.change < 0) ? 'text-[#00FFA0]' : 
                            (metric.isImprovement && metric.change < 0) || (!metric.isImprovement && metric.change > 0) ? 'text-[#FF3B6B]' : 'text-[#9BD8FF]'
                          }`}>
                            {formatChange(metric.change)}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Charts Section */}
      {metrics && (
        <div className="grid lg:grid-cols-2 gap-8">
          {renderMetricsChart()}
          {renderRadarChart()}
        </div>
      )}

      {/* Confusion Matrix */}
      {metrics && renderConfusionMatrix()}

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-4">
        <button
          onClick={handleEvaluate}
          disabled={loading}
          className="px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          Re-evaluate
        </button>
        
        <button className="px-6 py-3 bg-[#121628] border border-[#122033] rounded-lg font-semibold text-[#00F3FF] hover:border-[#00F3FF] hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] transition-all duration-300 hover:scale-105 flex items-center gap-2">
          <Download className="w-5 h-5" />
          Download Report
        </button>
        
        {selectedModel === 'original' && (
          <button className="px-6 py-3 bg-[#121628] border border-[#122033] rounded-lg font-semibold text-[#FF00D0] hover:border-[#FF00D0] hover:shadow-[0_0_20px_rgba(255,0,208,0.2)] transition-all duration-300 hover:scale-105">
            Evaluate Compressed Model
          </button>
        )}
      </div>
    </div>
  );
};

export default Evaluation;