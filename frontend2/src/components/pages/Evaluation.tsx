import React, { useState, useEffect } from 'react';
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
import { evaluationAPI } from '../../services/api';

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMetrics();
  }, [selectedModel]);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await evaluationAPI.getMetrics(selectedModel);
      setMetrics(data);
    } catch (err) {
      setError('Failed to load metrics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);
    try {
      await evaluationAPI.evaluate(selectedModel);
      await loadMetrics();
    } catch (err) {
      setError('Evaluation failed');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const formatChange = (value: number) => {
    if (value > 0) return `+${value.toFixed(2)}%`;
    if (value < 0) return `${value.toFixed(2)}%`;
    return '0%';
  };

  const getTrendIcon = (value: number) => {
    if (value > 0) return <TrendingUp className="w-4 h-4 text-[#00FFA0]" />;
    if (value < 0) return <TrendingDown className="w-4 h-4 text-[#FF3B6B]" />;
    return <Minus className="w-4 h-4 text-[#9BD8FF]" />;
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
                {getTrendIcon(2.3)}
                <span className="text-xs text-[#00FFA0]">{formatChange(2.3)}</span>
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
                {getTrendIcon(1.8)}
                <span className="text-xs text-[#00FFA0]">{formatChange(1.8)}</span>
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
                {getTrendIcon(-0.5)}
                <span className="text-xs text-[#FF3B6B]">{formatChange(-0.5)}</span>
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
                {getTrendIcon(1.2)}
                <span className="text-xs text-[#00FFA0]">{formatChange(1.2)}</span>
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
                  { name: 'Accuracy', value: `${(metrics.accuracy * 100).toFixed(2)}%`, change: 2.3, icon: Target },
                  { name: 'Precision', value: metrics.precision.toFixed(4), change: 1.8, icon: Crosshair },
                  { name: 'Recall', value: metrics.recall.toFixed(4), change: -0.5, icon: Layers },
                  { name: 'F1-Score', value: metrics.f1Score.toFixed(4), change: 1.2, icon: Activity },
                  { name: 'Inference Time', value: `${metrics.inferenceTime.toFixed(2)} ms`, change: -15.3, icon: Activity },
                  { name: 'Model Size', value: `${metrics.modelSize.toFixed(1)} MB`, change: 0, icon: Activity },
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
                          {getTrendIcon(metric.change)}
                          <span className={`text-sm font-medium ${
                            metric.change > 0 ? 'text-[#00FFA0]' : 
                            metric.change < 0 ? 'text-[#FF3B6B]' : 'text-[#9BD8FF]'
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