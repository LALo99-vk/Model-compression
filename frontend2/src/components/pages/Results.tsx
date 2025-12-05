import { useState, useEffect } from 'react';
import { 
  TrendingDown, 
  CheckCircle, 
  Zap, 
  Download, 
  Share2, 
  RotateCcw,
  FileDown,
  ArrowRight,
  BarChart3,
  Radar,
  Target,
  Clock,
  HardDrive,
  Activity
} from 'lucide-react';
import { comparisonService } from '../../api/services/comparisonService';
import { useToast } from '../ui/ToastContainer';

interface ComparisonData {
  original: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    inferenceTime: number;
    modelSize: number;
    parameters: number;
  };
  compressed: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    inferenceTime: number;
    modelSize: number;
    parameters: number;
  };
  improvements: {
    sizeReduction: number;
    accuracyPreserved: number;
    speedImprovement: number;
  };
}

const Results = () => {
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(false);
  const [showCelebration, setShowCelebration] = useState(false);
  const [viewMode, setViewMode] = useState<'comparison' | 'original' | 'compressed'>('comparison');

  const { showError, showInfo } = useToast();
  
  useEffect(() => {
    console.log('🔄 Results page mounted - loading comparison data...');
    loadComparison();
  }, []);
  
  // Debug: Log when comparison data changes
  useEffect(() => {
    if (comparisonData) {
      console.log('📊 Comparison data updated:', comparisonData);
    }
  }, [comparisonData]);

  useEffect(() => {
    if (comparisonData) {
      setShowCelebration(true);
      const timer = setTimeout(() => setShowCelebration(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [comparisonData]);

  // STEP 7: Download functions
  const downloadOriginalModel = async () => {
    try {
      console.log('📥 Downloading original model...');
      const response = await fetch('http://localhost:8000/api/model/download/original');
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      
      const blob = await response.blob();
      const sizeMB = (blob.size / (1024 * 1024)).toFixed(4);
      console.log(`✅ Downloaded ${blob.size.toLocaleString()} bytes (${sizeMB} MB)`);
      
      // Get filename from Content-Disposition header
      const disposition = response.headers.get('content-disposition');
      let filename = 'original_model.pt';
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
      
      showSuccess('Download Complete', `Original model downloaded (${sizeMB} MB)`);
    } catch (error) {
      console.error('Download failed:', error);
      showError('Download Failed', 'Could not download original model');
    }
  };

  const downloadCompressedModel = async () => {
    try {
      console.log('📥 Downloading compressed model...');
      const response = await fetch('http://localhost:8000/api/model/download/compressed');
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      
      const blob = await response.blob();
      const sizeMB = (blob.size / (1024 * 1024)).toFixed(4);
      console.log(`✅ Downloaded ${blob.size.toLocaleString()} bytes (${sizeMB} MB)`);
      
      // Get filename from Content-Disposition header
      const disposition = response.headers.get('content-disposition');
      let filename = 'compressed_model.pt';
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
    } catch (error) {
      console.error('Download failed:', error);
      showError('Download Failed', 'Could not download compressed model');
    }
  };

  const loadComparison = async () => {
    setLoading(true);
    try {
      const data = await comparisonService.compare();
      
      // Backend returns nested format with file_size, accuracy, detailed_metrics
      // Extract real values
      const originalSizeMB = data.file_size?.original_mb ?? 0;
      const compressedSizeMB = data.file_size?.compressed_mb ?? 0;
      const originalParams = data.detailed_metrics?.original?.parameters ?? 0;
      const compressedParams = data.detailed_metrics?.compressed?.parameters ?? 0;
      const sizeReduction = data.file_size?.reduction_percent ?? 0;
      
      const originalAcc = data.accuracy?.original ?? data.detailed_metrics?.original?.accuracy ?? 0;
      const compressedAcc = data.accuracy?.compressed ?? data.detailed_metrics?.compressed?.accuracy ?? 0;
      
      setComparisonData({
        original: {
          accuracy: originalAcc,
          precision: data.detailed_metrics?.original?.precision ?? 0,
          recall: data.detailed_metrics?.original?.recall ?? 0,
          f1Score: data.detailed_metrics?.original?.f1_score ?? 0,
          inferenceTime: data.inference_time?.original_ms ?? 0,
          modelSize: originalSizeMB,
          parameters: originalParams,
        },
        compressed: {
          accuracy: compressedAcc,
          precision: data.detailed_metrics?.compressed?.precision ?? 0,
          recall: data.detailed_metrics?.compressed?.recall ?? 0,
          f1Score: data.detailed_metrics?.compressed?.f1_score ?? 0,
          inferenceTime: data.inference_time?.compressed_ms ?? 0,
          modelSize: compressedSizeMB,
          parameters: compressedParams,
        },
        improvements: {
          sizeReduction: sizeReduction,
          accuracyPreserved: 100 + (data.accuracy?.difference_percent ?? 0),
          speedImprovement: data.inference_time?.speedup ?? 1,
        },
      });
      
      console.log('✅ Raw API response:', data);
      console.log('✅ Extracted values:', {
        originalSizeMB,
        compressedSizeMB,
        originalParams,
        compressedParams,
        sizeReduction
      });
      console.log('✅ Setting comparison data:', {
        original: { size: originalSizeMB, params: originalParams },
        compressed: { size: compressedSizeMB, params: compressedParams },
        reduction: sizeReduction
      });
      
    } catch (error) {
      // Don't show error if no comparison data exists yet
      if ((error as any)?.response?.status !== 404) {
        showError('Comparison Failed', (error as any)?.message ?? String(error));
      }
    } finally {
      setLoading(false);
    }
  };

  const getChangeColor = (change: number, isImprovement: boolean = true) => {
    if (change === 0) return 'text-[#9BD8FF]';
    const isPositive = isImprovement ? change > 0 : change < 0;
    return isPositive ? 'text-[#00FFA0]' : 'text-[#FF3B6B]';
  };

  const getChangeIcon = (change: number, isImprovement: boolean = true) => {
    if (change === 0) return null;
    const isPositive = isImprovement ? change > 0 : change < 0;
    return isPositive ? '↑' : '↓';
  };

  const formatChange = (original: number, compressed: number, isPercentage: boolean = false, isImprovement: boolean = true) => {
    const change = compressed - original;
    const changePercent = (change / original) * 100;
    const displayValue = isPercentage ? changePercent : change;
    const icon = getChangeIcon(displayValue, isImprovement);
    
    return {
      value: Math.abs(displayValue),
      icon,
      color: getChangeColor(displayValue, isImprovement),
      isPositive: isImprovement ? displayValue > 0 : displayValue < 0
    };
  };

  const renderHeroStats = () => {
    if (!comparisonData) return null;
    
    // Hide improvement stats when viewing individual models
    if (viewMode !== 'comparison') return null;
    
    return (
      <div className="grid md:grid-cols-3 gap-8 mb-12">
        <div className={`bg-[#121628]/50 border border-[#122033] rounded-xl p-8 text-center relative overflow-hidden ${showCelebration ? 'animate-pulse' : ''}`}>
          <div className="absolute inset-0 bg-gradient-to-br from-[rgba(0,255,160,0.05)] to-transparent"></div>
          <div className="relative z-10">
            <TrendingDown className="w-12 h-12 text-[#00FFA0] mx-auto mb-4" />
            <div className="text-4xl font-bold text-[#00FFA0] mb-2">
              {comparisonData.improvements.sizeReduction.toFixed(1)}%
            </div>
            <div className="text-lg font-semibold text-[#E6FBFF] mb-1">Size Reduction</div>
            <div className="text-sm text-[#9BD8FF]">Model is {comparisonData.improvements.sizeReduction.toFixed(1)}% smaller</div>
          </div>
        </div>

        <div className={`bg-[#121628]/50 border border-[#122033] rounded-xl p-8 text-center relative overflow-hidden ${showCelebration ? 'animate-pulse' : ''}`}>
          <div className="absolute inset-0 bg-gradient-to-br from-[rgba(0,243,255,0.05)] to-transparent"></div>
          <div className="relative z-10">
            <CheckCircle className="w-12 h-12 text-[#00F3FF] mx-auto mb-4" />
            <div className="text-4xl font-bold text-[#00F3FF] mb-2">
              {comparisonData.improvements.accuracyPreserved.toFixed(1)}%
            </div>
            <div className="text-lg font-semibold text-[#E6FBFF] mb-1">Accuracy Preserved</div>
            <div className="text-sm text-[#9BD8FF]">Minimal accuracy loss</div>
          </div>
        </div>

        <div className={`bg-[#121628]/50 border border-[#122033] rounded-xl p-8 text-center relative overflow-hidden ${showCelebration ? 'animate-pulse' : ''}`}>
          <div className="absolute inset-0 bg-gradient-to-br from-[rgba(255,184,77,0.05)] to-transparent"></div>
          <div className="relative z-10">
            <Zap className="w-12 h-12 text-[#FFB84D] mx-auto mb-4" />
            <div className="text-4xl font-bold text-[#FFB84D] mb-2">
              {comparisonData.improvements.speedImprovement.toFixed(1)}x
            </div>
            <div className="text-lg font-semibold text-[#E6FBFF] mb-1">Speed Improvement</div>
            <div className="text-sm text-[#9BD8FF]">Faster inference time</div>
          </div>
        </div>
      </div>
    );
  };

  const renderComparisonTable = () => {
    if (!comparisonData) return null;
    
    // Filter columns based on view mode
    const showOriginal = viewMode === 'comparison' || viewMode === 'original';
    const showCompressed = viewMode === 'comparison' || viewMode === 'compressed';
    
    const metrics = [
      { 
        name: 'File Size (MB)', 
        icon: HardDrive,
        original: comparisonData.original.modelSize, 
        compressed: comparisonData.compressed.modelSize,
        format: (val: number) => `${val.toFixed(1)} MB`,
        isImprovement: false // Lower is better
      },
      { 
        name: 'Accuracy', 
        icon: Target,
        original: comparisonData.original.accuracy, 
        compressed: comparisonData.compressed.accuracy,
        format: (val: number) => `${(val * 100).toFixed(2)}%`,
        isImprovement: true
      },
      { 
        name: 'Precision', 
        icon: Target,
        original: comparisonData.original.precision, 
        compressed: comparisonData.compressed.precision,
        format: (val: number) => val.toFixed(4),
        isImprovement: true
      },
      { 
        name: 'Recall', 
        icon: Activity,
        original: comparisonData.original.recall, 
        compressed: comparisonData.compressed.recall,
        format: (val: number) => val.toFixed(4),
        isImprovement: true
      },
      { 
        name: 'F1-Score', 
        icon: Activity,
        original: comparisonData.original.f1Score, 
        compressed: comparisonData.compressed.f1Score,
        format: (val: number) => val.toFixed(4),
        isImprovement: true
      },
      { 
        name: 'Inference Time (ms)', 
        icon: Clock,
        original: comparisonData.original.inferenceTime, 
        compressed: comparisonData.compressed.inferenceTime,
        format: (val: number) => `${val.toFixed(1)} ms`,
        isImprovement: false // Lower is better
      },
      { 
        name: 'Parameters', 
        icon: Activity,
        original: comparisonData.original.parameters, 
        compressed: comparisonData.compressed.parameters,
        format: (val: number) => val.toLocaleString(),
        isImprovement: false // Lower is better for compression
      },
    ];

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 mb-8">
        <h2 className="text-2xl font-semibold text-[#E6FBFF] mb-6">
          {viewMode === 'comparison' ? 'Detailed Comparison' : viewMode === 'original' ? 'Original Model Metrics' : 'Compressed Model Metrics'}
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#122033]">
                <th className="text-left py-4 px-4 text-[#9BD8FF] font-medium">Metric</th>
                {showOriginal && <th className="text-left py-4 px-4 text-[#9BD8FF] font-medium">Original Model</th>}
                {showCompressed && <th className="text-left py-4 px-4 text-[#9BD8FF] font-medium">Compressed Model</th>}
                {viewMode === 'comparison' && <th className="text-left py-4 px-4 text-[#9BD8FF] font-medium">Change</th>}
              </tr>
            </thead>
            <tbody>
              {metrics.map((metric, index) => {
                const IconComponent = metric.icon;
                const change = formatChange(metric.original, metric.compressed, false, metric.isImprovement);
                
                return (
                  <tr key={metric.name} className={`hover:bg-[#0b1220]/30 transition-colors ${index % 2 === 0 ? 'bg-[#0b1220]/20' : ''}`}>
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        <IconComponent className="w-4 h-4 text-[#00F3FF]" />
                        <span className="text-[#E6FBFF] font-medium">{metric.name}</span>
                      </div>
                    </td>
                    {showOriginal && <td className="py-4 px-4 text-[#9BD8FF]">{metric.format(metric.original)}</td>}
                    {showCompressed && <td className="py-4 px-4 text-[#E6FBFF] font-semibold">{metric.format(metric.compressed)}</td>}
                    {viewMode === 'comparison' && (
                      <td className="py-4 px-4">
                        <div className={`flex items-center gap-1 font-medium ${change.color}`}>
                          <span>{change.icon}</span>
                          <span>{metric.format(change.value)}</span>
                        </div>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderBarChart = () => {
    if (!comparisonData) return null;
    
    const showOriginal = viewMode === 'comparison' || viewMode === 'original';
    const showCompressed = viewMode === 'comparison' || viewMode === 'compressed';
    
    const chartData = [
      { name: 'Accuracy', original: comparisonData.original.accuracy * 100, compressed: comparisonData.compressed.accuracy * 100 },
      { name: 'Precision', original: comparisonData.original.precision * 100, compressed: comparisonData.compressed.precision * 100 },
      { name: 'Recall', original: comparisonData.original.recall * 100, compressed: comparisonData.compressed.recall * 100 },
      { name: 'F1-Score', original: comparisonData.original.f1Score * 100, compressed: comparisonData.compressed.f1Score * 100 },
    ];

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h3 className="text-xl font-semibold text-[#E6FBFF] mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-[#00F3FF]" />
          {viewMode === 'comparison' ? 'Performance Comparison' : 'Performance Metrics'}
        </h3>
        <div className="space-y-6">
          {chartData.map((item) => (
            <div key={item.name} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-[#9BD8FF]">{item.name}</span>
                <div className="flex gap-4">
                  {showOriginal && <span className="text-[#00F3FF]">Original: {item.original.toFixed(1)}%</span>}
                  {showCompressed && <span className="text-[#FF00D0]">Compressed: {item.compressed.toFixed(1)}%</span>}
                </div>
              </div>
              <div className="flex gap-2">
                {showOriginal && (
                  <div className="flex-1 bg-[#0b1220] rounded-full h-3 overflow-hidden">
                    <div 
                      className="h-full bg-[#00F3FF] transition-all duration-1000 ease-out"
                      style={{ width: `${item.original}%` }}
                    ></div>
                  </div>
                )}
                {showCompressed && (
                  <div className="flex-1 bg-[#0b1220] rounded-full h-3 overflow-hidden">
                    <div 
                      className="h-full bg-[#FF00D0] transition-all duration-1000 ease-out"
                      style={{ width: `${item.compressed}%` }}
                    ></div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        {viewMode === 'comparison' && (
          <div className="flex justify-center gap-6 mt-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-[#00F3FF] rounded-full"></div>
              <span className="text-[#9BD8FF]">Original Model</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-[#FF00D0] rounded-full"></div>
              <span className="text-[#9BD8FF]">Compressed Model</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderRadarChart = () => {
    if (!comparisonData) return null;
    
    const radarData = [
      { name: 'Accuracy', original: comparisonData.original.accuracy, compressed: comparisonData.compressed.accuracy },
      { name: 'Precision', original: comparisonData.original.precision, compressed: comparisonData.compressed.precision },
      { name: 'Recall', original: comparisonData.original.recall, compressed: comparisonData.compressed.recall },
      { name: 'F1-Score', original: comparisonData.original.f1Score, compressed: comparisonData.compressed.f1Score },
    ];

    return (
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h3 className="text-xl font-semibold text-[#E6FBFF] mb-4 flex items-center gap-2">
          <Radar className="w-5 h-5 text-[#00F3FF]" />
          Multi-Metric Overview
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
            {radarData.map((_, i) => {
              const angle = (i * 2 * Math.PI) / radarData.length - Math.PI / 2;
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
            
            {/* Original model polygon */}
            <polygon
              points={radarData.map((item, i) => {
                const angle = (i * 2 * Math.PI) / radarData.length - Math.PI / 2;
                const x = 100 + Math.cos(angle) * item.original * 80;
                const y = 100 + Math.sin(angle) * item.original * 80;
                return `${x},${y}`;
              }).join(' ')}
              fill="rgba(0, 243, 255, 0.1)"
              stroke="#00F3FF"
              strokeWidth="2"
            />
            
            {/* Compressed model polygon */}
            <polygon
              points={radarData.map((item, i) => {
                const angle = (i * 2 * Math.PI) / radarData.length - Math.PI / 2;
                const x = 100 + Math.cos(angle) * item.compressed * 80;
                const y = 100 + Math.sin(angle) * item.compressed * 80;
                return `${x},${y}`;
              }).join(' ')}
              fill="rgba(255, 0, 208, 0.1)"
              stroke="#FF00D0"
              strokeWidth="2"
            />
            
            {/* Labels */}
            {radarData.map((item, i) => {
              const angle = (i * 2 * Math.PI) / radarData.length - Math.PI / 2;
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
                  {item.name}
                </text>
              );
            })}
          </svg>
        </div>
        <div className="flex justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#00F3FF] rounded-full"></div>
            <span className="text-[#9BD8FF]">Original</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#FF00D0] rounded-full"></div>
            <span className="text-[#9BD8FF]">Compressed</span>
          </div>
        </div>
      </div>
    );
  };

  const renderAnalysisCard = () => {
    if (!comparisonData) return null;
    
    return (
    <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 mb-8">
      <h2 className="text-2xl font-semibold text-[#E6FBFF] mb-6">Compression Analysis</h2>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-[#00FFA0] mt-0.5" />
            <div>
              <h4 className="text-[#E6FBFF] font-semibold">Excellent size reduction achieved</h4>
              <p className="text-[#9BD8FF] text-sm">Model size reduced by 73.5% with minimal impact</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-[#00FFA0] mt-0.5" />
            <div>
              <h4 className="text-[#E6FBFF] font-semibold">Minimal accuracy loss (&lt; 2%)</h4>
              <p className="text-[#9BD8FF] text-sm">Accuracy preserved at 98.2% of original performance</p>
            </div>
          </div>
        </div>
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <Zap className="w-5 h-5 text-[#FFB84D] mt-0.5" />
            <div>
              <h4 className="text-[#E6FBFF] font-semibold">Significant speedup for inference</h4>
              <p className="text-[#9BD8FF] text-sm">3.8x faster inference time for real-time applications</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-5 h-5 text-[#FFB84D] mt-0.5">⚠️</div>
            <div>
              <h4 className="text-[#E6FBFF] font-semibold">Consider re-training if accuracy critical</h4>
              <p className="text-[#9BD8FF] text-sm">For mission-critical applications, fine-tuning may help</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 p-4 bg-[#00FFA0]/10 border border-[#00FFA0]/20 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <CheckCircle className="w-5 h-5 text-[#00FFA0]" />
          <span className="text-[#00FFA0] font-semibold">Recommendation</span>
        </div>
        <p className="text-[#E6FBFF]">
          This compression achieved excellent results. The model is ready for deployment with significant 
          performance improvements and minimal accuracy trade-offs.
        </p>
      </div>
    </div>
    );
  };

  const renderFileSizeVisualization = () => {
    if (!comparisonData) return null;
    
    return (
    <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6 mb-8">
      <h2 className="text-2xl font-semibold text-[#E6FBFF] mb-6 text-center">File Size Visualization</h2>
      <div className="flex items-center justify-center gap-8">
        <div className="text-center">
          <div className="w-32 h-20 bg-[#9BD8FF]/20 border-2 border-[#9BD8FF] rounded-lg mb-4 flex items-center justify-center">
            <HardDrive className="w-8 h-8 text-[#9BD8FF]" />
          </div>
          <h3 className="text-lg font-semibold text-[#9BD8FF] mb-1">Original Model</h3>
          <p className="text-2xl font-bold text-[#E6FBFF]">{comparisonData.original.modelSize} MB</p>
        </div>
        
        <div className="flex flex-col items-center">
          <ArrowRight className="w-8 h-8 text-[#00F3FF] mb-2" />
          <div className="px-3 py-1 bg-[#00FFA0]/20 text-[#00FFA0] text-sm rounded-full font-medium">
            -{comparisonData.improvements.sizeReduction}%
          </div>
        </div>
        
        <div className="text-center">
          <div className="w-20 h-12 bg-[#00FFA0]/20 border-2 border-[#00FFA0] rounded-lg mb-4 flex items-center justify-center">
            <HardDrive className="w-5 h-5 text-[#00FFA0]" />
          </div>
          <h3 className="text-lg font-semibold text-[#00FFA0] mb-1">Compressed Model</h3>
          <p className="text-2xl font-bold text-[#E6FBFF]">{comparisonData.compressed.modelSize} MB</p>
        </div>
      </div>
      
      <div className="mt-8 text-center">
        <p className="text-[#9BD8FF] mb-2">Storage Savings</p>
        <p className="text-3xl font-bold text-[#00FFA0]">
          {(comparisonData.original.modelSize - comparisonData.compressed.modelSize).toFixed(1)} MB saved
        </p>
      </div>
    </div>
    );
  };

  if (loading && !comparisonData) {
    return (
      <div className="space-y-8">
        <div className="text-center">
          <div className="mx-auto w-16 h-16 text-[#00F3FF] animate-spin">
            <Activity className="w-full h-full" />
          </div>
          <h2 className="text-2xl font-bold text-[#E6FBFF] mt-4">Loading Comparison...</h2>
        </div>
      </div>
    );
  }

  if (!comparisonData) {
    return (
      <div className="space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Model Comparison Results
          </h1>
          <p className="text-lg text-[#9BD8FF]">
            No comparison data available. Please train, compress, and evaluate both models first.
          </p>
          <button onClick={loadComparison} className="px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105">
            Load Comparison
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Celebration Effect */}
      {showCelebration && (
        <div className="fixed inset-0 pointer-events-none z-50">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-[#00FFA0] rounded-full animate-ping"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 2}s`,
                animationDuration: `${1 + Math.random()}s`,
              }}
            ></div>
          ))}
        </div>
      )}

      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Model Results
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Comprehensive analysis of model performance
        </p>
        {comparisonData && (
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium truncate max-w-md">
              {viewMode === 'comparison' ? 'Original vs Compressed Model' : viewMode === 'original' ? 'Original Model' : 'Compressed Model'}
            </span>
          </div>
        )}
      </div>

      {/* View Mode Toggle */}
      <div className="flex justify-center">
        <div className="flex space-x-2 bg-[#0b1220]/50 p-2 rounded-lg border border-[#122033]">
          <button
            onClick={() => setViewMode('comparison')}
            className={`px-6 py-2 rounded-lg font-semibold transition-all duration-200 ${
              viewMode === 'comparison'
                ? 'bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] text-white shadow-lg'
                : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
            }`}
          >
            Compare Both
          </button>
          <button
            onClick={() => setViewMode('original')}
            className={`px-6 py-2 rounded-lg font-semibold transition-all duration-200 ${
              viewMode === 'original'
                ? 'bg-gradient-to-r from-[#00F3FF] to-[#0088FF] text-white shadow-lg'
                : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
            }`}
          >
            Original Only
          </button>
          <button
            onClick={() => setViewMode('compressed')}
            className={`px-6 py-2 rounded-lg font-semibold transition-all duration-200 ${
              viewMode === 'compressed'
                ? 'bg-gradient-to-r from-[#00FFA0] to-[#00F3FF] text-white shadow-lg'
                : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
            }`}
          >
            Compressed Only
          </button>
        </div>
      </div>

      {/* Hero Stats */}
      {renderHeroStats()}

      {/* Load from Backend */}
      <div className="flex justify-center">
        <button onClick={loadComparison} disabled={loading} className="px-6 py-3 bg-[#121628] border border-[#122033] rounded-lg font-semibold text-[#00F3FF] hover:border-[#00F3FF] hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] transition-all duration-300 hover:scale-105 disabled:opacity-50">
          {loading ? 'Loading…' : 'Refresh Comparison'}
        </button>
      </div>

      {/* Detailed Comparison Table */}
      {renderComparisonTable()}

      {/* Charts Section */}
      <div className="grid lg:grid-cols-2 gap-8">
        {renderBarChart()}
        {renderRadarChart()}
      </div>

      {/* Analysis Card */}
      {renderAnalysisCard()}

      {/* File Size Visualization */}
      {renderFileSizeVisualization()}

      {/* Action Buttons - STEP 7: Download Models */}
      <div className="flex flex-wrap gap-4 justify-center">
        <button 
          onClick={downloadOriginalModel}
          className="px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#0088FF] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center gap-2"
        >
          <Download className="w-5 h-5" />
          Download Original Model
        </button>
        
        <button 
          onClick={downloadCompressedModel}
          className="px-6 py-3 bg-gradient-to-r from-[#00FFA0] to-[#00D67F] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,255,160,0.3)] transition-all duration-300 hover:scale-105 flex items-center gap-2"
        >
          <FileDown className="w-5 h-5" />
          Download Compressed Model
        </button>
        
        <button className="px-6 py-3 bg-[#121628] border border-[#122033] rounded-lg font-semibold text-[#00F3FF] hover:border-[#00F3FF] hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] transition-all duration-300 hover:scale-105 flex items-center gap-2">
          <Download className="w-5 h-5" />
          Download Comparison Report
        </button>
        
        <button 
          onClick={() => {
            const event = new CustomEvent('navigate-to', { detail: 'upload' });
            window.dispatchEvent(event);
          }}
          className="px-6 py-3 bg-[#121628] border border-[#122033] rounded-lg font-semibold text-[#FF00D0] hover:border-[#FF00D0] hover:shadow-[0_0_20px_rgba(255,0,208,0.2)] transition-all duration-300 hover:scale-105 flex items-center gap-2"
        >
          <RotateCcw className="w-5 h-5" />
          Start New Training
        </button>
      </div>
    </div>
  );
};

export default Results;