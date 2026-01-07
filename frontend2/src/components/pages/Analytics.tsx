import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Brain,
  Database,
  Zap,
  Clock,
  Target,
  HardDrive,
  Activity,
  Award,
  Layers,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  PieChart,
  LineChart,
} from 'lucide-react';
import { trainingService } from '../../api/services/trainingService';
import { compressionService } from '../../api/services/compressionService';
import { datasetService } from '../../api/services/datasetService';

interface AnalyticsData {
  totalModels: number;
  totalDatasets: number;
  avgAccuracy: number;
  avgCompressionRatio: number;
  totalTrainingTime: number;
  totalStorageSaved: number;
  trainingHistory: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    val_accuracy: number;
  }>;
  compressionResults: Array<{
    method: string;
    accuracy: number;
    size_reduction: number;
    compression_ratio: number;
  }>;
  modelType?: string;
  originalSize?: number;
  compressedSize?: number;
}

const Analytics: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | 'all'>('all');

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const [trainingLogs, compressionInfo, datasetList] = await Promise.all([
        trainingService.logs().catch(() => null),
        compressionService.info().catch(() => null),
        datasetService.list().catch(() => ({ files: [], count: 0 })),
      ]);

      const analyticsData: AnalyticsData = {
        totalModels: 0,
        totalDatasets: datasetList.files?.length || 0,
        avgAccuracy: 0,
        avgCompressionRatio: 0,
        totalTrainingTime: 0,
        totalStorageSaved: 0,
        trainingHistory: [],
        compressionResults: [],
      };

      // Process training data
      if (trainingLogs) {
        analyticsData.totalModels++;
        analyticsData.avgAccuracy = (trainingLogs.val_score || trainingLogs.train_score || 0) * 100;
        analyticsData.totalTrainingTime = trainingLogs.training_time || 0;
        analyticsData.trainingHistory = trainingLogs.history || [];
        analyticsData.modelType = trainingLogs.model_type;
        analyticsData.originalSize = trainingLogs.model_size_kb;
      }

      // Process compression data
      if (compressionInfo) {
        analyticsData.totalModels++;
        analyticsData.avgCompressionRatio = compressionInfo.compression_ratio || 0;
        analyticsData.compressedSize = compressionInfo.compressed_size_kb || compressionInfo.best_model?.size_kb;
        
        if (analyticsData.originalSize && analyticsData.compressedSize) {
          analyticsData.totalStorageSaved = analyticsData.originalSize - analyticsData.compressedSize;
        }

        // Build compression results from comparison report
        if (compressionInfo.comparison_report) {
          Object.entries(compressionInfo.comparison_report).forEach(([method, result]: [string, any]) => {
            if (result && typeof result === 'object') {
              analyticsData.compressionResults.push({
                method: method.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
                accuracy: (result.accuracy || 0) * 100,
                size_reduction: result.size_reduction_percent || 0,
                compression_ratio: result.compression_ratio || 0,
              });
            }
          });
        }
      }

      setData(analyticsData);
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const StatCard: React.FC<{
    icon: React.ReactNode;
    label: string;
    value: string | number;
    subValue?: string;
    trend?: 'up' | 'down' | 'neutral';
    trendValue?: string;
    color: string;
  }> = ({ icon, label, value, subValue, trend, trendValue, color }) => (
    <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5 hover:border-[#00F3FF]/30 transition-all">
      <div className="flex items-start justify-between">
        <div className={`p-3 rounded-xl ${color}`}>{icon}</div>
        {trend && (
          <div
            className={`flex items-center gap-1 text-xs ${
              trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-red-400' : 'text-[#9BD8FF]/60'
            }`}
          >
            {trend === 'up' ? (
              <ArrowUpRight className="w-3 h-3" />
            ) : trend === 'down' ? (
              <ArrowDownRight className="w-3 h-3" />
            ) : (
              <Minus className="w-3 h-3" />
            )}
            {trendValue}
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className="text-3xl font-bold text-white">{value}</p>
        <p className="text-sm text-[#9BD8FF]/60 mt-1">{label}</p>
        {subValue && <p className="text-xs text-[#9BD8FF]/40 mt-0.5">{subValue}</p>}
      </div>
    </div>
  );

  const SimpleBarChart: React.FC<{
    data: Array<{ label: string; value: number; color: string }>;
    maxValue: number;
  }> = ({ data, maxValue }) => (
    <div className="space-y-3">
      {data.map((item, index) => (
        <div key={index}>
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-[#9BD8FF]/80">{item.label}</span>
            <span className="text-white font-medium">{item.value.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-[#1e3a5f]/50 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full ${item.color}`}
              style={{ width: `${Math.min((item.value / maxValue) * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#00F3FF] mx-auto mb-4"></div>
          <p className="text-[#9BD8FF]/60">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-20 bg-[#0b1220]/50 border border-[#1e3a5f] rounded-xl">
        <BarChart3 className="w-16 h-16 mx-auto text-[#9BD8FF]/30 mb-4" />
        <h3 className="text-xl font-semibold text-white mb-2">No Analytics Data</h3>
        <p className="text-[#9BD8FF]/60">Train and compress a model to see analytics</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Analytics Dashboard
          </h1>
          <p className="text-[#9BD8FF]/70 mt-1">Track your model performance and compression metrics</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex bg-[#0b1220] border border-[#1e3a5f] rounded-lg p-1">
            {(['7d', '30d', 'all'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1.5 rounded text-sm ${
                  timeRange === range
                    ? 'bg-[#1e3a5f] text-[#00F3FF]'
                    : 'text-[#9BD8FF]/60 hover:text-[#9BD8FF]'
                }`}
              >
                {range === '7d' ? '7 Days' : range === '30d' ? '30 Days' : 'All Time'}
              </button>
            ))}
          </div>
          <button
            onClick={loadAnalytics}
            className="p-2.5 bg-[#0f1a2e] border border-[#1e3a5f] rounded-lg text-[#9BD8FF] hover:bg-[#152238] hover:border-[#00F3FF]/50 transition-all"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Brain className="w-6 h-6 text-cyan-400" />}
          label="Total Models"
          value={data.totalModels}
          subValue={data.modelType ? `Latest: ${data.modelType.toUpperCase()}` : undefined}
          color="bg-cyan-500/20"
        />
        <StatCard
          icon={<Database className="w-6 h-6 text-purple-400" />}
          label="Total Datasets"
          value={data.totalDatasets}
          color="bg-purple-500/20"
        />
        <StatCard
          icon={<Target className="w-6 h-6 text-emerald-400" />}
          label="Best Accuracy"
          value={`${data.avgAccuracy.toFixed(1)}%`}
          trend={data.avgAccuracy >= 90 ? 'up' : data.avgAccuracy >= 70 ? 'neutral' : 'down'}
          trendValue={data.avgAccuracy >= 90 ? 'Excellent' : data.avgAccuracy >= 70 ? 'Good' : 'Needs work'}
          color="bg-emerald-500/20"
        />
        <StatCard
          icon={<Zap className="w-6 h-6 text-amber-400" />}
          label="Compression Ratio"
          value={data.avgCompressionRatio > 0 ? `${data.avgCompressionRatio.toFixed(1)}x` : 'N/A'}
          subValue={
            data.totalStorageSaved > 0 ? `${data.totalStorageSaved.toFixed(1)} KB saved` : undefined
          }
          color="bg-amber-500/20"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Progress Chart */}
        <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <LineChart className="w-5 h-5 text-[#00F3FF]" />
              <h3 className="text-lg font-semibold text-white">Training Progress</h3>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-[#00F3FF]"></div>
                <span className="text-[#9BD8FF]/60">Train Loss</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-[#FF00D0]"></div>
                <span className="text-[#9BD8FF]/60">Val Loss</span>
              </div>
            </div>
          </div>

          {data.trainingHistory.length > 0 ? (
            <div className="h-48 flex items-end gap-1">
              {data.trainingHistory.map((point, index) => {
                const maxLoss = Math.max(
                  ...data.trainingHistory.map((p) => Math.max(p.train_loss, p.val_loss))
                );
                return (
                  <div
                    key={index}
                    className="flex-1 flex flex-col gap-1 items-center"
                    title={`Epoch ${point.epoch}: Train ${point.train_loss.toFixed(4)}, Val ${point.val_loss.toFixed(4)}`}
                  >
                    <div className="w-full flex gap-0.5 items-end h-40">
                      <div
                        className="flex-1 bg-gradient-to-t from-[#00F3FF] to-[#00F3FF]/50 rounded-t"
                        style={{ height: `${(point.train_loss / maxLoss) * 100}%` }}
                      />
                      <div
                        className="flex-1 bg-gradient-to-t from-[#FF00D0] to-[#FF00D0]/50 rounded-t"
                        style={{ height: `${(point.val_loss / maxLoss) * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-[#9BD8FF]/40">{point.epoch}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center text-[#9BD8FF]/40">
              <p>No training history available</p>
            </div>
          )}
        </div>

        {/* Compression Results */}
        <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5">
          <div className="flex items-center gap-2 mb-5">
            <PieChart className="w-5 h-5 text-[#FF00D0]" />
            <h3 className="text-lg font-semibold text-white">Compression Methods</h3>
          </div>

          {data.compressionResults.length > 0 ? (
            <SimpleBarChart
              data={data.compressionResults.map((result, index) => ({
                label: result.method,
                value: result.accuracy,
                color:
                  index === 0
                    ? 'bg-gradient-to-r from-[#00F3FF] to-[#00F3FF]/70'
                    : index === 1
                    ? 'bg-gradient-to-r from-[#FF00D0] to-[#FF00D0]/70'
                    : 'bg-gradient-to-r from-amber-500 to-amber-500/70',
              }))}
              maxValue={100}
            />
          ) : (
            <div className="h-40 flex items-center justify-center text-[#9BD8FF]/40">
              <p>No compression data available</p>
            </div>
          )}
        </div>
      </div>

      {/* Size Comparison */}
      {data.originalSize && data.compressedSize && (
        <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5">
          <div className="flex items-center gap-2 mb-5">
            <HardDrive className="w-5 h-5 text-emerald-400" />
            <h3 className="text-lg font-semibold text-white">Size Comparison</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="text-center">
              <p className="text-[#9BD8FF]/60 text-sm mb-2">Original Model</p>
              <p className="text-4xl font-bold text-white">{data.originalSize.toFixed(1)}</p>
              <p className="text-[#9BD8FF]/40 text-sm">KB</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="relative w-full h-4 bg-[#1e3a5f]/50 rounded-full overflow-hidden">
                <div
                  className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-500 to-amber-500 rounded-full"
                  style={{ width: '100%' }}
                />
                <div
                  className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full"
                  style={{ width: `${(data.compressedSize / data.originalSize) * 100}%` }}
                />
              </div>
              <div className="mt-3 text-center">
                <p className="text-2xl font-bold text-emerald-400">
                  {((1 - data.compressedSize / data.originalSize) * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-[#9BD8FF]/60">Size Reduction</p>
              </div>
            </div>

            <div className="text-center">
              <p className="text-[#9BD8FF]/60 text-sm mb-2">Compressed Model</p>
              <p className="text-4xl font-bold text-emerald-400">{data.compressedSize.toFixed(1)}</p>
              <p className="text-[#9BD8FF]/40 text-sm">KB</p>
            </div>
          </div>
        </div>
      )}

      {/* Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Clock className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-[#9BD8FF]/60">Total Training Time</p>
              <p className="text-xl font-bold text-white">
                {data.totalTrainingTime > 60
                  ? `${(data.totalTrainingTime / 60).toFixed(1)} min`
                  : `${data.totalTrainingTime.toFixed(1)} sec`}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-emerald-500/20 rounded-lg">
              <Activity className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <p className="text-sm text-[#9BD8FF]/60">Training Epochs</p>
              <p className="text-xl font-bold text-white">{data.trainingHistory.length || 'N/A'}</p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Award className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <p className="text-sm text-[#9BD8FF]/60">Best Method</p>
              <p className="text-xl font-bold text-white">
                {data.compressionResults.length > 0
                  ? data.compressionResults.reduce((best, curr) =>
                      curr.accuracy > best.accuracy ? curr : best
                    ).method
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;

