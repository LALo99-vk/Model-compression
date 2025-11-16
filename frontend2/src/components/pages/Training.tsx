import React, { useState, useEffect } from 'react';
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
import { TrainingStatus } from '../../types';

const Training = () => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    status: 'idle',
    currentEpoch: 0,
    totalEpochs: 20,
    progress: 0,
    estimatedTimeRemaining: '0m 0s',
    metrics: {
      trainingLoss: 0,
      validationLoss: 0,
      validationAccuracy: 0
    },
    logs: []
  });
  
  const [config, setConfig] = useState({
    epochs: 20,
    batchSize: 32,
    validationSplit: 0.2,
    datasetPath: 'customer_data.csv'
  });

  const [showLogs, setShowLogs] = useState(false);
  const [chartData, setChartData] = useState<Array<{
    epoch: number;
    trainingLoss: number;
    validationLoss: number;
    accuracy: number;
  }>>([]);

  useEffect(() => {
    if (trainingStatus.status === 'training') {
      const interval = setInterval(() => {
        setTrainingStatus(prev => {
          const newEpoch = Math.min(prev.currentEpoch + 1, prev.totalEpochs);
          const newProgress = (newEpoch / prev.totalEpochs) * 100;
          const isCompleted = newEpoch >= prev.totalEpochs;
          
          const newMetrics = {
            trainingLoss: Math.max(0.1, 1.5 - (newEpoch * 0.05) + Math.random() * 0.1),
            validationLoss: Math.max(0.15, 1.7 - (newEpoch * 0.045) + Math.random() * 0.1),
            validationAccuracy: Math.min(0.95, 0.3 + (newEpoch * 0.025) + Math.random() * 0.02)
          };

          // Add to chart data
          setChartData(prevChart => [...prevChart, {
            epoch: newEpoch,
            trainingLoss: newMetrics.trainingLoss,
            validationLoss: newMetrics.validationLoss,
            accuracy: newMetrics.validationAccuracy
          }]);

          const newLog = `Epoch ${newEpoch}/${prev.totalEpochs} - loss: ${newMetrics.trainingLoss.toFixed(4)} - val_loss: ${newMetrics.validationLoss.toFixed(4)} - val_accuracy: ${newMetrics.validationAccuracy.toFixed(4)}`;
          
          return {
            ...prev,
            currentEpoch: newEpoch,
            progress: newProgress,
            status: isCompleted ? 'completed' : 'training',
            estimatedTimeRemaining: isCompleted ? '0m 0s' : `${Math.ceil((prev.totalEpochs - newEpoch) * 0.5)}m ${Math.floor(Math.random() * 60)}s`,
            metrics: newMetrics,
            logs: [...prev.logs, `${new Date().toLocaleTimeString()} - ${newLog}`].slice(-10)
          };
        });
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [trainingStatus.status]);

  const startTraining = () => {
    setTrainingStatus(prev => ({
      ...prev,
      status: 'training',
      currentEpoch: 0,
      progress: 0,
      logs: [`${new Date().toLocaleTimeString()} - Starting training with ${config.epochs} epochs...`]
    }));
    setChartData([]);
  };

  const stopTraining = () => {
    setTrainingStatus(prev => ({
      ...prev,
      status: 'idle'
    }));
  };

  const renderTrainingChart = () => {
    if (chartData.length === 0) return null;

    const maxLoss = Math.max(...chartData.map(d => Math.max(d.trainingLoss, d.validationLoss)));
    const minLoss = Math.min(...chartData.map(d => Math.min(d.trainingLoss, d.validationLoss)));

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
              points={chartData.map((d, i) => 
                `${(i / (chartData.length - 1)) * 380 + 10},${190 - ((d.trainingLoss - minLoss) / (maxLoss - minLoss)) * 170}`
              ).join(' ')}
            />
            
            {/* Validation Loss Line */}
            <polyline
              fill="none"
              stroke="#FF00D0"
              strokeWidth="2"
              points={chartData.map((d, i) => 
                `${(i / (chartData.length - 1)) * 380 + 10},${190 - ((d.validationLoss - minLoss) / (maxLoss - minLoss)) * 170}`
              ).join(' ')}
            />

            {/* Accuracy Line (scaled) */}
            <polyline
              fill="none"
              stroke="#00FFA0"
              strokeWidth="2"
              points={chartData.map((d, i) => 
                `${(i / (chartData.length - 1)) * 380 + 10},${190 - d.accuracy * 170}`
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
            <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-sm text-[#00F3FF] font-medium">
              CNN Model
            </span>
            <span className="px-3 py-1 bg-[#121628] border border-[#122033] rounded-full text-sm text-[#FF00D0] font-medium">
              customer_data.csv
            </span>
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
              disabled={trainingStatus.status === 'training'}
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-[#9BD8FF]">Batch Size</label>
            <select
              value={config.batchSize}
              onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
              className="w-full px-3 py-2 bg-[#0b1220] border border-[#122033] rounded-lg text-[#E6FBFF] focus:border-[#00F3FF] focus:outline-none"
              disabled={trainingStatus.status === 'training'}
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
              disabled={trainingStatus.status === 'training'}
            />
            <div className="text-xs text-[#9BD8FF] text-center">{config.validationSplit}</div>
          </div>
          <div className="flex items-end">
            {trainingStatus.status === 'idle' ? (
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
        {trainingStatus.status === 'idle' && (
          <div className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 text-[#9BD8FF]/50">
              <Play className="w-full h-full" />
            </div>
            <h3 className="text-xl font-semibold text-[#E6FBFF]">Ready to Train</h3>
            <p className="text-[#9BD8FF]">Configure your parameters and start training</p>
          </div>
        )}

        {trainingStatus.status === 'training' && (
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
                    <span>{trainingStatus.estimatedTimeRemaining} remaining</span>
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-[#9BD8FF]">Epoch {trainingStatus.currentEpoch} / {trainingStatus.totalEpochs}</span>
                  <span className="text-[#E6FBFF] font-semibold">{Math.round(trainingStatus.progress)}%</span>
                </div>
                <div className="bg-[#0b1220] rounded-full h-3 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] h-full transition-all duration-500 relative"
                    style={{ width: `${trainingStatus.progress}%` }}
                  >
                    <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                  </div>
                </div>
              </div>

              {/* Live Metrics */}
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="bg-[#0b1220] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#00F3FF] rounded-full"></div>
                    <span className="text-sm text-[#9BD8FF]">Training Loss</span>
                  </div>
                  <div className="text-xl font-bold text-[#E6FBFF]">
                    {trainingStatus.metrics.trainingLoss.toFixed(4)}
                  </div>
                </div>
                <div className="bg-[#0b1220] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#FF00D0] rounded-full"></div>
                    <span className="text-sm text-[#9BD8FF]">Val Loss</span>
                  </div>
                  <div className="text-xl font-bold text-[#E6FBFF]">
                    {trainingStatus.metrics.validationLoss.toFixed(4)}
                  </div>
                </div>
                <div className="bg-[#0b1220] rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-[#00FFA0] rounded-full"></div>
                    <span className="text-sm text-[#9BD8FF]">Val Accuracy</span>
                  </div>
                  <div className="text-xl font-bold text-[#E6FBFF]">
                    {(trainingStatus.metrics.validationAccuracy * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {trainingStatus.status === 'completed' && (
          <div className="text-center space-y-6">
            <div className="mx-auto w-16 h-16 text-[#00FFA0] relative">
              <CheckCircle className="w-full h-full" />
              <div className="absolute inset-0 bg-[#00FFA0] blur-xl opacity-30 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h3 className="text-2xl font-bold text-[#00FFA0] mb-2">Training Completed!</h3>
              <p className="text-[#9BD8FF]">Model training finished successfully</p>
            </div>
            <div className="grid grid-cols-2 gap-4 max-w-md mx-auto text-sm">
              <div>
                <span className="text-[#9BD8FF]">Final Accuracy:</span>
                <div className="text-xl font-bold text-[#E6FBFF]">
                  {(trainingStatus.metrics.validationAccuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <span className="text-[#9BD8FF]">Training Time:</span>
                <div className="text-xl font-bold text-[#E6FBFF]">
                  {Math.floor(trainingStatus.currentEpoch * 2 / 60)}m {(trainingStatus.currentEpoch * 2) % 60}s
                </div>
              </div>
            </div>
            <button className="px-8 py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105">
              View Results
            </button>
          </div>
        )}
      </div>

      {/* Training Graph */}
      {chartData.length > 0 && renderTrainingChart()}

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
              {trainingStatus.logs.length} entries
            </span>
          </div>
          {showLogs ? <ChevronUp className="w-5 h-5 text-[#9BD8FF]" /> : <ChevronDown className="w-5 h-5 text-[#9BD8FF]" />}
        </button>
        
        {showLogs && (
          <div className="p-4 border-t border-[#122033]">
            <div className="bg-[#0b0820] rounded-lg p-4 font-mono text-sm max-h-64 overflow-y-auto space-y-1">
              {trainingStatus.logs.length > 0 ? (
                trainingStatus.logs.map((log, index) => (
                  <div key={index} className="text-[#9BD8FF]">
                    {log}
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