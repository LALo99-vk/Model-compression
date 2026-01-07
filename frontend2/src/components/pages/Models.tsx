import React, { useState, useEffect } from 'react';
import {
  Brain,
  Network,
  GitBranch,
  Download,
  RefreshCw,
  Layers,
  Clock,
  HardDrive,
  Zap,
  Target,
  BarChart3,
  ChevronDown,
  Database,
  TrendingDown,
  Package,
  Cpu,
  Calendar,
} from 'lucide-react';
import { modelService, TrainingSession } from '../../api/services/modelService';

const Models: React.FC = () => {
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const response = await modelService.getTrainedModels();
      setSessions(response.sessions || []);
      // Auto-expand if only one session
      if (response.sessions?.length === 1) {
        setExpandedId(response.sessions[0].id);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  const getModelIcon = (type: string) => {
    switch (type) {
      case 'decision_tree':
        return <GitBranch className="w-5 h-5" />;
      case 'cnn':
        return <Network className="w-5 h-5" />;
      case 'rnn':
        return <Layers className="w-5 h-5" />;
      default:
        return <Brain className="w-5 h-5" />;
    }
  };

  const getModelTypeName = (type: string) => {
    switch (type) {
      case 'decision_tree':
        return 'Decision Tree';
      case 'cnn':
        return 'CNN';
      case 'rnn':
        return 'RNN / LSTM';
      default:
        return type;
    }
  };

  const getModelColor = (type: string) => {
    switch (type) {
      case 'decision_tree':
        return { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' };
      case 'cnn':
        return { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' };
      case 'rnn':
        return { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/30' };
      default:
        return { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' };
    }
  };

  const formatSize = (kb: number) => {
    if (!kb || kb === 0) return 'N/A';
    if (kb >= 1024) {
      return `${(kb / 1024).toFixed(2)} MB`;
    }
    return `${kb.toFixed(2)} KB`;
  };

  const formatParams = (params: number) => {
    if (!params || params === 0) return 'N/A';
    if (params >= 1000000) {
      return `${(params / 1000000).toFixed(2)}M`;
    }
    if (params >= 1000) {
      return `${(params / 1000).toFixed(1)}K`;
    }
    return params.toString();
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return '';
    }
  };

  const handleDownload = (type: 'original' | 'compressed') => {
    const endpoint = type === 'original' 
      ? '/api/model/download/original'
      : '/api/model/download/compressed';
    window.open(`http://localhost:8000${endpoint}`, '_blank');
  };

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#00F3FF] mx-auto mb-4"></div>
          <p className="text-[#9BD8FF]/60">Loading models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            Trained Models
          </h1>
          <p className="text-[#9BD8FF]/70 mt-1">
            View your trained and compressed models
          </p>
        </div>
        <button
          onClick={loadModels}
          className="flex items-center gap-2 px-4 py-2 bg-[#0f1a2e] border border-[#1e3a5f] rounded-lg text-[#9BD8FF] hover:bg-[#152238] hover:border-[#00F3FF]/50 transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* No Models State */}
      {sessions.length === 0 ? (
        <div className="text-center py-20 bg-[#0b1220]/50 border border-[#1e3a5f] rounded-xl">
          <Brain className="w-16 h-16 mx-auto text-[#9BD8FF]/30 mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">No Trained Models</h3>
          <p className="text-[#9BD8FF]/60 mb-6">
            Train a model to see it here
          </p>
          <button
            onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'training' }))}
            className="px-6 py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg text-white font-medium hover:opacity-90 transition-all"
          >
            Start Training
          </button>
        </div>
      ) : (
        /* Models List */
        <div className="space-y-4">
          {sessions.map((session) => {
            const colors = getModelColor(session.model_type);
            const isExpanded = expandedId === session.id;
            
            return (
              <div
                key={session.id}
                className="bg-gradient-to-br from-[#0f1a2e] to-[#0b1220] border border-[#1e3a5f] rounded-xl overflow-hidden"
              >
                {/* Collapsed Header - Model Type + Dataset */}
                <button
                  onClick={() => toggleExpand(session.id)}
                  className="w-full p-5 flex items-center justify-between hover:bg-[#1e3a5f]/20 transition-all"
                >
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-xl ${colors.bg} ${colors.text}`}>
                      {getModelIcon(session.model_type)}
                    </div>
                    <div className="text-left">
                      <h3 className="text-lg font-semibold text-white">
                        {getModelTypeName(session.model_type)}
                      </h3>
                      <div className="flex items-center gap-4 text-sm text-[#9BD8FF]/60">
                        <span className="flex items-center gap-1.5">
                          <Database className="w-4 h-4" />
                          {session.dataset_name}
                        </span>
                        {session.created_at && (
                          <span className="flex items-center gap-1.5">
                            <Calendar className="w-4 h-4" />
                            {formatDate(session.created_at)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    {session.compressed && (
                      <span className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-emerald-400 text-sm">
                        <Zap className="w-3.5 h-3.5" />
                        Compressed
                      </span>
                    )}
                    <ChevronDown className={`w-5 h-5 text-[#9BD8FF]/50 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                  </div>
                </button>

                {/* Expanded Content - Original + Compressed Details */}
                {isExpanded && (
                  <div className="px-5 pb-5 border-t border-[#1e3a5f]/50">
                    <div className="pt-5 grid md:grid-cols-2 gap-5">
                      
                      {/* Trained (Original) Model */}
                      <div className="bg-[#0b1220]/70 border border-[#1e3a5f] rounded-xl p-5">
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-2">
                            <Package className="w-5 h-5 text-[#00F3FF]" />
                            <h4 className="font-semibold text-white">Trained Model</h4>
                          </div>
                          <button
                            onClick={() => handleDownload('original')}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#00F3FF]/10 border border-[#00F3FF]/30 rounded-lg text-[#00F3FF] text-sm hover:bg-[#00F3FF]/20 transition-all"
                          >
                            <Download className="w-4 h-4" />
                            Download
                          </button>
                        </div>
                        
                        {session.original ? (
                          <div className="space-y-3">
                            <div className="flex justify-between py-2 border-b border-[#1e3a5f]/50">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <Target className="w-4 h-4" /> Accuracy
                              </span>
                              <span className="text-white font-medium">
                                {session.original.accuracy > 0 ? `${session.original.accuracy.toFixed(2)}%` : 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between py-2 border-b border-[#1e3a5f]/50">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <HardDrive className="w-4 h-4" /> Size
                              </span>
                              <span className="text-white font-medium">{formatSize(session.original.size_kb)}</span>
                            </div>
                            <div className="flex justify-between py-2 border-b border-[#1e3a5f]/50">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <Cpu className="w-4 h-4" /> Parameters
                              </span>
                              <span className="text-white font-medium">{formatParams(session.original.parameters)}</span>
                            </div>
                            <div className="flex justify-between py-2">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <Clock className="w-4 h-4" /> Training Time
                              </span>
                              <span className="text-white font-medium">{session.training_time.toFixed(1)}s</span>
                            </div>
                          </div>
                        ) : (
                          <p className="text-[#9BD8FF]/40 text-center py-6">No data available</p>
                        )}
                      </div>

                      {/* Compressed Model */}
                      <div className={`rounded-xl p-5 ${
                        session.compressed 
                          ? 'bg-emerald-500/5 border border-emerald-500/30' 
                          : 'bg-[#0b1220]/70 border border-[#1e3a5f]'
                      }`}>
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-2">
                            <Zap className={`w-5 h-5 ${session.compressed ? 'text-emerald-400' : 'text-[#9BD8FF]/40'}`} />
                            <h4 className="font-semibold text-white">Compressed Model</h4>
                          </div>
                          {session.compressed && (
                            <button
                              onClick={() => handleDownload('compressed')}
                              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400 text-sm hover:bg-emerald-500/20 transition-all"
                            >
                              <Download className="w-4 h-4" />
                              Download
                            </button>
                          )}
                        </div>
                        
                        {session.compressed ? (
                          <div className="space-y-3">
                            <div className="flex justify-between py-2 border-b border-emerald-500/20">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <Target className="w-4 h-4" /> Accuracy
                              </span>
                              <span className="text-white font-medium">
                                {session.compressed.accuracy > 0 ? `${session.compressed.accuracy.toFixed(2)}%` : 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between py-2 border-b border-emerald-500/20">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <HardDrive className="w-4 h-4" /> Size
                              </span>
                              <div className="flex items-center gap-2">
                                <span className="text-white font-medium">{formatSize(session.compressed.size_kb)}</span>
                                {session.compressed.size_reduction && session.compressed.size_reduction > 0 && (
                                  <span className="text-emerald-400 text-xs">
                                    -{session.compressed.size_reduction.toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="flex justify-between py-2 border-b border-emerald-500/20">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <Cpu className="w-4 h-4" /> Parameters
                              </span>
                              <span className="text-white font-medium">{formatParams(session.compressed.parameters)}</span>
                            </div>
                            <div className="flex justify-between py-2">
                              <span className="text-[#9BD8FF]/60 text-sm flex items-center gap-2">
                                <Zap className="w-4 h-4" /> Method
                              </span>
                              <span className="text-emerald-400 font-medium capitalize">
                                {session.compressed.method || 'Best'}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-center py-6">
                            <Zap className="w-10 h-10 text-[#9BD8FF]/20 mx-auto mb-3" />
                            <p className="text-[#9BD8FF]/40 text-sm mb-4">Not compressed yet</p>
                            <button
                              onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'compression' }))}
                              className="px-4 py-2 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg text-white text-sm font-medium hover:opacity-90 transition-all"
                            >
                              Compress Now
                            </button>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Compression Summary (if compressed) */}
                    {session.original && session.compressed && (
                      <div className="mt-5 p-4 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20 rounded-xl">
                        <div className="flex items-center gap-2 mb-3">
                          <TrendingDown className="w-5 h-5 text-emerald-400" />
                          <span className="font-semibold text-white">Compression Summary</span>
                        </div>
                        <div className="grid grid-cols-3 gap-4 text-center">
                          <div>
                            <p className="text-2xl font-bold text-emerald-400">
                              {session.compressed.size_reduction?.toFixed(1) || '0'}%
                            </p>
                            <p className="text-xs text-[#9BD8FF]/60">Size Reduced</p>
                          </div>
                          <div>
                            <p className="text-2xl font-bold text-[#00F3FF]">
                              {session.compressed.compression_ratio?.toFixed(1) || '1.0'}x
                            </p>
                            <p className="text-xs text-[#9BD8FF]/60">Compression Ratio</p>
                          </div>
                          <div>
                            <p className="text-2xl font-bold text-white">
                              {formatSize(session.original.size_kb - session.compressed.size_kb)}
                            </p>
                            <p className="text-xs text-[#9BD8FF]/60">Space Saved</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Quick Actions */}
                    <div className="mt-5 flex gap-3">
                      <button
                        onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'results' }))}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-[#0b1220] border border-[#1e3a5f] rounded-lg text-[#00F3FF] hover:border-[#00F3FF]/50 transition-all"
                      >
                        <BarChart3 className="w-4 h-4" />
                        View Results
                      </button>
                      {!session.compressed && (
                        <button
                          onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'compression' }))}
                          className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-[#00F3FF]/10 to-[#FF00D0]/10 border border-[#00F3FF]/30 rounded-lg text-[#00F3FF] hover:border-[#00F3FF] transition-all"
                        >
                          <Zap className="w-4 h-4" />
                          Compress
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
};

export default Models;
