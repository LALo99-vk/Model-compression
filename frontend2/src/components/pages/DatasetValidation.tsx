import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  RefreshCw, 
  Wrench, 
  FileText,
  Loader,
  Download,
  Play
} from 'lucide-react';
import { validationService, ValidationResponse, ConditioningResponse } from '../../api/services/validationService';
import { useAppStore } from '../../store/useAppStore';
import { useToast } from '../ui/ToastContainer';
import { useTraining } from '../../hooks/useTraining';

const DatasetValidation = () => {
  const datasets = useAppStore((s) => s.datasets);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const selectedDatasetPath = useAppStore((s) => s.selectedDatasetPath);
  const selectedDatasetName = useAppStore((s) => s.selectedDatasetName);
  const setSelectedDataset = useAppStore((s) => s.setSelectedDataset);
  const { showSuccess, showError, showInfo } = useToast();
  const { start } = useTraining();
  
  const [selectedDataset, setSelectedDatasetLocal] = useState<string>('');
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isFixing, setIsFixing] = useState(false);
  const [fixResult, setFixResult] = useState<ConditioningResponse | null>(null);

  const modelType = selectedModel?.model_type || 'decision_tree';

  useEffect(() => {
    // Use the dataset that was selected during upload (from global state)
    if (selectedDatasetPath && !selectedDataset) {
      setSelectedDatasetLocal(selectedDatasetPath);
      showInfo('Dataset Loaded', `Using selected dataset: ${selectedDatasetName || 'dataset'}`);
    } else if (datasets.length > 0 && !selectedDataset) {
      // Fallback to first dataset if no dataset was selected
      setSelectedDatasetLocal(datasets[0].path);
    }
  }, [datasets, selectedDataset, selectedDatasetPath, selectedDatasetName]);

  // STEP 3: Auto-start validation when arriving from Model Selection
  useEffect(() => {
    if (selectedDataset && selectedModel && !validationResult && !isValidating) {
      // Auto-trigger validation after a short delay
      const timer = setTimeout(() => {
        showInfo('Auto-Validating', `Validating ${selectedDatasetName || 'dataset'} for ${selectedModel.model_type.toUpperCase()} model...`);
        handleValidate();
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [selectedDataset, selectedModel]);

  const handleValidate = async () => {
    if (!selectedDataset) {
      showError('No Dataset Selected', 'Please select a dataset first.');
      return;
    }

    setIsValidating(true);
    setValidationResult(null);
    setFixResult(null);

    try {
      const result = await validationService.validate({
        dataset_path: selectedDataset,
        model_type: modelType as 'decision_tree' | 'cnn' | 'rnn'
      });

      setValidationResult(result);

      if (result.status === 'valid') {
        showSuccess('Validation Passed', result.message);
        showInfo('Ready to Train', 'Click "Start Training" below to begin training your model.');
        // REMOVED: Auto-navigation to training - user will click "Start Training" button manually
      } else if (result.status === 'invalid') {
        showError('Validation Failed', result.message);
      } else {
        showError('Validation Error', result.message);
      }
    } catch (error: any) {
      showError('Validation Error', error.response?.data?.detail || error.message || 'Failed to validate dataset');
      setValidationResult({
        status: 'error',
        message: error.response?.data?.detail || error.message || 'Failed to validate dataset',
        issues: [],
        can_autofix: false
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleAutoFix = async () => {
    if (!selectedDataset) {
      showError('No Dataset Selected', 'Please select a dataset first.');
      return;
    }

    setIsFixing(true);
    setFixResult(null);

    try {
      const result = await validationService.condition({
        dataset_path: selectedDataset,
        model_type: modelType as 'decision_tree' | 'cnn' | 'rnn',
        auto_fix: true
      });

      setFixResult(result);

      if (result.status === 'fixed') {
        showSuccess('Dataset Fixed', result.message);
        // Re-validate after fixing to show "Start Training" button
        setTimeout(() => {
          handleValidate();
        }, 1000);
        // Also update validation result to show fixed status
        setValidationResult({
          status: 'valid',
          message: 'Dataset successfully fixed and validated. Ready for training.',
          issues: [],
          warnings: result.warnings || [],
          can_autofix: false
        });
      } else if (result.status === 'invalid_after_fix') {
        showError('Issues Remain', result.message);
        // Update validation result to show remaining issues
        setValidationResult({
          status: 'invalid',
          message: result.message,
          issues: result.issues || [],
          warnings: result.warnings || [],
          can_autofix: false
        });
      } else {
        showError('Fix Failed', result.message);
      }
    } catch (error: any) {
      showError('Auto-Fix Error', error.response?.data?.detail || error.message || 'Failed to fix dataset');
    } finally {
      setIsFixing(false);
    }
  };

  const handleFixManually = () => {
    showInfo('Manual Fix', 'Please download the validation report, fix the issues manually, and re-upload the dataset.');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'valid':
        return <CheckCircle className="w-6 h-6 text-[#00FFA0]" />;
      case 'invalid':
        return <XCircle className="w-6 h-6 text-[#FF3B6B]" />;
      case 'fixed':
        return <CheckCircle className="w-6 h-6 text-[#00FFA0]" />;
      default:
        return <AlertCircle className="w-6 h-6 text-[#FFAA00]" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
          Dataset Validation
        </h1>
        <p className="text-lg text-[#9BD8FF]">
          Validate your dataset format and structure before training
        </p>
        {selectedModel && (
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span className="px-3 py-1 bg-[#121628] border border-[#00F3FF]/60 rounded-full text-[#E6FBFF] font-medium">
              Model: {selectedModel.model_type === 'cnn' ? 'CNN' : selectedModel.model_type === 'rnn' ? 'RNN' : 'Decision Tree'}
            </span>
          </div>
        )}
      </div>

      {/* Dataset Selection */}
      <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
        <h2 className="text-xl font-semibold text-[#E6FBFF] mb-4">Select Dataset</h2>
        <div className="space-y-4">
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDatasetLocal(e.target.value)}
            className="w-full px-4 py-3 bg-[#0b1220] border border-[#122033] rounded-lg text-[#E6FBFF] focus:border-[#00F3FF] focus:outline-none"
            disabled={isValidating || isFixing}
          >
            <option value="">Select a dataset...</option>
            {datasets.map((dataset) => (
              <option key={dataset.path} value={dataset.path}>
                {dataset.filename} ({(dataset.size / 1024).toFixed(2)} KB)
              </option>
            ))}
          </select>
          
          <button
            onClick={handleValidate}
            disabled={!selectedDataset || isValidating || !selectedModel}
            className="w-full py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isValidating ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                Validating...
              </>
            ) : (
              <>
                <RefreshCw className="w-5 h-5" />
                Validate Dataset
              </>
            )}
          </button>
        </div>
      </div>

      {/* Detailed Validation Report */}
      {validationResult && validationResult.report_text && (
        <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <FileText className="w-6 h-6 text-[#00F3FF]" />
            <h2 className="text-xl font-semibold text-[#E6FBFF]">Detailed Validation Report</h2>
          </div>
          
          <div className="bg-[#0b1220] rounded-lg p-6 border border-[#122033]">
            <pre className="text-sm text-[#E6FBFF] whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto">
              {validationResult.report_text.split('\n').map((line, index) => {
                // Style different sections
                if (line.includes('DATASET VALIDATION REPORT')) {
                  return <div key={index} className="text-xl font-bold text-[#00F3FF] mb-2">{line}</div>;
                }
                if (line.includes('=')) {
                  return <div key={index} className="text-[#00F3FF] mb-2">{line}</div>;
                }
                if (line.includes('Model Selected:')) {
                  return <div key={index} className="text-[#E6FBFF] font-semibold mb-1">{line}</div>;
                }
                if (line.includes('Dataset Path:')) {
                  return <div key={index} className="text-[#E6FBFF] font-semibold mb-1">{line}</div>;
                }
                if (line.includes('Status:')) {
                  const statusMatch = line.match(/Status:\s*(.+)/);
                  const statusText = statusMatch ? statusMatch[1] : '';
                  const isInvalid = statusText.includes('INVALID');
                  const isValid = statusText.includes('VALID');
                  return (
                    <div key={index} className={`font-semibold mb-4 ${isValid ? 'text-[#00FFA0]' : isInvalid ? 'text-[#FF3B6B]' : 'text-[#E6FBFF]'}`}>
                      {line}
                    </div>
                  );
                }
                if (line.includes('📋 Dataset Information:')) {
                  return <div key={index} className="text-[#00F3FF] font-semibold mt-4 mb-2">{line}</div>;
                }
                if (line.includes('❌ Issues Found:')) {
                  return <div key={index} className="text-[#FF3B6B] font-semibold mt-4 mb-2">{line}</div>;
                }
                if (line.includes('⚠️  Warnings:')) {
                  return <div key={index} className="text-[#FFAA00] font-semibold mt-4 mb-2">{line}</div>;
                }
                if (line.includes('🔧 Suggested Fixes:')) {
                  return <div key={index} className="text-[#00F3FF] font-semibold mt-4 mb-2">{line}</div>;
                }
                if (line.includes('✅')) {
                  return <div key={index} className="text-[#00FFA0] font-semibold mt-4">{line}</div>;
                }
                if (line.includes('⚠️  Manual fixes')) {
                  return <div key={index} className="text-[#FFAA00] font-semibold mt-4">{line}</div>;
                }
                if (line.trim().startsWith('•')) {
                  return <div key={index} className="text-[#9BD8FF] ml-4">{line}</div>;
                }
                if (line.trim().match(/^\d+\./)) {
                  return <div key={index} className="text-[#E6FBFF] ml-4">{line}</div>;
                }
                if (line.trim() === '') {
                  return <div key={index} className="h-2"></div>;
                }
                return <div key={index} className="text-[#E6FBFF]">{line}</div>;
              })}
            </pre>
          </div>
        </div>
      )}

      {/* Validation Results */}
      {validationResult && (
        <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            {getStatusIcon(validationResult.status)}
            <h2 className="text-xl font-semibold text-[#E6FBFF]">Validation Summary</h2>
          </div>

          <div className={`p-4 rounded-lg mb-4 ${
            validationResult.status === 'valid' 
              ? 'bg-[#00FFA0]/10 border border-[#00FFA0]/30'
              : validationResult.status === 'invalid'
              ? 'bg-[#FF3B6B]/10 border border-[#FF3B6B]/30'
              : 'bg-[#FFAA00]/10 border border-[#FFAA00]/30'
          }`}>
            <p className={`font-medium ${
              validationResult.status === 'valid' 
                ? 'text-[#00FFA0]'
                : validationResult.status === 'invalid'
                ? 'text-[#FF3B6B]'
                : 'text-[#FFAA00]'
            }`}>
              {validationResult.message}
            </p>
          </div>

          {/* Issues */}
          {validationResult.issues && validationResult.issues.length > 0 && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-[#FF3B6B] mb-2">❌ Issues Found:</h3>
              <ul className="space-y-2">
                {validationResult.issues.map((issue, index) => (
                  <li key={index} className="flex items-start gap-2 text-[#E6FBFF]">
                    <span className="text-[#FF3B6B] mt-1">•</span>
                    <span>{issue}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {validationResult.warnings && validationResult.warnings.length > 0 && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-[#FFAA00] mb-2">⚠️ Warnings:</h3>
              <ul className="space-y-2">
                {validationResult.warnings.map((warning, index) => (
                  <li key={index} className="flex items-start gap-2 text-[#E6FBFF]">
                    <span className="text-[#FFAA00] mt-1">•</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Fix Suggestions */}
          {validationResult.fix_suggestions && validationResult.fix_suggestions.length > 0 && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-[#00F3FF] mb-2">🔧 Suggested Fixes:</h3>
              <ul className="space-y-2">
                {validationResult.fix_suggestions.map((suggestion, index) => (
                  <li key={index} className="flex items-start gap-2 text-[#E6FBFF]">
                    <span className="text-[#00F3FF] mt-1">•</span>
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action Buttons - ALWAYS show when invalid */}
          {validationResult.status === 'invalid' && (
            <div className="flex gap-4 mt-6">
              {validationResult.can_autofix && (
                <button
                  onClick={handleAutoFix}
                  disabled={isFixing}
                  className="flex-1 py-3 bg-gradient-to-r from-[#00F3FF] to-[#00FFA0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,255,160,0.3)] transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isFixing ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Fixing...
                    </>
                  ) : (
                    <>
                      <Wrench className="w-5 h-5" />
                      Auto-Fix Dataset
                    </>
                  )}
                </button>
              )}
              <button
                onClick={handleFixManually}
                className={`${validationResult.can_autofix ? 'flex-1' : 'w-full'} py-3 bg-[#121628] border-2 border-[#00F3FF] rounded-lg font-semibold text-[#00F3FF] hover:bg-[#00F3FF]/10 transition-all duration-300 flex items-center justify-center gap-2`}
              >
                <FileText className="w-5 h-5" />
                Fix Manually
              </button>
            </div>
          )}

          {/* Valid Status - Show Start Training Button */}
          {validationResult.status === 'valid' && (
            <div className="mt-6">
              <div className="p-4 bg-[#00FFA0]/10 border border-[#00FFA0]/30 rounded-lg mb-4">
                <p className="text-[#00FFA0] font-medium text-center">
                  ✅ Dataset is valid and ready for training!
                </p>
              </div>
              <button
                onClick={async () => {
                  // Find the dataset object
                  const dataset = datasets.find(d => d.path === selectedDataset);
                  if (dataset) {
                    // Set selected dataset in store
                    setSelectedDataset(dataset);
                    
                    // Navigate to training page
                    window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'training' }));
                    
                    // Dispatch event to auto-start training after navigation
                    setTimeout(() => {
                      window.dispatchEvent(new CustomEvent('auto-start-training', { 
                        detail: { 
                          dataset_path: selectedDataset,
                          epochs: 20,
                          batch_size: 32,
                          validation_split: 0.2
                        } 
                      }));
                    }, 300);
                  } else {
                    showError('Dataset Not Found', 'Selected dataset not found in the list.');
                  }
                }}
                className="w-full py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                Start Training
              </button>
            </div>
          )}
        </div>
      )}

      {/* Fix Results */}
      {fixResult && (
        <div className="bg-[#121628]/50 border border-[#122033] rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            {getStatusIcon(fixResult.status)}
            <h2 className="text-xl font-semibold text-[#E6FBFF]">Auto-Fix Results</h2>
          </div>

          <div className={`p-4 rounded-lg mb-4 ${
            fixResult.status === 'fixed' 
              ? 'bg-[#00FFA0]/10 border border-[#00FFA0]/30'
              : 'bg-[#FFAA00]/10 border border-[#FFAA00]/30'
          }`}>
            <p className={`font-medium ${
              fixResult.status === 'fixed' ? 'text-[#00FFA0]' : 'text-[#FFAA00]'
            }`}>
              {fixResult.message}
            </p>
          </div>

          {/* Changes Made */}
          {fixResult.changes_made && fixResult.changes_made.length > 0 && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-[#00F3FF] mb-2">✅ Changes Made:</h3>
              <ul className="space-y-2">
                {fixResult.changes_made.map((change, index) => (
                  <li key={index} className="flex items-start gap-2 text-[#E6FBFF]">
                    <span className="text-[#00FFA0] mt-1">•</span>
                    <span>{change}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Remaining Issues */}
          {fixResult.issues && fixResult.issues.length > 0 && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-[#FF3B6B] mb-2">⚠️ Remaining Issues:</h3>
              <ul className="space-y-2">
                {fixResult.issues.map((issue, index) => (
                  <li key={index} className="flex items-start gap-2 text-[#E6FBFF]">
                    <span className="text-[#FF3B6B] mt-1">•</span>
                    <span>{issue}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Backup Path */}
          {fixResult.backup_path && (
            <div className="mt-4 p-3 bg-[#0b1220] rounded-lg">
              <p className="text-sm text-[#9BD8FF]">
                <strong>Backup saved:</strong> {fixResult.backup_path}
              </p>
            </div>
          )}

          {/* Start Training Button after successful fix */}
          {fixResult.status === 'fixed' && (
            <div className="mt-6">
              <div className="p-4 bg-[#00FFA0]/10 border border-[#00FFA0]/30 rounded-lg mb-4">
                <p className="text-[#00FFA0] font-medium text-center">
                  ✅ Dataset is now valid and ready for training!
                </p>
              </div>
              <button
                onClick={async () => {
                  // Find the dataset object (use the fixed dataset path if available)
                  const datasetPath = fixResult?.new_path || selectedDataset;
                  const dataset = datasets.find(d => d.path === datasetPath);
                  
                  if (dataset) {
                    // Set selected dataset in store
                    setSelectedDataset(dataset);
                    
                    // Navigate to training page
                    window.dispatchEvent(new CustomEvent('navigate-to', { detail: 'training' }));
                    
                    // Dispatch event to auto-start training after navigation
                    setTimeout(() => {
                      window.dispatchEvent(new CustomEvent('auto-start-training', { 
                        detail: { 
                          dataset_path: datasetPath,
                          epochs: 20,
                          batch_size: 32,
                          validation_split: 0.2
                        } 
                      }));
                    }, 300);
                  } else {
                    showError('Dataset Not Found', 'Dataset not found in the list.');
                  }
                }}
                className="w-full py-3 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                Start Training
              </button>
            </div>
          )}

          {/* Remaining Issues - Show buttons again */}
          {fixResult.status === 'invalid_after_fix' && fixResult.issues && fixResult.issues.length > 0 && (
            <div className="mt-6 flex gap-4">
              <button
                onClick={handleFixManually}
                className="flex-1 py-3 bg-[#121628] border-2 border-[#00F3FF] rounded-lg font-semibold text-[#00F3FF] hover:bg-[#00F3FF]/10 transition-all duration-300 flex items-center justify-center gap-2"
              >
                <FileText className="w-5 h-5" />
                Fix Manually
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DatasetValidation;

