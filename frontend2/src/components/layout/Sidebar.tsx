import React from 'react';
import { 
  Upload, 
  Brain, 
  Play, 
  BarChart3, 
  Zap, 
  GitCompare,
  CheckCircle,
  Circle,
  Clock
} from 'lucide-react';

const Sidebar = () => {
  const steps = [
    { id: 'upload', name: 'Upload Dataset', icon: Upload, status: 'completed' },
    { id: 'select', name: 'Select Model', icon: Brain, status: 'current' },
    { id: 'train', name: 'Train Model', icon: Play, status: 'pending' },
    { id: 'evaluate', name: 'Evaluate Model', icon: BarChart3, status: 'pending' },
    { id: 'compress', name: 'Compress Model', icon: Zap, status: 'pending' },
    { id: 'compare', name: 'Compare Results', icon: GitCompare, status: 'pending' },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-[#00FFA0]" />;
      case 'current':
        return (
          <div className="relative">
            <Circle className="w-5 h-5 text-[#00F3FF] animate-pulse" />
            <div className="absolute inset-1 bg-[#00F3FF] rounded-full opacity-20 animate-pulse"></div>
          </div>
        );
      default:
        return <Circle className="w-5 h-5 text-[#9BD8FF]/40" />;
    }
  };

  const stats = [
    { label: 'Total Datasets', value: '12', icon: Upload },
    { label: 'Models Trained', value: '8', icon: Brain },
    { label: 'Avg Compression', value: '68%', icon: Zap },
  ];

  return (
    <div className="fixed left-0 top-[70px] bottom-0 w-[280px] bg-[#0b1220] border-r border-[#122033]/50 p-6">
      {/* Workflow Steps */}
      <div className="mb-8">
        <h3 className="text-[#E6FBFF] font-semibold mb-4">Workflow Progress</h3>
        <div className="space-y-3">
          {steps.map((step, index) => {
            const IconComponent = step.icon;
            return (
              <div key={step.id} className="flex items-center space-x-3 relative">
                {index < steps.length - 1 && (
                  <div className="absolute left-2.5 top-8 w-0.5 h-6 bg-[#122033]"></div>
                )}
                {getStatusIcon(step.status)}
                <div className="flex items-center space-x-2">
                  <IconComponent className={`w-4 h-4 ${
                    step.status === 'completed' ? 'text-[#00FFA0]' :
                    step.status === 'current' ? 'text-[#00F3FF]' :
                    'text-[#9BD8FF]/40'
                  }`} />
                  <span className={`text-sm ${
                    step.status === 'completed' ? 'text-[#00FFA0]' :
                    step.status === 'current' ? 'text-[#00F3FF]' :
                    'text-[#9BD8FF]/60'
                  }`}>
                    {step.name}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick Stats */}
      <div>
        <h3 className="text-[#E6FBFF] font-semibold mb-4">Quick Stats</h3>
        <div className="space-y-3">
          {stats.map((stat) => {
            const IconComponent = stat.icon;
            return (
              <div key={stat.label} className="bg-[#121628]/50 rounded-lg p-3 border border-[#122033]/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <IconComponent className="w-4 h-4 text-[#00F3FF]" />
                    <span className="text-[#9BD8FF] text-sm">{stat.label}</span>
                  </div>
                  <span className="text-[#E6FBFF] font-semibold">{stat.value}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;