import React, { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle, Info, X, AlertTriangle } from 'lucide-react';

export interface ToastProps {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message?: string;
  duration?: number;
  onClose: (id: string) => void;
  action?: {
    label: string;
    onClick: () => void;
  };
}

const Toast: React.FC<ToastProps> = ({
  id,
  type,
  title,
  message,
  duration = 5000,
  onClose,
  action
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);

  useEffect(() => {
    // Trigger entrance animation
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        handleClose();
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [duration]);

  const handleClose = () => {
    setIsLeaving(true);
    setTimeout(() => onClose(id), 300);
  };

  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-[#00FFA0]" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-[#FF3B6B]" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-[#FFB84D]" />;
      case 'info':
      default:
        return <Info className="w-5 h-5 text-[#00F3FF]" />;
    }
  };

  const getBorderColor = () => {
    switch (type) {
      case 'success':
        return 'border-[#00FFA0]';
      case 'error':
        return 'border-[#FF3B6B]';
      case 'warning':
        return 'border-[#FFB84D]';
      case 'info':
      default:
        return 'border-[#00F3FF]';
    }
  };

  const getGlowColor = () => {
    switch (type) {
      case 'success':
        return 'shadow-[0_0_20px_rgba(0,255,160,0.3)]';
      case 'error':
        return 'shadow-[0_0_20px_rgba(255,59,107,0.3)]';
      case 'warning':
        return 'shadow-[0_0_20px_rgba(255,184,77,0.3)]';
      case 'info':
      default:
        return 'shadow-[0_0_20px_rgba(0,243,255,0.3)]';
    }
  };

  return (
    <div
      className={`
        w-full max-w-sm transform transition-all duration-300 ease-out
        ${isVisible && !isLeaving 
          ? 'translate-y-0 opacity-100' 
          : 'translate-y-4 opacity-0'
        }
      `}
    >
      <div
        className={`
          bg-[#0b1220]/90 backdrop-blur-xl border rounded-2xl p-5
          ${getBorderColor()} ${getGlowColor()}
        `}
      >
        <div className="flex items-start gap-4">
          {getIcon()}
          <div className="flex-1 min-w-0">
            <h4 className="text-[#E6FBFF] font-semibold text-base">{title}</h4>
            {message && (
              <p className="text-[#9BD8FF] text-sm mt-1">{message}</p>
            )}
            {action && (
              <button
                onClick={action.onClick}
                className="text-[#00F3FF] text-sm font-semibold hover:text-[#66FBFF] transition-colors mt-2"
              >
                {action.label}
              </button>
            )}
          </div>
          <button
            onClick={handleClose}
            className="text-[#9BD8FF] hover:text-[#E6FBFF] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Toast;