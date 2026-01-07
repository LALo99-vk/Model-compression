import React, { useState, useCallback, useEffect } from 'react';
import { ToastProps } from './Toast';

interface ToastData extends Omit<ToastProps, 'onClose'> {
  id: string;
}

export interface ToastContextType {
  showToast: (toast: Omit<ToastData, 'id'>) => string;
  showSuccess: (title: string, message?: string) => string;
  showError: (title: string, message?: string) => string;
  showInfo: (title: string, message?: string) => string;
  showWarning: (title: string, message?: string) => string;
}

let toastCounter = 0;

const ToastContainer: React.FC = () => {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  const showToast = (toast: Omit<ToastData, 'id'>) => {
    // Generate unique ID using timestamp + counter + random
    const id = `${Date.now()}-${++toastCounter}-${Math.random().toString(36).substr(2, 9)}`;
    setToasts(prev => [...prev, { ...toast, id }]);
    // Auto-remove toast after 3 seconds (reduced for less intrusion)
    setTimeout(() => removeToast(id), 3000);
    return id;
  };

  const showSuccess = (title: string, message?: string) => {
    return showToast({ type: 'success', title, message });
  };

  const showError = (title: string, message?: string) => {
    return showToast({ type: 'error', title, message });
  };

  const showInfo = (title: string, message?: string) => {
    return showToast({ type: 'info', title, message });
  };

  const showWarning = (title: string, message?: string) => {
    return showToast({ type: 'warning', title, message });
  };

  // Export the functions to window for global access
  useEffect(() => {
    // @ts-ignore
    window.showToast = showToast;
    // @ts-ignore
    window.showSuccess = showSuccess;
    // @ts-ignore
    window.showError = showError;
    // @ts-ignore
    window.showInfo = showInfo;
    // @ts-ignore
    window.showWarning = showWarning;

    return () => {
      // Cleanup
      // @ts-ignore
      delete window.showToast;
      // @ts-ignore
      delete window.showSuccess;
      // @ts-ignore
      delete window.showError;
      // @ts-ignore
      delete window.showInfo;
      // @ts-ignore
      delete window.showWarning;
    };
  }, []);

  return (
    <div className="fixed top-4 right-4 z-[100] w-full max-w-sm">
      <div className="space-y-2">
        {toasts.map((toast) => (
          <div 
            key={toast.id} 
            className={`
              w-full
              animate-fade-in-down
              transition-all duration-300
            `}
          >
            <div className={`
              w-full max-w-sm bg-[#0b1220] 
              rounded-lg shadow-lg
              border-l-4 
              ${toast.type === 'success' ? 'border-[#00FFA0]' : 
                toast.type === 'error' ? 'border-[#FF3B6B]' :
                toast.type === 'warning' ? 'border-[#FFB84D]' : 'border-[#00F3FF]'}
              overflow-hidden
            `}>
              <div className="p-3">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    {toast.type === 'success' && (
                      <svg className="h-5 w-5 text-[#00FFA0]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                    {toast.type === 'error' && (
                      <svg className="h-5 w-5 text-[#FF3B6B]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    )}
                    {toast.type === 'warning' && (
                      <svg className="h-5 w-5 text-[#FFB84D]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    )}
                    {toast.type === 'info' && (
                      <svg className="h-5 w-5 text-[#00F3FF]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    )}
                  </div>
                  <div className="ml-3 w-0 flex-1">
                    <p className="text-sm font-medium text-[#E6FBFF]">
                      {toast.title}
                    </p>
                    {toast.message && (
                      <p className="mt-0.5 text-xs text-[#9BD8FF]">
                        {toast.message}
                      </p>
                    )}
                  </div>
                  <div className="ml-3 flex-shrink-0 flex">
                    <button
                      onClick={() => removeToast(toast.id)}
                      className="inline-flex text-[#9BD8FF] hover:text-[#E6FBFF] focus:outline-none transition-colors"
                    >
                      <span className="sr-only">Close</span>
                      <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Create a context for toast functionality
export const ToastContext = React.createContext<ToastContextType | null>(null);

export const useToast = () => {
  const context = React.useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  }, []);

  const showToast = useCallback((toast: Omit<ToastData, 'id'>) => {
    // Generate unique ID using timestamp + counter + random
    const id = `${Date.now()}-${++toastCounter}-${Math.random().toString(36).substr(2, 9)}`;
    setToasts(prev => [...prev, { ...toast, id }]);
    // Auto-remove toast after 5 seconds
    setTimeout(() => removeToast(id), 5000);
    return id;
  }, [removeToast]);

  const showSuccess = useCallback((title: string, message?: string) => {
    return showToast({ type: 'success', title, message });
  }, [showToast]);

  const showError = useCallback((title: string, message?: string) => {
    return showToast({ type: 'error', title, message });
  }, [showToast]);

  const showInfo = useCallback((title: string, message?: string) => {
    return showToast({ type: 'info', title, message });
  }, [showToast]);

  const showWarning = useCallback((title: string, message?: string) => {
    return showToast({ type: 'warning', title, message });
  }, [showToast]);

  const contextValue: ToastContextType = {
    showToast,
    showSuccess,
    showError,
    showInfo,
    showWarning,
  };

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <div className="fixed top-4 right-4 z-[100] w-full max-w-md">
        <div className="space-y-3">
          {toasts.map((toast) => (
            <div 
              key={toast.id}
              className={`
                w-full
                animate-fade-in-down
                transition-all duration-300
              `}
            >
              <div className={`
                w-full max-w-md bg-white dark:bg-gray-800 
                rounded-lg shadow-lg
                border-l-4 
                ${toast.type === 'success' ? 'border-green-500' : 
                  toast.type === 'error' ? 'border-red-500' :
                  toast.type === 'warning' ? 'border-yellow-500' : 'border-blue-500'}
                overflow-hidden
              `}>
                <div className="p-4">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      {toast.type === 'success' && (
                        <svg className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                      {toast.type === 'error' && (
                        <svg className="h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      )}
                      {toast.type === 'warning' && (
                        <svg className="h-6 w-6 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                      )}
                      {toast.type === 'info' && (
                        <svg className="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                    </div>
                    <div className="ml-3 w-0 flex-1 pt-0.5">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {toast.title}
                      </p>
                      {toast.message && (
                        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                          {toast.message}
                        </p>
                      )}
                    </div>
                    <div className="ml-4 flex-shrink-0 flex">
                      <button
                        onClick={() => removeToast(toast.id)}
                        className="inline-flex text-gray-400 hover:text-gray-500 focus:outline-none"
                      >
                        <span className="sr-only">Close</span>
                        <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </ToastContext.Provider>
  );
};

export default ToastContainer;