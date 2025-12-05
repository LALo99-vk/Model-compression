import React, { useState, useEffect } from 'react';
import { ToastProvider } from './components/ui/ToastContainer';
import Navbar from './components/layout/Navbar';
import Dashboard from './components/pages/Dashboard';
import DatasetUpload from './components/pages/DatasetUpload';
import ModelSelection from './components/pages/ModelSelection';
import Training from './components/pages/Training';
import Compression from './components/pages/Compression';
import Results from './components/pages/Results';
import DatasetValidation from './components/pages/DatasetValidation';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  // Listen for navigation events from child components
  useEffect(() => {
    const handleNavigate = (event: Event) => {
      const customEvent = event as CustomEvent<string>;
      setCurrentPage(customEvent.detail);
    };
    
    window.addEventListener('navigate-to', handleNavigate);
    
    return () => {
      window.removeEventListener('navigate-to', handleNavigate);
    };
  }, []);

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'upload':
        return <DatasetUpload />;
      case 'validation':
        return <DatasetValidation />;
      case 'select':
        return <ModelSelection />;
      case 'training':
        return <Training />;
      case 'compression':
        return <Compression />;
      case 'results':
      case 'comparison': // Keep old route for backward compatibility
        return <Results />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <ToastProvider>
      <div className="min-h-screen bg-gradient-to-br from-[#050410] via-[#0b0820] to-[#0f172a] text-white">
        <Navbar />
        <div className="flex">
          <main className="flex-1 pt-[70px]">
            <div className="p-8">
              {/* Navigation Pills for Testing */}
              <div className="mb-8 flex justify-center">
                <div className="flex flex-wrap gap-2 bg-[#0b1220]/50 p-2 rounded-lg border border-[#122033]">
                  {[
                    { id: 'dashboard', name: 'Dashboard' },
                    { id: 'upload', name: 'Upload' },
                    { id: 'select', name: 'Select Model' },
                    { id: 'validation', name: 'Dataset Validation' },
                    { id: 'training', name: 'Training' },
                    { id: 'compression', name: 'Compression' },
                    { id: 'results', name: 'Results' }
                  ].map((page) => (
                    <button
                      key={page.id}
                      onClick={() => setCurrentPage(page.id)}
                      className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                        currentPage === page.id
                          ? 'bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] text-white shadow-lg'
                          : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
                      }`}
                    >
                      {page.name}
                    </button>
                  ))}
                </div>
              </div>
              
              {renderCurrentPage()}
            </div>
          </main>
        </div>
      </div>
    </ToastProvider>
  );
}

export default App;