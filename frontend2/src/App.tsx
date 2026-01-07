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
import Models from './components/pages/Models';
import Datasets from './components/pages/Datasets';
import Analytics from './components/pages/Analytics';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  // Handle hash-based navigation
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '') || 'dashboard';
      setCurrentPage(hash);
    };

    // Initial load
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);

    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, []);

  // Listen for navigation events from child components
  useEffect(() => {
    const handleNavigate = (event: Event) => {
      const customEvent = event as CustomEvent<string>;
      const page = customEvent.detail;
      window.location.hash = page;
      setCurrentPage(page);
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
      case 'comparison':
        return <Results />;
      case 'models':
        return <Models />;
      case 'datasets':
        return <Datasets />;
      case 'analytics':
        return <Analytics />;
      default:
        return <Dashboard />;
    }
  };

  // Workflow pages (main pipeline)
  const workflowPages = [
    { id: 'dashboard', name: 'Dashboard' },
    { id: 'upload', name: 'Upload' },
    { id: 'select', name: 'Model' },
    { id: 'validation', name: 'Validate' },
    { id: 'training', name: 'Train' },
    { id: 'compression', name: 'Compress' },
    { id: 'results', name: 'Results' }
  ];

  return (
    <ToastProvider>
      <div className="min-h-screen bg-gradient-to-br from-[#050410] via-[#0b0820] to-[#0f172a] text-white">
        <Navbar />
        <div className="flex">
          <main className="flex-1 pt-[70px]">
            <div className="p-8">
              {/* Workflow Navigation Pills */}
              <div className="mb-8 flex justify-center">
                <div className="flex flex-wrap gap-2 bg-[#0b1220]/50 p-2 rounded-lg border border-[#122033]">
                  {workflowPages.map((page, index) => (
                    <React.Fragment key={page.id}>
                      <button
                        onClick={() => {
                          window.location.hash = page.id;
                          setCurrentPage(page.id);
                        }}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                          currentPage === page.id
                            ? 'bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] text-white shadow-lg'
                            : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#121628]'
                        }`}
                      >
                        {page.name}
                      </button>
                      {index < workflowPages.length - 1 && (
                        <span className="self-center text-[#1e3a5f]">→</span>
                      )}
                    </React.Fragment>
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