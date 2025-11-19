import { useState } from 'react';
import { ToastProvider } from './components/ui/ToastContainer';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './components/pages/Dashboard';
import DatasetUpload from './components/pages/DatasetUpload';
import ModelSelection from './components/pages/ModelSelection';
import Training from './components/pages/Training';
import Evaluation from './components/pages/Evaluation';
import Compression from './components/pages/Compression';
import Comparison from './components/pages/Comparison';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'upload':
        return <DatasetUpload />;
      case 'select':
        return <ModelSelection />;
      case 'training':
        return <Training />;
      case 'evaluation':
        return <Evaluation />;
      case 'compression':
        return <Compression />;
      case 'comparison':
        return <Comparison />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <ToastProvider>
      <div className="min-h-screen bg-gradient-to-br from-[#050410] via-[#0b0820] to-[#0f172a] text-white">
        <Navbar />
        <div className="flex">
          <Sidebar />
          <main className="flex-1 ml-[280px] pt-[70px]">
            <div className="p-8">
              {/* Navigation Pills for Testing */}
              <div className="mb-8 flex justify-center">
                <div className="flex flex-wrap gap-2 bg-[#0b1220]/50 p-2 rounded-lg border border-[#122033]">
                  {[
                    { id: 'dashboard', name: 'Dashboard' },
                    { id: 'upload', name: 'Upload' },
                    { id: 'select', name: 'Select Model' },
                    { id: 'training', name: 'Training' },
                    { id: 'evaluation', name: 'Evaluation' },
                    { id: 'compression', name: 'Compression' },
                    { id: 'comparison', name: 'Comparison' }
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