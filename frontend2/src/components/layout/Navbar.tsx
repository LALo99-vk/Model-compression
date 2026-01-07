import { useState, useEffect } from 'react';
import { Brain, User, BarChart3, Database, Box, LayoutDashboard } from 'lucide-react';
import { useBackendStatus } from '../../hooks/useBackendStatus';
import { useAppStore } from '../../store/useAppStore';

const Navbar = () => {
  const [currentHash, setCurrentHash] = useState(window.location.hash.replace('#', '') || 'dashboard');

  useEffect(() => {
    const handleHashChange = () => {
      setCurrentHash(window.location.hash.replace('#', '') || 'dashboard');
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navItems = [
    { name: 'Dashboard', href: '#dashboard', id: 'dashboard', icon: LayoutDashboard },
    { name: 'Models', href: '#models', id: 'models', icon: Box },
    { name: 'Datasets', href: '#datasets', id: 'datasets', icon: Database },
    { name: 'Analytics', href: '#analytics', id: 'analytics', icon: BarChart3 },
  ];

  useBackendStatus();
  const connected = useAppStore((s) => s.backendConnected);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-[70px] bg-[#0b1220]/80 backdrop-blur-lg border-b border-[#122033]/50">
      <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
        {/* Logo */}
        <a href="#dashboard" className="flex items-center space-x-3 hover:opacity-90 transition-opacity">
          <div className="relative">
            <Brain className="w-8 h-8 text-[#00F3FF]" />
            <div className="absolute inset-0 bg-[#00F3FF] blur-lg opacity-30 rounded-full"></div>
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            ML Forge
          </span>
        </a>

        {/* Navigation Links */}
        <div className="hidden md:flex items-center space-x-2">
          {navItems.map((item) => {
            const isActive = currentHash === item.id;
            const Icon = item.icon;
            return (
              <a
                key={item.name}
                href={item.href}
                className={`relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? 'text-[#00F3FF] bg-[#00F3FF]/10'
                    : 'text-[#9BD8FF] hover:text-[#00F3FF] hover:bg-[#00F3FF]/5'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.name}
                {isActive && (
                  <div className="absolute bottom-0 left-2 right-2 h-0.5 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-full"></div>
                )}
              </a>
            );
          })}
        </div>

        {/* Status & Profile */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="relative">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-[#00FFA0] animate-pulse' : 'bg-[#FF3B6B]'}`}></div>
              <div className={`absolute inset-0 blur-sm opacity-50 rounded-full ${connected ? 'bg-[#00FFA0]' : 'bg-[#FF3B6B]'}`}></div>
            </div>
            <span className="text-sm text-[#9BD8FF]">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <div className="w-8 h-8 bg-gradient-to-br from-[#00F3FF] to-[#FF00D0] rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;