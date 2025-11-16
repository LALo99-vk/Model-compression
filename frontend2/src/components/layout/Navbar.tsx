import React from 'react';
import { Brain, User, Zap } from 'lucide-react';

const Navbar = () => {
  const navItems = [
    { name: 'Dashboard', href: '#dashboard', active: true },
    { name: 'Models', href: '#models', active: false },
    { name: 'Datasets', href: '#datasets', active: false },
    { name: 'Analytics', href: '#analytics', active: false },
    { name: 'Documentation', href: '#docs', active: false },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-[70px] bg-[#0b1220]/80 backdrop-blur-lg border-b border-[#122033]/50">
      <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Brain className="w-8 h-8 text-[#00F3FF]" />
            <div className="absolute inset-0 bg-[#00F3FF] blur-lg opacity-30 rounded-full"></div>
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] bg-clip-text text-transparent">
            ML Forge
          </span>
        </div>

        {/* Navigation Links */}
        <div className="hidden md:flex items-center space-x-8">
          {navItems.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className={`relative px-3 py-2 text-sm font-medium transition-all duration-300 ${
                item.active
                  ? 'text-[#00F3FF]'
                  : 'text-[#9BD8FF] hover:text-[#00F3FF]'
              }`}
            >
              {item.name}
              {item.active && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-full"></div>
              )}
            </a>
          ))}
        </div>

        {/* Status & Profile */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="relative">
              <div className="w-2 h-2 bg-[#00FFA0] rounded-full animate-pulse"></div>
              <div className="absolute inset-0 bg-[#00FFA0] blur-sm opacity-50 rounded-full"></div>
            </div>
            <span className="text-sm text-[#9BD8FF]">Backend Connected</span>
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