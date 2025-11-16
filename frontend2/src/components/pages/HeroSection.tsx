import React from 'react';
import { ArrowRight, Zap, BarChart3, Cpu } from 'lucide-react';

const HeroSection = () => {
  const features = [
    { icon: Zap, text: '3 Model Types' },
    { icon: Cpu, text: '3 Compression Methods' },
    { icon: BarChart3, text: 'Real-time Analytics' },
  ];

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#050410] via-[#0b0820] to-[#0f172a]"></div>
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-20 w-64 h-64 bg-[#00F3FF] rounded-full blur-[120px] opacity-20"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-[#FF00D0] rounded-full blur-[150px] opacity-15"></div>
      </div>
      
      {/* Grid Pattern */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMTIyMDMzIiBzdHJva2Utd2lkdGg9IjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')] opacity-20"></div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-32 pb-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Column */}
          <div className="space-y-8">
            <div className="space-y-6">
              <h1 className="text-6xl lg:text-7xl font-bold leading-tight">
                <span className="bg-gradient-to-r from-[#00F3FF] via-[#66FBFF] to-[#FF00D0] bg-clip-text text-transparent">
                  Train, Compress, Deploy
                </span>
                <br />
                <span className="text-[#E6FBFF]">AI Models at</span>
                <br />
                <span className="bg-gradient-to-r from-[#FF00D0] to-[#00F3FF] bg-clip-text text-transparent">
                  Lightning Speed
                </span>
              </h1>
              
              <p className="text-xl text-[#9BD8FF] leading-relaxed max-w-lg">
                Complete ML workflow with automatic model compression. Reduce model size by 70% while maintaining 95%+ accuracy.
              </p>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <button className="group relative px-8 py-4 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg font-semibold text-white shadow-lg hover:shadow-[0_0_30px_rgba(0,243,255,0.3)] transition-all duration-300 hover:scale-105">
                <span className="relative z-10 flex items-center gap-2">
                  Start Training
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-[#00F3FF] to-[#FF00D0] rounded-lg blur opacity-0 group-hover:opacity-50 transition-opacity"></div>
              </button>
              
              <button className="px-8 py-4 border-2 border-[#00F3FF]/30 rounded-lg font-semibold text-[#00F3FF] hover:border-[#00F3FF] hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] transition-all duration-300 hover:scale-105">
                View Demo
              </button>
            </div>

            {/* Feature Pills */}
            <div className="flex flex-wrap gap-3">
              {features.map((feature, index) => {
                const IconComponent = feature.icon;
                return (
                  <div
                    key={index}
                    className="flex items-center gap-2 px-4 py-2 bg-[#0b1220]/50 border border-[#122033] rounded-full backdrop-blur-sm"
                  >
                    <IconComponent className="w-4 h-4 text-[#00F3FF]" />
                    <span className="text-sm text-[#9BD8FF] font-medium">{feature.text}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column - Animation */}
          <div className="relative">
            <div className="relative w-full h-96 flex items-center justify-center">
              {/* Neural Network Visualization */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative w-80 h-80">
                  {/* Nodes */}
                  <div className="absolute top-1/2 left-0 transform -translate-y-1/2">
                    <div className="w-4 h-4 bg-[#00F3FF] rounded-full shadow-[0_0_20px_rgba(0,243,255,0.5)] animate-pulse"></div>
                  </div>
                  <div className="absolute top-1/4 left-1/3 transform -translate-y-1/2">
                    <div className="w-4 h-4 bg-[#FF00D0] rounded-full shadow-[0_0_20px_rgba(255,0,208,0.5)] animate-pulse delay-100"></div>
                  </div>
                  <div className="absolute top-3/4 left-1/3 transform -translate-y-1/2">
                    <div className="w-4 h-4 bg-[#00FFA0] rounded-full shadow-[0_0_20px_rgba(0,255,160,0.5)] animate-pulse delay-200"></div>
                  </div>
                  <div className="absolute top-1/2 left-2/3 transform -translate-y-1/2">
                    <div className="w-4 h-4 bg-[#FFB84D] rounded-full shadow-[0_0_20px_rgba(255,184,77,0.5)] animate-pulse delay-300"></div>
                  </div>
                  <div className="absolute top-1/2 right-0 transform -translate-y-1/2">
                    <div className="w-4 h-4 bg-[#00F3FF] rounded-full shadow-[0_0_20px_rgba(0,243,255,0.5)] animate-pulse delay-500"></div>
                  </div>

                  {/* Connection Lines */}
                  <svg className="absolute inset-0 w-full h-full">
                    <line x1="16" y1="50%" x2="33%" y2="25%" stroke="url(#gradient1)" strokeWidth="2" opacity="0.6">
                      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" />
                    </line>
                    <line x1="16" y1="50%" x2="33%" y2="75%" stroke="url(#gradient2)" strokeWidth="2" opacity="0.6">
                      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" begin="0.5s" />
                    </line>
                    <line x1="33%" y1="25%" x2="66%" y2="50%" stroke="url(#gradient3)" strokeWidth="2" opacity="0.6">
                      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" begin="1s" />
                    </line>
                    <line x1="33%" y1="75%" x2="66%" y2="50%" stroke="url(#gradient4)" strokeWidth="2" opacity="0.6">
                      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" begin="1.5s" />
                    </line>
                    <line x1="66%" y1="50%" x2="calc(100% - 16px)" y2="50%" stroke="url(#gradient5)" strokeWidth="2" opacity="0.6">
                      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" begin="2s" />
                    </line>
                    
                    <defs>
                      <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#00F3FF" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#FF00D0" stopOpacity="0.4"/>
                      </linearGradient>
                      <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#00F3FF" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#00FFA0" stopOpacity="0.4"/>
                      </linearGradient>
                      <linearGradient id="gradient3" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#FF00D0" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#FFB84D" stopOpacity="0.4"/>
                      </linearGradient>
                      <linearGradient id="gradient4" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#00FFA0" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#FFB84D" stopOpacity="0.4"/>
                      </linearGradient>
                      <linearGradient id="gradient5" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#FFB84D" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#00F3FF" stopOpacity="0.4"/>
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
              </div>

              {/* Floating Particles */}
              <div className="absolute inset-0">
                {[...Array(20)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute w-1 h-1 bg-[#00F3FF] rounded-full opacity-20 animate-ping"
                    style={{
                      left: `${Math.random() * 100}%`,
                      top: `${Math.random() * 100}%`,
                      animationDelay: `${Math.random() * 2}s`,
                      animationDuration: `${2 + Math.random() * 2}s`,
                    }}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;