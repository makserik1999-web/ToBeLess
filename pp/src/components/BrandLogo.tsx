import { Sparkles, Cpu, Zap } from 'lucide-react';

export function BrandLogo() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      {/* Logo Concept 1 - Neural Network */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <div className="flex flex-col items-center">
          <div className="relative w-32 h-32 mb-6">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-400 to-purple-600 rounded-2xl shadow-lg shadow-purple-500/50 flex items-center justify-center">
              <div className="relative">
                <div className="absolute inset-0 blur-xl bg-purple-300 opacity-50"></div>
                <Cpu className="w-16 h-16 text-white relative z-10" strokeWidth={1.5} />
              </div>
            </div>
          </div>
          <h3 className="text-xl mb-2 text-purple-100">Neural Core</h3>
          <p className="text-sm text-purple-300 text-center">Represents AI processing and intelligence</p>
        </div>
      </div>

      {/* Logo Concept 2 - AI Spark */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <div className="flex flex-col items-center">
          <div className="relative w-32 h-32 mb-6">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-purple-700 rounded-full shadow-lg shadow-purple-500/50 flex items-center justify-center">
              <div className="relative">
                <div className="absolute inset-0 blur-xl bg-purple-300 opacity-50"></div>
                <Sparkles className="w-16 h-16 text-white relative z-10" strokeWidth={1.5} />
              </div>
            </div>
          </div>
          <h3 className="text-xl mb-2 text-purple-100">Innovation Spark</h3>
          <p className="text-sm text-purple-300 text-center">Symbolizes creativity and breakthrough ideas</p>
        </div>
      </div>

      {/* Logo Concept 3 - Energy Flow */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <div className="flex flex-col items-center">
          <div className="relative w-32 h-32 mb-6">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-400 via-purple-600 to-purple-800 rounded-2xl rotate-45 shadow-lg shadow-purple-500/50 flex items-center justify-center">
              <div className="relative -rotate-45">
                <div className="absolute inset-0 blur-xl bg-purple-300 opacity-50"></div>
                <Zap className="w-16 h-16 text-white relative z-10" strokeWidth={1.5} />
              </div>
            </div>
          </div>
          <h3 className="text-xl mb-2 text-purple-100">Power Flow</h3>
          <p className="text-sm text-purple-300 text-center">Dynamic energy and rapid processing</p>
        </div>
      </div>

      {/* Text Logo */}
      <div className="md:col-span-3 backdrop-blur-xl bg-white/10 rounded-2xl p-12 border border-white/20 shadow-2xl">
        <div className="flex flex-col items-center">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-400 to-purple-600 rounded-xl shadow-lg shadow-purple-500/50 flex items-center justify-center">
              <Cpu className="w-8 h-8 text-white" strokeWidth={1.5} />
            </div>
            <h2 className="text-5xl bg-gradient-to-r from-purple-300 via-purple-200 to-purple-300 bg-clip-text text-transparent">
              Tobeles AI
            </h2>
          </div>
          <p className="text-purple-300">Recommended: Logo mark + wordmark combination</p>
        </div>
      </div>
    </div>
  );
}
