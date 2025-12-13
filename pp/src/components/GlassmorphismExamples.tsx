import { Code, Copy, Check } from 'lucide-react';
import { useState } from 'react';

export function GlassmorphismExamples() {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const examples = [
    {
      title: 'Standard Glass Card',
      description: 'Most common glassmorphism style with medium blur and subtle border',
      className: 'backdrop-blur-xl bg-white/10 border border-white/20 shadow-2xl',
      code: 'backdrop-blur-xl bg-white/10 border border-white/20 shadow-2xl',
    },
    {
      title: 'Elevated Glass Card',
      description: 'Stronger glass effect with increased background opacity',
      className: 'backdrop-blur-2xl bg-white/15 border border-white/30 shadow-2xl',
      code: 'backdrop-blur-2xl bg-white/15 border border-white/30 shadow-2xl',
    },
    {
      title: 'Subtle Glass Panel',
      description: 'Very light glass effect for nested elements',
      className: 'backdrop-blur-lg bg-white/5 border border-white/10 shadow-xl',
      code: 'backdrop-blur-lg bg-white/5 border border-white/10 shadow-xl',
    },
    {
      title: 'Purple Tinted Glass',
      description: 'Glass with purple tint for branded elements',
      className: 'backdrop-blur-xl bg-purple-500/10 border border-purple-400/30 shadow-2xl shadow-purple-500/10',
      code: 'backdrop-blur-xl bg-purple-500/10 border border-purple-400/30 shadow-2xl shadow-purple-500/10',
    },
    {
      title: 'Dark Glass Panel',
      description: 'Darker glass effect for contrast',
      className: 'backdrop-blur-xl bg-black/30 border border-white/10 shadow-2xl',
      code: 'backdrop-blur-xl bg-black/30 border border-white/10 shadow-2xl',
    },
    {
      title: 'Gradient Glass',
      description: 'Glass with subtle gradient for depth',
      className: 'backdrop-blur-xl bg-gradient-to-br from-white/10 to-purple-500/10 border border-white/20 shadow-2xl',
      code: 'backdrop-blur-xl bg-gradient-to-br from-white/10 to-purple-500/10 border border-white/20 shadow-2xl',
    },
  ];

  const handleCopy = (code: string, index: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Design Principles */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Glassmorphism Design Principles</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-lg mb-3 text-purple-200 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              Transparency & Blur
            </h4>
            <p className="text-sm text-purple-300">
              Use backdrop-blur with semi-transparent backgrounds (white/5-15%) to create the frosted glass effect.
              Stronger blur (xl, 2xl) creates more pronounced glass.
            </p>
          </div>
          <div>
            <h4 className="text-lg mb-3 text-purple-200 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              Subtle Borders
            </h4>
            <p className="text-sm text-purple-300">
              Light borders (white/10-30%) define edges without being harsh. Use slightly stronger borders for elevated elements.
            </p>
          </div>
          <div>
            <h4 className="text-lg mb-3 text-purple-200 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              Layered Shadows
            </h4>
            <p className="text-sm text-purple-300">
              Combine shadow-xl or shadow-2xl with optional colored shadows (shadow-purple-500/10) for depth and floating effect.
            </p>
          </div>
          <div>
            <h4 className="text-lg mb-3 text-purple-200 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              Hierarchical Blur
            </h4>
            <p className="text-sm text-purple-300">
              Use stronger blur and opacity for primary cards, lighter for nested elements. This creates visual hierarchy.
            </p>
          </div>
        </div>
      </div>

      {/* Live Examples */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {examples.map((example, index) => (
          <div key={index} className="space-y-3">
            <div className={`${example.className} rounded-2xl p-8 min-h-[200px] flex flex-col justify-between`}>
              <div>
                <h4 className="text-xl mb-2 text-purple-100">{example.title}</h4>
                <p className="text-sm text-purple-300">{example.description}</p>
              </div>
              <div className="flex items-center gap-2 mt-4">
                <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                <div className="w-3 h-3 bg-purple-600 rounded-full"></div>
              </div>
            </div>
            <div className="backdrop-blur-lg bg-black/30 rounded-xl p-4 border border-white/10">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 mb-2">
                  <Code className="w-4 h-4 text-purple-400" />
                  <span className="text-xs text-purple-400">CSS Classes</span>
                </div>
                <button
                  onClick={() => handleCopy(example.code, index)}
                  className="p-1 hover:bg-white/10 rounded transition-colors"
                  title="Copy to clipboard"
                >
                  {copiedIndex === index ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4 text-purple-400" />
                  )}
                </button>
              </div>
              <code className="text-xs text-purple-300 break-all">{example.code}</code>
            </div>
          </div>
        ))}
      </div>

      {/* Shadow Guidelines */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Shadow & Depth Guidelines</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="backdrop-blur-lg bg-white/5 rounded-xl p-6 border border-white/10 shadow-lg">
            <h4 className="text-lg mb-3 text-purple-200">Subtle Depth</h4>
            <p className="text-sm text-purple-300 mb-3">For nested or secondary elements</p>
            <code className="text-xs text-purple-400">shadow-lg</code>
          </div>
          <div className="backdrop-blur-xl bg-white/10 rounded-xl p-6 border border-white/20 shadow-xl">
            <h4 className="text-lg mb-3 text-purple-200">Medium Elevation</h4>
            <p className="text-sm text-purple-300 mb-3">For cards and containers</p>
            <code className="text-xs text-purple-400">shadow-xl</code>
          </div>
          <div className="backdrop-blur-xl bg-white/10 rounded-xl p-6 border border-white/20 shadow-2xl shadow-purple-500/20">
            <h4 className="text-lg mb-3 text-purple-200">High Elevation</h4>
            <p className="text-sm text-purple-300 mb-3">For modals and prominent features</p>
            <code className="text-xs text-purple-400">shadow-2xl shadow-purple-500/20</code>
          </div>
        </div>
      </div>
    </div>
  );
}
