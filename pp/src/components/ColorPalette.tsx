export function ColorPalette() {
  const colors = {
    primary: [
      { name: 'Purple 950', value: '#2e1065', hex: '#2e1065', usage: 'Dark backgrounds' },
      { name: 'Purple 900', value: '#581c87', hex: '#581c87', usage: 'Primary dark' },
      { name: 'Purple 800', value: '#6b21a8', hex: '#6b21a8', usage: 'Dark accents' },
      { name: 'Purple 700', value: '#7e22ce', hex: '#7e22ce', usage: 'Primary button' },
      { name: 'Purple 600', value: '#9333ea', hex: '#9333ea', usage: 'Primary brand' },
      { name: 'Purple 500', value: '#a855f7', hex: '#a855f7', usage: 'Hover states' },
      { name: 'Purple 400', value: '#c084fc', hex: '#c084fc', usage: 'Light accents' },
      { name: 'Purple 300', value: '#d8b4fe', hex: '#d8b4fe', usage: 'Subtle highlights' },
      { name: 'Purple 200', value: '#e9d5ff', hex: '#e9d5ff', usage: 'Very light accents' },
    ],
    neutral: [
      { name: 'Slate 950', value: '#020617', hex: '#020617', usage: 'Deep backgrounds' },
      { name: 'Slate 900', value: '#0f172a', hex: '#0f172a', usage: 'Dark backgrounds' },
      { name: 'Slate 800', value: '#1e293b', hex: '#1e293b', usage: 'Card backgrounds' },
      { name: 'Slate 700', value: '#334155', hex: '#334155', usage: 'Borders' },
      { name: 'Slate 400', value: '#94a3b8', hex: '#94a3b8', usage: 'Secondary text' },
      { name: 'Slate 200', value: '#e2e8f0', hex: '#e2e8f0', usage: 'Light text' },
      { name: 'White', value: '#ffffff', hex: '#ffffff', usage: 'Primary text' },
    ],
  };

  return (
    <div className="space-y-8">
      {/* Primary Purple Palette */}
      <div>
        <h3 className="text-xl mb-4 text-purple-200">Primary Purple Scale</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {colors.primary.map((color) => (
            <div key={color.name} className="backdrop-blur-xl bg-white/10 rounded-xl p-4 border border-white/20">
              <div
                className="w-full h-24 rounded-lg mb-3 shadow-lg"
                style={{ backgroundColor: color.value }}
              ></div>
              <h4 className="text-sm text-purple-100 mb-1">{color.name}</h4>
              <p className="text-xs text-purple-300 mb-2">{color.hex}</p>
              <p className="text-xs text-purple-400">{color.usage}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Neutral Palette */}
      <div>
        <h3 className="text-xl mb-4 text-purple-200">Neutral Scale</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {colors.neutral.map((color) => (
            <div key={color.name} className="backdrop-blur-xl bg-white/10 rounded-xl p-4 border border-white/20">
              <div
                className="w-full h-24 rounded-lg mb-3 shadow-lg border border-white/10"
                style={{ backgroundColor: color.value }}
              ></div>
              <h4 className="text-sm text-purple-100 mb-1">{color.name}</h4>
              <p className="text-xs text-purple-300 mb-2">{color.hex}</p>
              <p className="text-xs text-purple-400">{color.usage}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Glassmorphism Background Examples */}
      <div>
        <h3 className="text-xl mb-4 text-purple-200">Glassmorphism Backgrounds</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="backdrop-blur-xl bg-white/10 rounded-xl p-6 border border-white/20 shadow-2xl">
            <h4 className="text-purple-100 mb-2">Light Glass</h4>
            <code className="text-xs text-purple-300 block bg-black/20 p-3 rounded">
              backdrop-blur-xl bg-white/10 border border-white/20
            </code>
          </div>
          <div className="backdrop-blur-xl bg-white/5 rounded-xl p-6 border border-white/10 shadow-2xl">
            <h4 className="text-purple-100 mb-2">Subtle Glass</h4>
            <code className="text-xs text-purple-300 block bg-black/20 p-3 rounded">
              backdrop-blur-xl bg-white/5 border border-white/10
            </code>
          </div>
          <div className="backdrop-blur-lg bg-purple-900/30 rounded-xl p-6 border border-purple-500/30 shadow-2xl">
            <h4 className="text-purple-100 mb-2">Purple Tinted Glass</h4>
            <code className="text-xs text-purple-300 block bg-black/20 p-3 rounded">
              backdrop-blur-lg bg-purple-900/30 border border-purple-500/30
            </code>
          </div>
          <div className="backdrop-blur-2xl bg-gradient-to-br from-purple-900/20 to-purple-800/20 rounded-xl p-6 border border-purple-400/20 shadow-2xl">
            <h4 className="text-purple-100 mb-2">Gradient Glass</h4>
            <code className="text-xs text-purple-300 block bg-black/20 p-3 rounded">
              backdrop-blur-2xl bg-gradient-to-br from-purple-900/20 to-purple-800/20
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}
