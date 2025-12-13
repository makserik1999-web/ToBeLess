export function Typography() {
  return (
    <div className="space-y-6">
      {/* Font Recommendations */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Recommended Fonts</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-xl mb-3 text-purple-200">Primary: Inter</h4>
            <p className="text-purple-300 mb-2">
              Modern, clean, and highly readable geometric sans-serif. Perfect for UI elements, body text, and headings.
            </p>
            <p className="text-sm text-purple-400">
              Use for: Headings, buttons, navigation, body text
            </p>
          </div>
          <div>
            <h4 className="text-xl mb-3 text-purple-200">Alternative: Space Grotesk</h4>
            <p className="text-purple-300 mb-2">
              Techy, futuristic feel with distinctive character. Great for logos and headlines.
            </p>
            <p className="text-sm text-purple-400">
              Use for: Logo, hero headings, emphasis
            </p>
          </div>
          <div>
            <h4 className="text-xl mb-3 text-purple-200">Monospace: JetBrains Mono</h4>
            <p className="text-purple-300 mb-2">
              Clean monospace font perfect for code snippets and technical data display.
            </p>
            <p className="text-sm text-purple-400">
              Use for: Code, data, technical specifications
            </p>
          </div>
          <div>
            <h4 className="text-xl mb-3 text-purple-200">Accent: Outfit</h4>
            <p className="text-purple-300 mb-2">
              Rounded, friendly geometric sans-serif. Adds warmth while maintaining modern aesthetic.
            </p>
            <p className="text-sm text-purple-400">
              Use for: Marketing copy, callouts, friendly messaging
            </p>
          </div>
        </div>
      </div>

      {/* Typography Scale */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Typography Scale</h3>
        <div className="space-y-4">
          <div className="pb-4 border-b border-white/10">
            <h1 className="text-6xl text-purple-100 mb-2">Display Large</h1>
            <p className="text-sm text-purple-400">text-6xl - Hero sections, main headlines</p>
          </div>
          <div className="pb-4 border-b border-white/10">
            <h2 className="text-4xl text-purple-100 mb-2">Heading 1</h2>
            <p className="text-sm text-purple-400">text-4xl - Page titles, section headers</p>
          </div>
          <div className="pb-4 border-b border-white/10">
            <h3 className="text-2xl text-purple-100 mb-2">Heading 2</h3>
            <p className="text-sm text-purple-400">text-2xl - Subsections, card titles</p>
          </div>
          <div className="pb-4 border-b border-white/10">
            <h4 className="text-xl text-purple-100 mb-2">Heading 3</h4>
            <p className="text-sm text-purple-400">text-xl - Component headings</p>
          </div>
          <div className="pb-4 border-b border-white/10">
            <p className="text-lg text-purple-200 mb-2">Body Large</p>
            <p className="text-sm text-purple-400">text-lg - Lead paragraphs, important content</p>
          </div>
          <div className="pb-4 border-b border-white/10">
            <p className="text-base text-purple-200 mb-2">Body Regular</p>
            <p className="text-sm text-purple-400">text-base - Standard body text</p>
          </div>
          <div className="pb-4 border-b border-white/10">
            <p className="text-sm text-purple-300 mb-2">Body Small</p>
            <p className="text-sm text-purple-400">text-sm - Secondary information, captions</p>
          </div>
          <div>
            <p className="text-xs text-purple-400 mb-2">Fine Print</p>
            <p className="text-sm text-purple-400">text-xs - Labels, metadata, legal text</p>
          </div>
        </div>
      </div>

      {/* Text Color Guidelines */}
      <div className="backdrop-blur-xl bg-white/10 rounded-2xl p-8 border border-white/20 shadow-2xl">
        <h3 className="text-2xl mb-6 text-purple-100">Text Color Guidelines</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-lg">
            <span className="text-white">Primary Text - White/Purple-50</span>
            <code className="text-xs text-purple-300">text-white / text-purple-50</code>
          </div>
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-lg">
            <span className="text-purple-100">Headings - Purple-100</span>
            <code className="text-xs text-purple-300">text-purple-100</code>
          </div>
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-lg">
            <span className="text-purple-200">Body Text - Purple-200</span>
            <code className="text-xs text-purple-300">text-purple-200</code>
          </div>
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-lg">
            <span className="text-purple-300">Secondary Text - Purple-300</span>
            <code className="text-xs text-purple-300">text-purple-300</code>
          </div>
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-lg">
            <span className="text-purple-400">Muted Text - Purple-400</span>
            <code className="text-xs text-purple-300">text-purple-400</code>
          </div>
        </div>
      </div>
    </div>
  );
}
