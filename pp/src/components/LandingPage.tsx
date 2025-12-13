import { useState, useEffect, useRef, useMemo } from 'react';
import { motion, useMotionValue, useTransform, useSpring, useScroll } from 'motion/react';
import { Shield, ArrowRight, Sparkles, Eye, Brain, Zap } from 'lucide-react';

interface LandingPageProps {
  onEnterDashboard: () => void;
}

export function LandingPage({ onEnterDashboard }: LandingPageProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll();

  useEffect(() => {
    let rafId: number;
    const handleMouseMove = (e: MouseEvent) => {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        setMousePosition({ x: e.clientX, y: e.clientY });
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, []);

  const springConfig = { damping: 25, stiffness: 150 };
  const mouseX = useSpring(mousePosition.x, springConfig);
  const mouseY = useSpring(mousePosition.y, springConfig);

  // Memoize orb positions so they don't recalculate
  const orbs = useMemo(() => {
    return [...Array(8)].map((_, i) => ({
      id: i,
      color: i % 3 === 0 ? 'rgba(168, 85, 247, 0.15)' : i % 3 === 1 ? 'rgba(147, 51, 234, 0.15)' : 'rgba(126, 34, 206, 0.15)',
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      x: [0, Math.random() * 200 - 100, 0],
      y: [0, Math.random() * 200 - 100, 0],
      duration: 10 + Math.random() * 10,
      delay: Math.random() * 2
    }));
  }, []);

  // Memoize particle positions
  const particles = useMemo(() => {
    return [...Array(3)].map((_, i) => ({
      id: i,
      startX: Math.random() * 100 - 50,
      startY: Math.random() * 100 - 50,
      endX: Math.random() * 100 - 50,
      endY: -100,
      duration: 2 + Math.random() * 2,
      delay: Math.random() * 2
    }));
  }, []);

  return (
    <div ref={containerRef} className="bg-black text-white overflow-hidden">
      {/* Floating Navigation */}
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.6, 0.05, 0.01, 0.9] }}
        className="fixed top-6 left-1/2 -translate-x-1/2 z-50"
      >
        <div className="backdrop-blur-2xl bg-zinc-900/40 border border-zinc-800/50 rounded-full px-6 py-3 shadow-2xl">
          <div className="flex items-center gap-8">
            <motion.div 
              className="flex items-center gap-2 cursor-pointer"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-full flex items-center justify-center relative overflow-hidden">
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  animate={{ x: ['-100%', '200%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
                <Shield className="w-4 h-4 text-white relative z-10" />
              </div>
              <span className="font-semibold">Tobeles AI</span>
            </motion.div>
            <button 
              onClick={onEnterDashboard}
              className="group relative px-5 py-2 rounded-full overflow-hidden"
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-purple-600 to-purple-700"
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.3 }}
              />
              <span className="relative z-10 flex items-center gap-2 font-medium">
                Enter Platform
                <motion.div
                  animate={{ x: [0, 4, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <ArrowRight className="w-4 h-4" />
                </motion.div>
              </span>
            </button>
          </div>
        </div>
      </motion.nav>

      {/* Hero Section with Flowing Orbs */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background layer - lowest z-index */}
        <div className="absolute inset-0 z-0">
          <motion.div
            className="absolute inset-0 opacity-30"
            animate={{
              background: [
                'radial-gradient(circle at 20% 50%, rgba(168, 85, 247, 0.3) 0%, transparent 50%)',
                'radial-gradient(circle at 80% 50%, rgba(168, 85, 247, 0.3) 0%, transparent 50%)',
                'radial-gradient(circle at 50% 80%, rgba(168, 85, 247, 0.3) 0%, transparent 50%)',
                'radial-gradient(circle at 20% 50%, rgba(168, 85, 247, 0.3) 0%, transparent 50%)',
              ]
            }}
            transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
          />

          {/* Floating Orbs - optimized */}
          {orbs.map((orb) => (
            <motion.div
              key={orb.id}
              className="absolute w-64 h-64 rounded-full blur-3xl opacity-40 pointer-events-none will-change-transform"
              style={{
                background: `radial-gradient(circle, ${orb.color} 0%, transparent 70%)`,
                left: orb.left,
                top: orb.top,
              }}
              animate={{
                x: orb.x,
                y: orb.y,
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: orb.duration,
                repeat: Infinity,
                ease: 'easeInOut',
                delay: orb.delay,
              }}
            />
          ))}

          {/* Interactive cursor follow effect */}
          <motion.div
            className="absolute w-96 h-96 rounded-full blur-3xl pointer-events-none opacity-30 will-change-transform"
            style={{
              background: 'radial-gradient(circle, rgba(168, 85, 247, 0.2) 0%, transparent 70%)',
              x: useTransform(mouseX, (value) => value - 192),
              y: useTransform(mouseY, (value) => value - 192),
            }}
          />
        </div>

        {/* Content layer - highest z-index */}
        <div className="relative z-20 max-w-7xl mx-auto px-6 text-center">
          {/* Animated badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="inline-block mb-8"
          >
            <motion.div
              className="relative inline-flex items-center gap-3 px-6 py-3 rounded-full overflow-hidden group cursor-pointer"
              whileHover={{ scale: 1.05 }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-purple-700/20 border border-purple-500/30 rounded-full" />
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-purple-400/0 via-purple-400/20 to-purple-400/0"
                animate={{ x: ['-100%', '200%'] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              />
              <Sparkles className="w-4 h-4 text-purple-400 relative z-10" />
              <span className="text-sm font-medium text-purple-300 relative z-10">Powered by Advanced AI</span>
            </motion.div>
          </motion.div>

          {/* Main heading with character animation */}
          <div className="mb-8">
            {['Security', 'That', 'Thinks'].map((word, wordIndex) => (
              <div key={wordIndex} className="inline-block">
                <motion.h1
                  className="text-7xl md:text-9xl font-bold inline-block mr-6"
                  initial={{ opacity: 0, y: 100 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ 
                    duration: 0.8, 
                    delay: 0.3 + wordIndex * 0.1,
                    ease: [0.6, 0.05, 0.01, 0.9]
                  }}
                >
                  {word.split('').map((char, i) => (
                    <motion.span
                      key={i}
                      className={`inline-block ${
                        wordIndex === 2 
                          ? 'bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent'
                          : 'text-white'
                      }`}
                      whileHover={{ 
                        y: -10,
                        transition: { duration: 0.2 }
                      }}
                    >
                      {char}
                    </motion.span>
                  ))}
                </motion.h1>
              </div>
            ))}
          </div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="text-2xl md:text-3xl text-zinc-400 mb-12 max-w-3xl mx-auto font-medium"
          >
            AI-powered threat detection before danger strikes
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1 }}
            className="flex items-center justify-center gap-6"
          >
            <motion.button
              onClick={onEnterDashboard}
              className="group relative px-8 py-4 rounded-2xl overflow-hidden"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-purple-700 rounded-2xl" />
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-purple-400 to-purple-600 rounded-2xl opacity-0 group-hover:opacity-100"
                transition={{ duration: 0.3 }}
              />
              <span className="relative z-10 flex items-center gap-2 text-lg font-semibold">
                Experience the Future
                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <ArrowRight className="w-5 h-5" />
                </motion.div>
              </span>
            </motion.button>
          </motion.div>

          {/* Floating cards preview */}
          <motion.div
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1.2 }}
            className="mt-20 relative"
          >
            <div className="flex items-center justify-center gap-4 flex-wrap">
              {[
                { icon: Shield, label: 'Weapons', color: 'from-orange-500 to-red-600', delay: 0 },
                { icon: Eye, label: 'Crowds', color: 'from-blue-500 to-purple-600', delay: 0.15 },
                { icon: Brain, label: 'Behavior', color: 'from-purple-500 to-pink-600', delay: 0.3 },
                { icon: Zap, label: 'Real-time', color: 'from-green-500 to-emerald-600', delay: 0.45 },
              ].map((item, index) => (
                <motion.div
                  key={item.label}
                  className="relative group cursor-pointer"
                  initial={{ opacity: 0, y: 100, rotateX: -90, scale: 0.5 }}
                  animate={{ 
                    opacity: 1, 
                    y: 0, 
                    rotateX: 0, 
                    scale: 1,
                  }}
                  transition={{ 
                    delay: 1.4 + item.delay,
                    duration: 0.8,
                    type: "spring",
                    stiffness: 100,
                    damping: 15
                  }}
                  whileHover={{ 
                    y: -20, 
                    scale: 1.1,
                    rotateY: 10,
                    rotateX: 5,
                    transition: { duration: 0.3 }
                  }}
                  style={{ perspective: 1000 }}
                >
                  <div className="backdrop-blur-xl bg-zinc-900/40 border border-zinc-800/50 rounded-2xl p-6 w-40 h-40 flex flex-col items-center justify-center relative overflow-hidden">
                    {/* Animated background gradient */}
                    <motion.div
                      className={`absolute inset-0 bg-gradient-to-br ${item.color} opacity-0 group-hover:opacity-20`}
                      animate={{
                        scale: [1, 1.2, 1],
                        rotate: [0, 180, 360],
                      }}
                      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                    />
                    
                    {/* Particles effect */}
                    {particles.map((particle) => (
                      <motion.div
                        key={particle.id}
                        className={`absolute w-1 h-1 rounded-full bg-gradient-to-r ${item.color} pointer-events-none will-change-transform`}
                        style={{ 
                          left: '50%',
                          top: '50%',
                        }}
                        animate={{
                          x: [particle.startX, particle.endX],
                          y: [particle.startY, particle.endY],
                          opacity: [0, 1, 0],
                        }}
                        transition={{
                          duration: particle.duration,
                          repeat: Infinity,
                          delay: particle.delay,
                          ease: "linear"
                        }}
                      />
                    ))}

                    <motion.div
                      className={`w-16 h-16 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center mb-3 relative overflow-hidden z-10`}
                      whileHover={{ rotate: 360, scale: 1.1 }}
                      transition={{ duration: 0.6 }}
                      animate={{
                        boxShadow: [
                          '0 0 20px rgba(168, 85, 247, 0.3)',
                          '0 0 40px rgba(168, 85, 247, 0.5)',
                          '0 0 20px rgba(168, 85, 247, 0.3)',
                        ]
                      }}
                      style={{ transition: 'box-shadow 2s ease-in-out' }}
                    >
                      <motion.div
                        animate={{ 
                          scale: [1, 1.2, 1],
                          rotate: [0, 180, 360]
                        }}
                        transition={{ duration: 3, repeat: Infinity }}
                      >
                        <item.icon className="w-8 h-8 text-white relative z-10" />
                      </motion.div>
                      
                      {/* Ripple effect */}
                      <motion.div
                        className="absolute inset-0 bg-white/20 rounded-xl"
                        animate={{ 
                          scale: [1, 1.5, 1],
                          opacity: [0.5, 0, 0.5]
                        }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                    </motion.div>
                    
                    <motion.span 
                      className="text-sm font-semibold text-zinc-300 relative z-10"
                      animate={{
                        y: [0, -2, 0],
                      }}
                      transition={{ duration: 2, repeat: Infinity, delay: index * 0.2 }}
                    >
                      {item.label}
                    </motion.span>

                    {/* Scan line effect */}
                    <motion.div
                      className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/50 to-transparent"
                      animate={{ y: [0, 160, 0] }}
                      transition={{ duration: 3, repeat: Infinity, delay: index * 0.5 }}
                    />
                  </div>
                  
                  {/* Glow effect */}
                  <motion.div
                    className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${item.color} opacity-0 group-hover:opacity-30 blur-2xl -z-10`}
                    animate={{
                      scale: [1, 1.2, 1],
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-10 left-1/2 -translate-x-1/2 z-20"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 border-2 border-purple-500/30 rounded-full flex items-start justify-center p-2">
            <motion.div
              className="w-1.5 h-1.5 bg-purple-500 rounded-full"
              animate={{ y: [0, 20, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
        </motion.div>
      </section>

      {/* Flowing features section */}
      <section className="relative py-32 px-6">
        <motion.div 
          className="max-w-7xl mx-auto"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1 }}
        >
          <motion.div
            className="text-center mb-20"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-5xl font-bold mb-4">
              Intelligence That{' '}
              <span className="bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
                Never Sleeps
              </span>
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {[
              {
                title: 'Instant Detection',
                description: 'AI processes thousands of video frames per second, identifying threats faster than human perception',
                stat: '<500ms',
                color: 'from-purple-500 to-purple-700'
              },
              {
                title: 'Learning System',
                description: 'Continuously evolving algorithms that adapt to new threat patterns and environmental changes',
                stat: '99.9%',
                color: 'from-blue-500 to-purple-600'
              },
              {
                title: 'Multi-Sensor Fusion',
                description: 'Combines visual, audio, and behavioral data for comprehensive situational awareness',
                stat: '24/7',
                color: 'from-pink-500 to-purple-600'
              },
              {
                title: 'Zero False Positives',
                description: 'Advanced neural networks trained on millions of scenarios to minimize false alarms',
                stat: '0.01%',
                color: 'from-emerald-500 to-purple-600'
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                className="group relative"
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
              >
                <motion.div
                  className="backdrop-blur-xl bg-zinc-900/40 border border-zinc-800/50 rounded-3xl p-8 h-full relative overflow-hidden"
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.3 }}
                >
                  <motion.div
                    className={`absolute top-0 right-0 w-40 h-40 bg-gradient-to-br ${feature.color} rounded-full blur-3xl opacity-0 group-hover:opacity-30`}
                    transition={{ duration: 0.5 }}
                  />
                  
                  <div className="relative z-10">
                    <motion.div
                      className="text-6xl font-bold bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent mb-4"
                      initial={{ scale: 0 }}
                      whileInView={{ scale: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.5, delay: index * 0.1 + 0.3 }}
                    >
                      {feature.stat}
                    </motion.div>
                    <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                    <p className="text-zinc-400 leading-relaxed">{feature.description}</p>
                  </div>

                  <motion.div
                    className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-0 group-hover:opacity-100"
                    transition={{ duration: 0.3 }}
                  />
                </motion.div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Immersive CTA */}
      <section className="relative py-32 px-6 overflow-hidden">
        <motion.div
          className="absolute inset-0 z-0"
          animate={{
            background: [
              'radial-gradient(circle at 0% 0%, rgba(168, 85, 247, 0.15) 0%, transparent 50%)',
              'radial-gradient(circle at 100% 100%, rgba(168, 85, 247, 0.15) 0%, transparent 50%)',
              'radial-gradient(circle at 0% 100%, rgba(168, 85, 247, 0.15) 0%, transparent 50%)',
              'radial-gradient(circle at 100% 0%, rgba(168, 85, 247, 0.15) 0%, transparent 50%)',
              'radial-gradient(circle at 0% 0%, rgba(168, 85, 247, 0.15) 0%, transparent 50%)',
            ]
          }}
          transition={{ duration: 20, repeat: Infinity }}
        />

        <div className="relative z-10 max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-6xl font-bold mb-6">
              Ready to Transform
              <br />
              <span className="bg-gradient-to-r from-purple-400 via-purple-500 to-purple-600 bg-clip-text text-transparent">
                Your Security?
              </span>
            </h2>
            <p className="text-xl text-zinc-400 mb-12">
              Join the future of intelligent protection
            </p>
            <motion.button
              onClick={onEnterDashboard}
              className="group relative px-12 py-6 rounded-2xl overflow-hidden"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-purple-700 to-purple-600 bg-[length:200%_100%] animate-shimmer" />
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-purple-400 to-purple-600 opacity-0 group-hover:opacity-100"
                transition={{ duration: 0.3 }}
              />
              <span className="relative z-10 text-xl font-bold flex items-center gap-3">
                Launch Dashboard
                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <ArrowRight className="w-6 h-6" />
                </motion.div>
              </span>
            </motion.button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-zinc-900 py-12 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <motion.p
            className="text-zinc-500 mb-4"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            © 2024 Tobeles AI. Protecting what matters most.
          </motion.p>
          <div className="flex items-center justify-center gap-6 text-sm text-zinc-600">
            <a href="#" className="hover:text-purple-500 transition-colors">Privacy</a>
            <a href="#" className="hover:text-purple-500 transition-colors">Terms</a>
            <a href="#" className="hover:text-purple-500 transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
