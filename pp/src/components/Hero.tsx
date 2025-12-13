import { ArrowRight, Play, Shield } from 'lucide-react';
import { motion } from 'motion/react';

export function Hero() {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center px-6 py-20">
      <div className="max-w-7xl mx-auto w-full">
        <div className="text-center max-w-4xl mx-auto">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 backdrop-blur-xl bg-purple-500/20 border border-purple-400/30 rounded-full mb-8"
          >
            <Shield className="w-4 h-4 text-purple-300" />
            <span className="text-sm text-purple-200">Advanced AI Security Platform</span>
          </motion.div>

          {/* Main Heading with staggered animation */}
          <motion.h1
            className="text-5xl md:text-7xl mb-6 bg-gradient-to-r from-purple-300 via-white to-purple-300 bg-clip-text text-transparent"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            Intelligent Security
            <br />
            <span className="bg-gradient-to-r from-purple-400 via-purple-300 to-purple-200 bg-clip-text text-transparent">
              Powered by AI
            </span>
          </motion.h1>

          {/* Subtitle with fade-in animation */}
          <motion.p
            className="text-xl md:text-2xl text-purple-200 mb-12 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            Real-time detection of fights, weapons, screams, and crowd anomalies.
            Protect your spaces with cutting-edge artificial intelligence.
          </motion.p>

          {/* CTA Buttons with staggered animation */}
          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            <motion.button
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-purple-700 rounded-xl text-white shadow-lg shadow-purple-500/30 hover:shadow-xl hover:shadow-purple-500/50 transition-all flex items-center gap-2 group"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Start Free Trial
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.button>

            <motion.button
              className="px-8 py-4 backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl text-purple-100 hover:bg-white/20 transition-all flex items-center gap-2 group"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Play className="w-5 h-5" />
              Watch Demo
            </motion.button>
          </motion.div>

          {/* Stats with fade-in animation */}
          <motion.div
            className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-20 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
          >
            {[
              { value: '99.9%', label: 'Accuracy Rate' },
              { value: '<1s', label: 'Response Time' },
              { value: '24/7', label: 'Monitoring' },
              { value: '500+', label: 'Active Sites' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                className="backdrop-blur-xl bg-white/5 rounded-xl p-6 border border-white/10"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 1 + index * 0.1 }}
                whileHover={{ scale: 1.05 }}
              >
                <div className="text-3xl text-purple-300 mb-1">{stat.value}</div>
                <div className="text-sm text-purple-400">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Floating particles effect indicator */}
      <motion.div
        className="absolute bottom-10 left-1/2 -translate-x-1/2"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          duration: 1,
          delay: 1.5,
          repeat: Infinity,
          repeatType: 'reverse',
        }}
      >
        <div className="w-6 h-10 border-2 border-purple-400/30 rounded-full flex items-start justify-center p-2">
          <motion.div
            className="w-1.5 h-1.5 bg-purple-400 rounded-full"
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </section>
  );
}
