import { Shield, Swords, Volume2, Users } from 'lucide-react';
import { motion } from 'motion/react';
import { useInView } from './hooks/useInView';

export function Features() {
  const [ref, isInView] = useInView({ threshold: 0.1 });

  const features = [
    {
      icon: Swords,
      title: 'Fight Detection',
      description: 'Advanced computer vision algorithms identify physical altercations in real-time, enabling rapid response and intervention.',
      gradient: 'from-purple-500 to-purple-700',
    },
    {
      icon: Shield,
      title: 'Weapon Recognition',
      description: 'Instantly detect and classify weapons including firearms, knives, and other dangerous objects with industry-leading accuracy.',
      gradient: 'from-purple-600 to-purple-800',
    },
    {
      icon: Volume2,
      title: 'Scream Analysis',
      description: 'Audio AI distinguishes distress calls from ambient noise, alerting security teams to potential emergencies immediately.',
      gradient: 'from-purple-500 to-purple-700',
    },
    {
      icon: Users,
      title: 'Crowd Monitoring',
      description: 'Track crowd density, flow patterns, and anomalous behavior to prevent overcrowding and identify potential safety risks.',
      gradient: 'from-purple-600 to-purple-800',
    },
  ];

  return (
    <section id="features" className="py-24 px-6" ref={ref}>
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            className="inline-block px-4 py-2 backdrop-blur-xl bg-purple-500/20 border border-purple-400/30 rounded-full mb-4"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={isInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.5 }}
          >
            <span className="text-sm text-purple-300">Core Capabilities</span>
          </motion.div>
          <h2 className="text-4xl md:text-5xl mb-4 text-purple-100">
            Comprehensive AI Protection
          </h2>
          <p className="text-xl text-purple-300 max-w-2xl mx-auto">
            Four powerful detection systems working together to keep your environment safe
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="group backdrop-blur-xl bg-white/5 rounded-2xl p-8 border border-white/10 hover:border-purple-400/30 hover:bg-white/10 transition-all duration-300 cursor-pointer"
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.15 }}
              whileHover={{ scale: 1.02, y: -5 }}
            >
              {/* Icon */}
              <motion.div
                className={`w-16 h-16 bg-gradient-to-br ${feature.gradient} rounded-xl mb-6 flex items-center justify-center shadow-lg shadow-purple-500/30 group-hover:shadow-xl group-hover:shadow-purple-500/50 transition-all`}
                whileHover={{ rotate: [0, -10, 10, -10, 0] }}
                transition={{ duration: 0.5 }}
              >
                <feature.icon className="w-8 h-8 text-white" strokeWidth={1.5} />
              </motion.div>

              {/* Content */}
              <h3 className="text-2xl mb-3 text-purple-100 group-hover:text-white transition-colors">
                {feature.title}
              </h3>
              <p className="text-purple-300 group-hover:text-purple-200 transition-colors">
                {feature.description}
              </p>

              {/* Hover Indicator */}
              <motion.div
                className="mt-6 flex items-center gap-2 text-purple-400 opacity-0 group-hover:opacity-100 transition-opacity"
                initial={{ x: -10 }}
                whileHover={{ x: 0 }}
              >
                <span className="text-sm">Learn more</span>
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </motion.div>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          className="text-center mt-16"
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <p className="text-purple-300 mb-6">
            All features integrate seamlessly into a unified security platform
          </p>
          <motion.button
            className="px-8 py-3 backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl text-purple-100 hover:bg-white/20 transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            View Full Feature List
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
