import { ArrowRight, CheckCircle } from 'lucide-react';
import { motion } from 'motion/react';
import { useInView } from './hooks/useInView';

export function CTA() {
  const [ref, isInView] = useInView({ threshold: 0.1 });

  const benefits = [
    'No credit card required',
    '14-day free trial',
    'Full access to all features',
    'Cancel anytime',
  ];

  return (
    <section id="cta" className="py-24 px-6" ref={ref}>
      <div className="max-w-5xl mx-auto">
        <motion.div
          className="relative backdrop-blur-2xl bg-gradient-to-br from-purple-900/40 to-purple-800/40 rounded-3xl p-12 md:p-16 border border-purple-400/30 shadow-2xl shadow-purple-500/20 overflow-hidden"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8 }}
        >
          {/* Background decoration */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500 rounded-full filter blur-3xl opacity-20"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-purple-700 rounded-full filter blur-3xl opacity-20"></div>

          <div className="relative z-10 text-center">
            {/* Badge */}
            <motion.div
              className="inline-block px-4 py-2 backdrop-blur-xl bg-white/10 border border-white/20 rounded-full mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <span className="text-sm text-purple-200">Limited Time Offer</span>
            </motion.div>

            {/* Heading */}
            <motion.h2
              className="text-4xl md:text-5xl mb-6 text-white"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              Ready to Secure Your Space?
            </motion.h2>

            <motion.p
              className="text-xl text-purple-200 mb-8 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              Join hundreds of organizations using Tobeles AI to protect their people and assets with intelligent, proactive security.
            </motion.p>

            {/* Benefits */}
            <motion.div
              className="flex flex-wrap justify-center gap-6 mb-10"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              {benefits.map((benefit, index) => (
                <motion.div
                  key={benefit}
                  className="flex items-center gap-2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={isInView ? { opacity: 1, x: 0 } : {}}
                  transition={{ duration: 0.4, delay: 0.6 + index * 0.1 }}
                >
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-purple-200">{benefit}</span>
                </motion.div>
              ))}
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <motion.button
                className="px-10 py-4 bg-white text-purple-900 rounded-xl shadow-lg hover:shadow-xl transition-all flex items-center gap-2 group"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Start Your Free Trial
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </motion.button>

              <motion.button
                className="px-10 py-4 backdrop-blur-xl bg-white/10 border border-white/30 rounded-xl text-white hover:bg-white/20 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Schedule a Demo
              </motion.button>
            </motion.div>

            {/* Trust Badge */}
            <motion.p
              className="text-sm text-purple-300 mt-8"
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 1 } : {}}
              transition={{ duration: 0.6, delay: 1 }}
            >
              Trusted by security teams at Fortune 500 companies
            </motion.p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
