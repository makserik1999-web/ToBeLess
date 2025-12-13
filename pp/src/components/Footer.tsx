import { Cpu, Mail, MapPin, Phone, Twitter, Linkedin, Github, Youtube } from 'lucide-react';
import { motion } from 'motion/react';

export function Footer() {
  const footerSections = {
    product: {
      title: 'Product',
      links: ['Features', 'Pricing', 'Case Studies', 'Demo', 'API Documentation'],
    },
    company: {
      title: 'Company',
      links: ['About Us', 'Careers', 'Blog', 'Press Kit', 'Contact'],
    },
    resources: {
      title: 'Resources',
      links: ['Documentation', 'Support', 'Community', 'Status', 'Partners'],
    },
    legal: {
      title: 'Legal',
      links: ['Privacy Policy', 'Terms of Service', 'Cookie Policy', 'GDPR', 'Security'],
    },
  };

  const socialLinks = [
    { icon: Twitter, href: '#', label: 'Twitter' },
    { icon: Linkedin, href: '#', label: 'LinkedIn' },
    { icon: Github, href: '#', label: 'GitHub' },
    { icon: Youtube, href: '#', label: 'YouTube' },
  ];

  return (
    <footer className="relative border-t border-white/10 backdrop-blur-xl bg-white/5">
      <div className="max-w-7xl mx-auto px-6 py-16">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-12 mb-12">
          {/* Brand Column */}
          <div className="lg:col-span-2">
            <motion.div
              className="flex items-center gap-3 mb-4"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg shadow-lg shadow-purple-500/50 flex items-center justify-center">
                <Cpu className="w-6 h-6 text-white" strokeWidth={1.5} />
              </div>
              <span className="text-xl bg-gradient-to-r from-purple-300 via-purple-200 to-purple-300 bg-clip-text text-transparent">
                Tobeles AI
              </span>
            </motion.div>
            <motion.p
              className="text-purple-300 mb-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              Intelligent security solutions powered by advanced artificial intelligence. Protecting what matters most.
            </motion.p>

            {/* Contact Info */}
            <motion.div
              className="space-y-3"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex items-center gap-3 text-sm text-purple-300">
                <Mail className="w-4 h-4" />
                <a href="mailto:contact@tobeles.ai" className="hover:text-purple-200 transition-colors">
                  contact@tobeles.ai
                </a>
              </div>
              <div className="flex items-center gap-3 text-sm text-purple-300">
                <Phone className="w-4 h-4" />
                <a href="tel:+1234567890" className="hover:text-purple-200 transition-colors">
                  +1 (234) 567-890
                </a>
              </div>
              <div className="flex items-center gap-3 text-sm text-purple-300">
                <MapPin className="w-4 h-4" />
                <span>San Francisco, CA</span>
              </div>
            </motion.div>
          </div>

          {/* Links Columns */}
          {Object.entries(footerSections).map(([key, section], sectionIndex) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 * (sectionIndex + 1) }}
            >
              <h3 className="text-purple-100 mb-4">{section.title}</h3>
              <ul className="space-y-3">
                {section.links.map((link) => (
                  <li key={link}>
                    <a
                      href="#"
                      className="text-sm text-purple-300 hover:text-purple-200 transition-colors"
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        {/* Bottom Bar */}
        <motion.div
          className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-6"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {/* Copyright */}
          <p className="text-sm text-purple-400">
            © {new Date().getFullYear()} Tobeles AI. All rights reserved.
          </p>

          {/* Social Links */}
          <div className="flex items-center gap-4">
            {socialLinks.map((social, index) => (
              <motion.a
                key={social.label}
                href={social.href}
                className="w-10 h-10 backdrop-blur-lg bg-white/5 border border-white/10 rounded-lg flex items-center justify-center text-purple-300 hover:bg-white/10 hover:text-purple-200 hover:border-purple-400/30 transition-all"
                aria-label={social.label}
                whileHover={{ scale: 1.1, rotate: 5 }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: 0.5 + index * 0.1 }}
              >
                <social.icon className="w-5 h-5" />
              </motion.a>
            ))}
          </div>

          {/* Legal Links */}
          <div className="flex items-center gap-6">
            <a href="#privacy" className="text-sm text-purple-400 hover:text-purple-200 transition-colors">
              Privacy Policy
            </a>
            <a href="#terms" className="text-sm text-purple-400 hover:text-purple-200 transition-colors">
              Terms of Service
            </a>
          </div>
        </motion.div>
      </div>
    </footer>
  );
}
