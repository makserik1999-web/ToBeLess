import { useState, createContext, useContext } from 'react';
import { TopNav } from './dashboard/TopNav';
import { Overview } from './dashboard/Overview';
import { LiveMonitoring } from './dashboard/LiveMonitoring';
import { AlertsView } from './dashboard/AlertsView';
import { Analytics } from './dashboard/Analytics';
import { IncidentsView } from './dashboard/IncidentsView';
import { UsersView } from './dashboard/UsersView';
import { ReportsView } from './dashboard/ReportsView';
import { SettingsView } from './dashboard/SettingsView';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>({ theme: 'light', toggleTheme: () => {} });

export const useTheme = () => useContext(ThemeContext);

interface DashboardProps {
  onBackToLanding: () => void;
}

export function Dashboard({ onBackToLanding }: DashboardProps) {
  const [activeView, setActiveView] = useState('overview');
  const [theme, setTheme] = useState<Theme>('light');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const renderView = () => {
    switch (activeView) {
      case 'overview':
        return <Overview />;
      case 'monitoring':
        return <LiveMonitoring />;
      case 'alerts':
        return <AlertsView />;
      case 'analytics':
        return <Analytics />;
      case 'incidents':
        return <IncidentsView />;
      case 'users':
        return <UsersView />;
      case 'reports':
        return <ReportsView />;
      case 'settings':
        return <SettingsView />;
      default:
        return <Overview />;
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <div className={`min-h-screen ${
        theme === 'light' 
          ? 'bg-gradient-to-br from-purple-50 via-white to-purple-100' 
          : 'bg-black'
      }`}>
        {/* Subtle background pattern */}
        {theme === 'light' && (
          <div className="fixed inset-0 opacity-20">
            <div className="absolute inset-0" style={{
              backgroundImage: `radial-gradient(circle at 1px 1px, rgba(168, 85, 247, 0.15) 1px, transparent 0)`,
              backgroundSize: '40px 40px'
            }}></div>
          </div>
        )}

        <div className="relative">
          <TopNav 
            activeView={activeView}
            setActiveView={setActiveView}
            onBackToLanding={onBackToLanding}
          />
          
          <main className="pt-24 px-6 lg:px-16 max-w-[2000px] mx-auto pb-12">
            {renderView()}
          </main>
        </div>
      </div>
    </ThemeContext.Provider>
  );
}