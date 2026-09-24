import React from 'react';
import { LayoutDashboard, List, Settings } from 'lucide-react';

interface SidebarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
}

export function Sidebar({ currentTab, onTabChange }: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">AN</div>
        <div className="sidebar-title">ANPR System</div>
      </div>
      
      <div className="nav-links">
        <button 
          className={`nav-link ${currentTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => onTabChange('dashboard')}
        >
          <LayoutDashboard size={20} />
          Dashboard
        </button>
        <button 
          className={`nav-link ${currentTab === 'history' ? 'active' : ''}`}
          onClick={() => onTabChange('history')}
        >
          <List size={20} />
          Activity Log
        </button>
        <button 
          className={`nav-link ${currentTab === 'settings' ? 'active' : ''}`}
          onClick={() => onTabChange('settings')}
        >
          <Settings size={20} />
          Settings
        </button>
      </div>
    </div>
  );
}
