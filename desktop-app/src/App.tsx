import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { ActivityLog } from './components/ActivityLog';
import { Settings } from './components/Settings';
import './App.css';

function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');

  return (
    <div className="app-container">
      <Sidebar currentTab={currentTab} onTabChange={setCurrentTab} />
      <div className="main-content">
        {currentTab === 'dashboard' && <Dashboard />}
        {currentTab === 'history' && <ActivityLog />}
        {currentTab === 'settings' && <Settings />}
      </div>
    </div>
  );
}

export default App;
