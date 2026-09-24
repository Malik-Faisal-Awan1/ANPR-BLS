import React, { useEffect, useState } from 'react';
import { ConfigData } from '../types';
import { Settings as SettingsIcon, AlertTriangle } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001';

export function Settings() {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [statusData, setStatusData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/config`).then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      }),
      fetch(`${API_BASE}/api/status`).then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
    ])
    .then(([configData, statusRes]) => {
      setConfig(configData);
      setStatusData(statusRes);
      setError(null);
    })
    .catch(err => {
      console.error("Failed to fetch settings data:", err);
      setError(err.message || "Failed to connect to backend");
    });
  }, []);

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings</h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
            System configuration (Read-only)
          </p>
        </div>
      </div>

      <div style={{ marginBottom: '1.5rem', backgroundColor: 'var(--status-warning-bg)', border: '1px solid var(--status-warning)', padding: '1rem', borderRadius: 'var(--radius-md)', display: 'flex', gap: '12px', alignItems: 'flex-start', color: 'var(--status-warning)' }}>
        <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.95rem', lineHeight: 1.5 }}>
          <strong>Configuration is read-only.</strong> These settings are currently loaded from the backend's <code>config.py</code> file. To modify them, edit the file and restart the system.
        </div>
      </div>
      
      {error && (
        <div style={{ marginBottom: '1.5rem', backgroundColor: 'var(--status-error-bg)', color: 'var(--status-error)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--status-error)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={20} />
          <strong>Connection Error:</strong> {error}
        </div>
      )}

      <div className="dashboard-grid">
        <div className="card">
          <div className="card-title">
            <SettingsIcon size={18} /> Effective Configuration
          </div>
          
          <div className="settings-list">
            {config ? Object.entries(config).map(([key, value]) => (
              <div key={key} className="settings-item">
                <div className="settings-key">{key}</div>
                <div className="settings-value">{String(value)}</div>
              </div>
            )) : (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                Loading configuration...
              </div>
            )}
          </div>
        </div>

        <div className="card" style={{ height: 'fit-content' }}>
          <div className="card-title">
            Directory Paths
          </div>
          
          <div className="settings-list">
            <div className="settings-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <div className="settings-key">Watch Directory</div>
              <div className="settings-value" style={{ width: '100%', wordBreak: 'break-all' }}>
                {statusData?.watcher_directory || 'Loading...'}
              </div>
            </div>
            
            <div className="settings-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <div className="settings-key">Output Directory</div>
              <div className="settings-value" style={{ width: '100%', wordBreak: 'break-all' }}>
                {statusData?.output_directory || 'Loading...'}
              </div>
            </div>

            <div className="settings-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <div className="settings-key">Failed Directory</div>
              <div className="settings-value" style={{ width: '100%', wordBreak: 'break-all' }}>
                {statusData?.failed_directory || 'Loading...'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
