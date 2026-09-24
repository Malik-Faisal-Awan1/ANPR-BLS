import React, { useEffect, useState } from 'react';
import { Camera, PlateReadEvent } from '../types';
import { Camera as CameraIcon, Activity, AlertCircle, CheckCircle2 } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001';

export function Dashboard() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [events, setEvents] = useState<PlateReadEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch cameras
    fetch(`${API_BASE}/api/cameras`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data.cameras) setCameras(data.cameras);
        setError(null);
      })
      .catch(err => {
        console.error("Failed to fetch cameras:", err);
        setError(err.message || "Failed to connect to backend");
      });

    // Connect to SSE
    const eventSource = new EventSource(`${API_BASE}/api/events`);
    
    eventSource.onopen = () => setIsConnected(true);
    
    eventSource.onmessage = (e) => {
      try {
        const newEvent: PlateReadEvent = JSON.parse(e.data);
        setEvents(prev => {
          const updated = [newEvent, ...prev];
          return updated.slice(0, 10); // Keep last 10
        });
      } catch (err) {
        console.error("Failed to parse SSE event:", err);
      }
    };

    eventSource.onerror = (e) => {
      setIsConnected(false);
      console.error("SSE Connection error", e);
      setError("SSE Stream disconnected or failed");
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const latestEvent = events[0];

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
            Live ANPR Monitoring
          </p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          {cameras.map(cam => (
            <div key={cam.id} className="card" style={{ padding: '0.75rem 1rem', flexDirection: 'row', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '10px', height: '10px', borderRadius: '50%',
                backgroundColor: cam.status === 'online' ? 'var(--status-success)' : 'var(--status-error)',
                boxShadow: `0 0 10px ${cam.status === 'online' ? 'var(--status-success)' : 'var(--status-error)'}`
              }} />
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{cam.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{cam.location}</div>
              </div>
            </div>
          ))}
          <div className="card" style={{ padding: '0.75rem 1rem', flexDirection: 'row', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} color={isConnected ? 'var(--status-success)' : 'var(--status-error)'} />
            <span style={{ fontSize: '0.9rem', fontWeight: 500, color: isConnected ? 'var(--status-success)' : 'var(--status-error)' }}>
              {isConnected ? 'Stream Active' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        {error && (
          <div style={{ gridColumn: '1 / -1', backgroundColor: 'var(--status-error-bg)', color: 'var(--status-error)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--status-error)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle size={20} />
            <strong>Connection Error:</strong> {error}
          </div>
        )}
        <div className="card hero-read">
          <div className="card-title" style={{ width: '100%', justifyContent: 'flex-start' }}>
            <CameraIcon size={18} /> Latest Capture
          </div>
          
          {latestEvent ? (
            <>
              <div className="hero-image-container">
                {latestEvent.imagePath ? (
                  <img 
                    src={`${API_BASE}${latestEvent.imagePath}`} 
                    alt="Latest plate capture" 
                    className="hero-image"
                  />
                ) : (
                  <div className="hero-placeholder">
                    <AlertCircle size={48} opacity={0.5} />
                    <span>Image unavailable</span>
                  </div>
                )}
              </div>
              
              <div className="plate-display">
                {latestEvent.success && latestEvent.plateNumber ? latestEvent.plateNumber : 'NO PLATE'}
              </div>
              
              <div className="plate-meta">
                <span className={`badge ${latestEvent.success ? 'badge-success' : 'badge-error'}`}>
                  {latestEvent.success ? 'SUCCESS' : 'FAILED'}
                </span>
                {latestEvent.success && (
                  <span className="badge badge-warning">
                    CONF {(latestEvent.confidence * 100).toFixed(1)}%
                  </span>
                )}
                <span>{new Date(latestEvent.timestamp).toLocaleTimeString()}</span>
                <span>{latestEvent.processingMs.toFixed(1)}ms</span>
              </div>
            </>
          ) : (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              <div className="loading-spinner" style={{ marginBottom: '1rem' }}></div>
              <p>Waiting for plate reads...</p>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">Recent Activity</div>
          <div className="activity-feed">
            {events.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem 0' }}>
                No events yet
              </div>
            ) : (
              events.map(event => (
                <div key={event.id} className="feed-item">
                  <div className="feed-thumb">
                    {event.imagePath ? (
                      <img src={`${API_BASE}${event.imagePath}`} alt="thumbnail" />
                    ) : (
                      <AlertCircle size={24} color="var(--text-muted)" />
                    )}
                  </div>
                  <div className="feed-info">
                    <div className="feed-plate" style={{ color: event.success ? 'var(--text-primary)' : 'var(--status-error)' }}>
                      {event.success && event.plateNumber ? event.plateNumber : 'FAILED'}
                    </div>
                    <div className="feed-time">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  {event.success ? (
                    <CheckCircle2 size={20} color="var(--status-success)" />
                  ) : (
                    <AlertCircle size={20} color="var(--status-error)" />
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
