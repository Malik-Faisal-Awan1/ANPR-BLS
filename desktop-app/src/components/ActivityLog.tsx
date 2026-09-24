import React, { useEffect, useState } from 'react';
import { HistoryResponse } from '../types';
import { AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001';

export function ActivityLog() {
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [page, setPage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const limit = 10;

  useEffect(() => {
    fetch(`${API_BASE}/api/history?page=${page}&limit=${limit}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then(data => {
        setHistory(data);
        setError(null);
      })
      .catch(err => {
        console.error("Failed to fetch history:", err);
        setError(err.message || "Failed to connect to backend");
      });
  }, [page]);

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Activity Log</h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
            Historical plate reads and system events
          </p>
        </div>
      </div>
      
      {error && (
        <div style={{ marginBottom: '1.5rem', backgroundColor: 'var(--status-error-bg)', color: 'var(--status-error)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--status-error)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={20} />
          <strong>Connection Error:</strong> {error}
        </div>
      )}

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>TIMESTAMP</th>
                <th>IMAGE</th>
                <th>PLATE NUMBER</th>
                <th>CONFIDENCE</th>
                <th>PROCESSING</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody>
              {history?.data.map((row) => (
                <tr key={row.id}>
                  <td>
                    {new Date(row.timestamp).toLocaleString()}
                  </td>
                  <td>
                    <div className="feed-thumb" style={{ width: '80px', height: '60px' }}>
                      {row.imagePath ? (
                        <img src={`${API_BASE}${row.imagePath}`} alt="thumbnail" />
                      ) : (
                        <AlertCircle size={24} color="var(--text-muted)" />
                      )}
                    </div>
                  </td>
                  <td className="td-plate" style={{ color: row.success ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    {row.success && row.plateNumber ? row.plateNumber : 'N/A'}
                  </td>
                  <td>
                    {row.success ? `${(row.confidence * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td>
                    {row.processingMs.toFixed(1)}ms
                  </td>
                  <td>
                    <span className={`badge ${row.success ? 'badge-success' : 'badge-error'}`}>
                      {row.success ? 'SUCCESS' : 'FAILED'}
                    </span>
                  </td>
                </tr>
              ))}
              
              {history?.data.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    No historical data available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {history && history.pagination.totalPages > 1 && (
          <div className="pagination">
            <span>
              Showing {((history.pagination.page - 1) * history.pagination.limit) + 1} to {Math.min(history.pagination.page * history.pagination.limit, history.pagination.total)} of {history.pagination.total} entries
            </span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                className="btn" 
                disabled={page === 1}
                onClick={() => setPage(p => Math.max(1, p - 1))}
                style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <ChevronLeft size={16} /> Prev
              </button>
              <button 
                className="btn" 
                disabled={page >= history.pagination.totalPages}
                onClick={() => setPage(p => Math.min(history.pagination.totalPages, p + 1))}
                style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                Next <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
