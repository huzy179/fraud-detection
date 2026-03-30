import { useState, useEffect } from "react";
import axios from "axios";
import Head from "next/head";
import {
  Shield,
  AlertTriangle,
  Activity,
  TrendingUp,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Sample transaction with realistic values
const SAMPLE_LEGIT = {
  V1: -1.359, V2: -0.072, V3: 2.536, V4: 1.378, V5: -0.338,
  V6: 0.462, V7: 0.239, V8: 0.098, V9: -0.664, V10: 0.463,
  V11: -0.931, V12: -2.304, V13: 0.772, V14: -1.576, V15: -0.230,
  V16: -0.050, V17: -0.844, V18: -0.380, V19: 0.597, V20: -0.697,
  V21: -0.055, V22: -0.270, V23: -0.233, V24: 0.140, V25: -0.052,
  V26: 0.265, V27: 0.825, V28: -0.068, Amount: 149.52, Time: 40680,
};

const SAMPLE_FRAUD = {
  V1: -3.043, V2: 3.033, V3: -1.833, V4: 2.531, V5: -3.136,
  V6: 1.874, V7: -4.731, V8: 3.201, V9: 1.716, V10: -1.223,
  V11: 2.301, V12: -2.842, V13: -2.011, V14: 2.671, V15: -1.099,
  V16: -3.219, V17: 3.914, V18: -1.915, V19: 1.199, V20: -0.434,
  V21: 0.570, V22: -0.055, V23: -2.074, V24: 0.808, V25: -0.253,
  V26: 1.110, V27: 0.921, V28: -0.054, Amount: 999.99, Time: 85400,
};

export default function Dashboard() {
  const [transactions, setTransactions] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [formData, setFormData] = useState(SAMPLE_LEGIT);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [txRes, statsRes] = await Promise.all([
        axios.get(`${API_URL}/transactions?limit=20`),
        axios.get(`${API_URL}/transactions/stats`),
      ]);
      setTransactions(txRes.data);
      setStats(statsRes.data);
    } catch (e) {
      console.error("Failed to fetch data:", e);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await axios.post(`${API_URL}/transactions`, formData);
      setResult(res.data);
      fetchData();
    } catch (e: any) {
      setError(e?.response?.data?.message || e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (type: "legit" | "fraud") => {
    setFormData(type === "legit" ? SAMPLE_LEGIT : SAMPLE_FRAUD);
    setResult(null);
  };

  return (
    <>
      <Head>
        <title>Fraud Detection Dashboard</title>
      </Head>

      <nav className="navbar">
        <h1>
          <Shield size={20} style={{ display: "inline", marginRight: 8 }} />
          Fraud Shield — Real-time Detection
        </h1>
        <span className="badge">System Online</span>
      </nav>

      <main className="container">
        {/* Stats */}
        {stats && (
          <div className="stats-grid">
            <div className="stat-card total">
              <div className="label">
                <Activity size={14} style={{ display: "inline", marginRight: 4 }} />
                Total Transactions
              </div>
              <div className="value">{stats.total_transactions.toLocaleString()}</div>
            </div>
            <div className="stat-card fraud">
              <div className="label">
                <AlertTriangle size={14} style={{ display: "inline", marginRight: 4 }} />
                Fraud Detected
              </div>
              <div className="value">{stats.fraud_count}</div>
            </div>
            <div className="stat-card rate">
              <div className="label">
                <TrendingUp size={14} style={{ display: "inline", marginRight: 4 }} />
                Fraud Rate
              </div>
              <div className="value">{stats.fraud_rate.toFixed(2)}%</div>
            </div>
            <div className="stat-card total">
              <div className="label">Avg Fraud Probability</div>
              <div className="value">{(stats.avg_fraud_probability * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}

        {/* Prediction Form */}
        <div className="panel">
          <h2>Test Transaction</h2>

          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
            <button className="btn btn-primary" onClick={() => loadSample("legit")}>
              Load Legit Sample
            </button>
            <button className="btn btn-danger" onClick={() => loadSample("fraud")}>
              Load Fraud Sample
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"].map((v) => (
                <div key={v}>
                  <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: 2, display: "block" }}>{v}</label>
                  <input
                    type="number"
                    step="any"
                    value={formData[v as keyof typeof formData]}
                    onChange={(e) => setFormData({ ...formData, [v]: parseFloat(e.target.value) })}
                    required
                  />
                </div>
              ))}
              <div>
                <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: 2, display: "block" }}>Amount ($)</label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.Amount}
                  onChange={(e) => setFormData({ ...formData, Amount: parseFloat(e.target.value) })}
                  required
                />
              </div>
              <div>
                <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: 2, display: "block" }}>Time (s)</label>
                <input
                  type="number"
                  value={formData.Time}
                  onChange={(e) => setFormData({ ...formData, Time: parseFloat(e.target.value) })}
                  required
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? "Analyzing..." : "Predict Fraud"}
            </button>
          </form>

          {error && <div className="error">{error}</div>}

          {result && (
            <div className={`prediction-result ${result.is_fraud ? "fraud" : "legit"}`}>
              <strong>{result.is_fraud ? "FRAUD DETECTED" : "Transaction Legitimate"}</strong>
              <br />
              Fraud Probability: {(result.fraud_probability * 100).toFixed(2)}%
              &nbsp;·&nbsp; Confidence: {result.confidence.toUpperCase()}
            </div>
          )}
        </div>

        {/* Transaction History */}
        <div className="panel">
          <h2>Recent Transactions</h2>
          {transactions.length === 0 ? (
            <div className="loading">No transactions yet. Submit one above!</div>
          ) : (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Amount</th>
                    <th>Fraud Prob</th>
                    <th>Status</th>
                    <th>Confidence</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((tx) => (
                    <tr key={tx.id}>
                      <td style={{ fontFamily: "monospace", fontSize: "0.75rem" }}>{tx.id.slice(0, 8)}...</td>
                      <td>${parseFloat(tx.amount).toFixed(2)}</td>
                      <td>{(tx.fraud_probability * 100).toFixed(2)}%</td>
                      <td>
                        <span className={tx.is_fraud ? "badge-fraud" : "badge-legit"}>
                          {tx.is_fraud ? "FRAUD" : "LEGIT"}
                        </span>
                      </td>
                      <td>{tx.confidence}</td>
                      <td>{new Date(tx.created_at).toLocaleTimeString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </>
  );
}
