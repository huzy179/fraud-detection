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

// Real samples extracted from creditcard.csv test set
const SAMPLE_LEGIT = {
  V1: -0.674466, V2: 1.408105, V3: -1.110622, V4: -1.328366, V5: 1.388996,
  V6: -1.308439, V7: 1.885879, V8: -0.614233, V9: 0.311652, V10: 0.650757,
  V11: -0.857785, V12: -0.229961, V13: -0.199817, V14: 0.266371, V15: -0.046544,
  V16: -0.741398, V17: -0.605617, V18: -0.392568, V19: -0.162648, V20: 0.394322,
  V21: 0.080084, V22: 0.810034, V23: -0.224327, V24: 0.707899, V25: -0.135837,
  V26: 0.045102, V27: 0.533837, V28: 0.291319, Amount: 23.00, Time: 160760.00,
};

const SAMPLE_FRAUD = {
  V1: -1.271244, V2: 2.462675, V3: -2.851395, V4: 2.324480, V5: -1.372245,
  V6: -0.948196, V7: -3.065234, V8: 1.166927, V9: -2.268771, V10: -4.881143,
  V11: 2.255147, V12: -4.686387, V13: 0.652375, V14: -6.174288, V15: 0.594380,
  V16: -4.849692, V17: -6.536521, V18: -3.119094, V19: 1.715494, V20: 0.560478,
  V21: 0.652941, V22: 0.081931, V23: -0.221348, V24: -0.523582, V25: 0.224228,
  V26: 0.756335, V27: 0.632800, V28: 0.250187, Amount: 0.01, Time: 57007.00,
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
