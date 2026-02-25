import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import AuthPage from "./Auth";
import { getForecast, getSymbols } from "./api";

// constants

const horizons = [
  { key: "1d",  label: "1 Day",   subtitle: "Minute-wise intraday forecast" },
  { key: "15d", label: "15 Days", subtitle: "Hour-wise movement forecast"   },
  { key: "30d", label: "30 Days", subtitle: "Day-wise curve with high/low"  },
];

const initialForm = {
  ticker: "RELIANCE",
  horizon: "1d",
  data_source: "auto",
  local_data_dir: "",
};

const INITIAL_BALANCE = 100000;

// heplers

function inr(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function pct(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${value.toFixed(2)}%`;
}

function formatDate(date) {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit", month: "short", year: "numeric",
  }).format(date);
}

function formatTime(date) {
  return new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit", minute: "2-digit", hour12: true,
  }).format(date);
}

function weatherLabel(code) {
  const map = {
    0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime Fog", 51: "Light Drizzle", 53: "Drizzle",
    55: "Heavy Drizzle", 61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 80: "Showers", 95: "Thunder",
  };
  return map[code] || "Weather";
}

function saveLocal(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function loadLocal(key, fallback = null) {
  const raw = localStorage.getItem(key);
  if (!raw) return fallback;
  try { return JSON.parse(raw); } catch { return fallback; }
}

// paper trading panel

function PaperTradingPanel({ ticker, lastPrice, forecastPrice, projectedChangePct }) {
  const [portfolio, setPortfolio] = useState(() =>
    loadLocal("paper_portfolio", { cash: INITIAL_BALANCE, positions: {}, trades: [] })
  );
  const [order, setOrder] = useState({ side: "BUY", qty: "1", price: "" });
  const [toast, setToast] = useState(null);

  const displayPrice  = Number(order.price) || lastPrice || 0;

  useEffect(() => { saveLocal("paper_portfolio", portfolio); }, [portfolio]);

  useEffect(() => {
    if (lastPrice) setOrder((p) => ({ ...p, price: lastPrice.toFixed(2) }));
  }, [lastPrice]);

  const position      = portfolio.positions[ticker] || { qty: 0, avg: 0 };
  const positionValue = position.qty * displayPrice;
  const pnl           = position.qty > 0 ? (displayPrice - position.avg) * position.qty : 0;
  const pnlPct        = position.qty > 0 && position.avg > 0
    ? ((displayPrice - position.avg) / position.avg) * 100 : 0;
  const totalValue    = portfolio.cash + positionValue;
  const totalPnl      = totalValue - INITIAL_BALANCE;
  const totalPnlPct   = ((totalValue - INITIAL_BALANCE) / INITIAL_BALANCE) * 100;

  const suggestion = useMemo(() => {
    if (!projectedChangePct) return null;
    if (projectedChangePct > 1)  return { action: "BUY",  reason: `Forecast shows +${projectedChangePct.toFixed(2)}% upside`,  color: "#10b981" };
    if (projectedChangePct < -1) return { action: "SELL", reason: `Forecast shows ${projectedChangePct.toFixed(2)}% downside`, color: "#ef4444" };
    return { action: "HOLD", reason: "Forecast shows sideways movement", color: "#f59e0b" };
  }, [projectedChangePct]);

  function showToast(msg, type = "success") {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 2800);
  }

  function executeTrade() {
    const qty   = Math.max(1, Number(order.qty  || 0));
    const price = Number(order.price || 0);
    if (!price || price <= 0) { showToast("Enter a valid price.", "error"); return; }
    const cost = qty * price;

    if (order.side === "BUY" && portfolio.cash < cost) {
      showToast(`Insufficient balance. Need ₹${inr(cost)}, have ₹${inr(portfolio.cash)}`, "error");
      return;
    }
    if (order.side === "SELL" && position.qty < qty) {
      showToast(`Cannot sell ${qty} shares. You only hold ${position.qty}.`, "error");
      return;
    }

    const nextPositions = { ...portfolio.positions };
    let nextCash = portfolio.cash;
    let nextAvg  = position.avg;
    let nextQty  = position.qty;

    if (order.side === "BUY") {
      nextCash -= cost;
      nextAvg   = (position.avg * position.qty + cost) / (position.qty + qty);
      nextQty   = position.qty + qty;
      showToast(`Bought ${qty} shares of ${ticker} @ ₹${inr(price)}`);
    } else {
      const sellValue   = qty * price;
      const realizedPnl = sellValue - qty * position.avg;
      nextCash  += sellValue;
      nextQty    = position.qty - qty;
      nextAvg    = nextQty === 0 ? 0 : position.avg;
      showToast(
        realizedPnl >= 0
          ? `Sold ${qty} x ${ticker} @ ₹${inr(price)} | Profit: ₹${inr(realizedPnl)}`
          : `Sold ${qty} x ${ticker} @ ₹${inr(price)} | Loss: ₹${inr(Math.abs(realizedPnl))}`
      );
    }

    nextPositions[ticker] = { qty: nextQty, avg: nextAvg };

    const trade = {
      id:   `${Date.now()}`,
      side: order.side,
      qty,
      price,
      ticker,
      time: new Date().toLocaleString("en-IN"),
      pnl:  order.side === "SELL" ? (qty * price) - (qty * position.avg) : null,
    };

    setPortfolio({
      cash:      nextCash,
      positions: nextPositions,
      trades:    [trade, ...portfolio.trades].slice(0, 12),
    });
  }

  function resetPortfolio() {
    if (!window.confirm("Reset paper trading portfolio to ₹1,00,000? This cannot be undone.")) return;
    setPortfolio({ cash: INITIAL_BALANCE, positions: {}, trades: [] });
    showToast("Portfolio reset to ₹1,00,000");
  }

  const orderCost = Number(order.qty || 0) * (Number(order.price) || lastPrice || 0);

  return (
    <div className="panel paper-panel">
      {toast && (
        <div className={`trade-toast ${toast.type === "error" ? "toast-error" : "toast-success"}`}>
          {toast.msg}
        </div>
      )}

      <div className="panel-head">
        <div>
          <h3>Paper Trading</h3>
          <span>Risk-free virtual trading with AI forecasts</span>
        </div>
        <button className="btn-ghost-sm" type="button" onClick={resetPortfolio}>Reset Portfolio</button>
      </div>

      {suggestion && (
        <div className="suggestion-banner" style={{ borderColor: suggestion.color }}>
          <div className="suggestion-icon" style={{ background: suggestion.color }}>
            {suggestion.action === "BUY" ? "↑" : suggestion.action === "SELL" ? "↓" : "→"}
          </div>
          <div>
            <strong style={{ color: suggestion.color }}>AI Suggests: {suggestion.action}</strong>
            <p>{suggestion.reason}</p>
          </div>
          {suggestion.action !== "HOLD" && (
            <button
              className="btn-suggestion"
              style={{ borderColor: suggestion.color, color: suggestion.color }}
              type="button"
              onClick={() => setOrder((p) => ({ ...p, side: suggestion.action }))}
            >
              Use This
            </button>
          )}
        </div>
      )}

      <div className="paper-body">
        <div className="paper-overview">
          <div className="overview-card">
            <p>Total Portfolio Value</p>
            <h2>₹{inr(totalValue)}</h2>
            <span className={totalPnl >= 0 ? "badge-up" : "badge-down"}>
              {totalPnl >= 0 ? "+" : ""}₹{inr(totalPnl)} ({totalPnl >= 0 ? "+" : ""}{totalPnlPct.toFixed(2)}%)
            </span>
          </div>
          <div className="overview-metrics">
            <div className="om-cell">
              <p>Cash Available</p>
              <h4>₹{inr(portfolio.cash)}</h4>
            </div>
            <div className="om-cell">
              <p>{ticker} Holdings</p>
              <h4>{position.qty} shares</h4>
              {position.qty > 0 && <small>Avg ₹{inr(position.avg)}</small>}
            </div>
            <div className="om-cell">
              <p>Unrealized P&L</p>
              <h4 className={pnl >= 0 ? "up" : "down"}>
                {pnl >= 0 ? "+" : ""}₹{inr(pnl)}
              </h4>
              {position.qty > 0 && (
                <small className={pnlPct >= 0 ? "up" : "down"}>
                  {pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(2)}%
                </small>
              )}
            </div>
            <div className="om-cell">
              <p>Current Price</p>
              <h4>₹{inr(lastPrice)}</h4>
              {forecastPrice && <small>Forecast: ₹{inr(forecastPrice)}</small>}
            </div>
          </div>
        </div>

        <div className="paper-order-section">
          <div className="order-form-wrap">
            <div className="order-side-tabs">
              <button
                type="button"
                className={`side-tab buy ${order.side === "BUY" ? "active" : ""}`}
                onClick={() => setOrder((p) => ({ ...p, side: "BUY" }))}
              >BUY</button>
              <button
                type="button"
                className={`side-tab sell ${order.side === "SELL" ? "active" : ""}`}
                onClick={() => setOrder((p) => ({ ...p, side: "SELL" }))}
              >SELL</button>
            </div>

            <div className="order-fields">
              <label>
                <span>Quantity</span>
                <div className="qty-row">
                  <button type="button" className="qty-btn" onClick={() => setOrder((p) => ({ ...p, qty: String(Math.max(1, Number(p.qty) - 1)) }))}>-</button>
                  <input
                    value={order.qty}
                    onChange={(e) => setOrder((p) => ({ ...p, qty: e.target.value }))}
                    type="number" min="1"
                  />
                  <button type="button" className="qty-btn" onClick={() => setOrder((p) => ({ ...p, qty: String(Number(p.qty) + 1) }))}>+</button>
                </div>
              </label>
              <label>
                <span>Price per Share (₹)</span>
                <input
                  value={order.price}
                  onChange={(e) => setOrder((p) => ({ ...p, price: e.target.value }))}
                  type="number" step="0.01"
                />
              </label>
            </div>

            <div className="order-summary">
              <div className="order-summary-row">
                <span>Order Value</span>
                <strong>₹{inr(orderCost)}</strong>
              </div>
              {order.side === "BUY" && (
                <div className="order-summary-row">
                  <span>Balance After</span>
                  <strong className={portfolio.cash - orderCost < 0 ? "down" : ""}>
                    ₹{inr(portfolio.cash - orderCost)}
                  </strong>
                </div>
              )}
              {order.side === "SELL" && position.qty > 0 && (
                <div className="order-summary-row">
                  <span>Est. P&L</span>
                  <strong className={(Number(order.price || 0) - position.avg) >= 0 ? "up" : "down"}>
                    {((Number(order.price || 0) - position.avg) * Number(order.qty || 0)) >= 0 ? "+" : ""}
                    ₹{inr((Number(order.price || 0) - position.avg) * Number(order.qty || 0))}
                  </strong>
                </div>
              )}
            </div>

            <button
              className={`btn-execute ${order.side === "BUY" ? "execute-buy" : "execute-sell"}`}
              type="button"
              onClick={executeTrade}
            >
              Place {order.side} Order - {ticker}
            </button>
          </div>

          <div className="paper-trades">
            <h4>Trade History</h4>
            {portfolio.trades.length === 0 ? (
              <div className="empty-trades">
                <p>No trades yet. Place your first paper trade!</p>
              </div>
            ) : (
              <div className="trades-list">
                {portfolio.trades.map((trade) => (
                  <div key={trade.id} className="trade-row">
                    <div className={`trade-badge ${trade.side === "BUY" ? "badge-buy" : "badge-sell"}`}>
                      {trade.side}
                    </div>
                    <div className="trade-detail">
                      <strong>{trade.qty} x {trade.ticker}</strong>
                      <span>@ ₹{inr(trade.price)}</span>
                    </div>
                    {trade.pnl !== null && (
                      <div className={`trade-pnl ${trade.pnl >= 0 ? "up" : "down"}`}>
                        {trade.pnl >= 0 ? "+" : ""}₹{inr(trade.pnl)}
                      </div>
                    )}
                    <div className="trade-time">{trade.time}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .paper-panel { position: relative; overflow: visible; }
        .trade-toast {
          position: absolute; top: -16px; left: 50%; transform: translateX(-50%);
          padding: 10px 22px; border-radius: 30px; font-size: 0.84rem; font-weight: 600;
          z-index: 100; white-space: nowrap; animation: toastIn 0.3s ease;
          box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .toast-success { background: #10b981; color: #fff; }
        .toast-error   { background: #ef4444; color: #fff; }
        @keyframes toastIn {
          from { opacity:0; transform:translateX(-50%) translateY(8px); }
          to   { opacity:1; transform:translateX(-50%) translateY(0); }
        }
        .suggestion-banner {
          display: flex; align-items: center; gap: 12px; padding: 12px 16px;
          border-radius: 12px; border: 1.5px solid; background: rgba(255,255,255,0.03);
          margin-bottom: 20px;
        }
        .suggestion-icon {
          width: 36px; height: 36px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; color: #fff;
          font-size: 1.1rem; font-weight: 900; flex-shrink: 0;
        }
        .suggestion-banner strong { display: block; font-size: 0.9rem; }
        .suggestion-banner p { font-size: 0.78rem; opacity: 0.7; margin: 2px 0 0; }
        .btn-suggestion {
          margin-left: auto; padding: 6px 14px; border-radius: 8px; border: 1.5px solid;
          background: transparent; font-size: 0.8rem; font-weight: 700; cursor: pointer;
          white-space: nowrap; transition: all 0.2s;
        }
        .btn-suggestion:hover { opacity: 0.8; }
        .panel-head { display: flex; align-items: flex-start; justify-content: space-between; }
        .btn-ghost-sm {
          padding: 6px 14px; border-radius: 8px; border: 1.5px solid var(--border, #dde5f0);
          background: transparent; font-size: 0.78rem; cursor: pointer; color: inherit;
          opacity: 0.7; transition: opacity 0.2s;
        }
        .btn-ghost-sm:hover { opacity: 1; }
        .paper-overview { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .overview-card {
          background: linear-gradient(135deg, #0e7ce9, #1ba97a);
          border-radius: 16px; padding: 20px 24px; color: #fff; min-width: 200px; flex-shrink: 0;
        }
        .overview-card p  { margin: 0; opacity: 0.85; font-size: 0.8rem; }
        .overview-card h2 { margin: 4px 0 8px; font-size: 1.6rem; }
        .badge-up   { background: rgba(255,255,255,0.2); border-radius: 20px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700; }
        .badge-down { background: rgba(255,80,80,0.25);  border-radius: 20px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700; }
        .overview-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; flex: 1; }
        .om-cell {
          background: var(--card-bg, rgba(255,255,255,0.05));
          border: 1px solid var(--border, rgba(0,0,0,0.08));
          border-radius: 12px; padding: 14px 16px;
        }
        .om-cell p     { margin: 0 0 4px; font-size: 0.75rem; opacity: 0.6; }
        .om-cell h4    { margin: 0; font-size: 1.05rem; }
        .om-cell small { font-size: 0.72rem; opacity: 0.6; }
        .paper-order-section { display: grid; grid-template-columns: 320px 1fr; gap: 20px; }
        @media (max-width: 700px) {
          .paper-order-section { grid-template-columns: 1fr; }
          .overview-metrics    { grid-template-columns: 1fr 1fr; }
        }
        .order-form-wrap {
          background: var(--card-bg, rgba(255,255,255,0.04));
          border: 1px solid var(--border, rgba(0,0,0,0.08));
          border-radius: 16px; padding: 20px;
        }
        .order-side-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
        .side-tab {
          flex: 1; padding: 10px; border-radius: 10px; border: 2px solid transparent;
          font-weight: 700; font-size: 0.88rem; cursor: pointer; transition: all 0.2s; background: transparent;
        }
        .side-tab.buy        { color: #10b981; border-color: #10b981; }
        .side-tab.buy.active { background: #10b981; color: #fff; }
        .side-tab.sell        { color: #ef4444; border-color: #ef4444; }
        .side-tab.sell.active { background: #ef4444; color: #fff; }
        .order-fields { display: flex; flex-direction: column; gap: 12px; margin-bottom: 14px; }
        .order-fields label { display: flex; flex-direction: column; gap: 5px; }
        .order-fields span  { font-size: 0.75rem; font-weight: 600; opacity: 0.6; }
        .order-fields input {
          padding: 9px 12px; border-radius: 8px;
          border: 1.5px solid var(--border, #dde5f0);
          background: var(--input-bg, rgba(255,255,255,0.05));
          font-size: 0.9rem; width: 100%; box-sizing: border-box; color: inherit;
        }
        .qty-row { display: flex; gap: 6px; align-items: center; }
        .qty-row input { flex: 1; text-align: center; }
        .qty-btn {
          width: 34px; height: 34px; border-radius: 8px;
          border: 1.5px solid var(--border, #dde5f0); background: transparent;
          font-size: 1.1rem; cursor: pointer; color: inherit;
          display: flex; align-items: center; justify-content: center; transition: background 0.15s;
        }
        .qty-btn:hover { background: var(--hover, rgba(0,0,0,0.06)); }
        .order-summary {
          border-top: 1px solid var(--border, #dde5f0); padding-top: 12px;
          margin-bottom: 14px; display: flex; flex-direction: column; gap: 6px;
        }
        .order-summary-row { display: flex; justify-content: space-between; font-size: 0.82rem; }
        .order-summary-row span { opacity: 0.6; }
        .btn-execute {
          width: 100%; padding: 13px; border-radius: 12px; border: none;
          font-size: 0.9rem; font-weight: 700; cursor: pointer; transition: all 0.2s; letter-spacing: 0.02em;
        }
        .execute-buy       { background: #10b981; color: #fff; }
        .execute-buy:hover { background: #059669; }
        .execute-sell       { background: #ef4444; color: #fff; }
        .execute-sell:hover { background: #dc2626; }
        .paper-trades h4 { margin: 0 0 12px; font-size: 0.88rem; opacity: 0.7; }
        .empty-trades {
          display: flex; flex-direction: column; align-items: center;
          justify-content: center; height: 160px; opacity: 0.45;
        }
        .empty-trades p { font-size: 0.82rem; }
        .trades-list { display: flex; flex-direction: column; gap: 8px; max-height: 280px; overflow-y: auto; }
        .trade-row {
          display: flex; align-items: center; gap: 10px; padding: 10px 12px;
          border-radius: 10px; background: var(--card-bg, rgba(255,255,255,0.04));
          border: 1px solid var(--border, rgba(0,0,0,0.07));
        }
        .trade-badge { padding: 3px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 800; }
        .badge-buy  { background: rgba(16,185,129,0.15); color: #10b981; }
        .badge-sell { background: rgba(239,68,68,0.12);  color: #ef4444; }
        .trade-detail { flex: 1; }
        .trade-detail strong { display: block; font-size: 0.83rem; }
        .trade-detail span   { font-size: 0.75rem; opacity: 0.6; }
        .trade-pnl  { font-size: 0.82rem; font-weight: 700; }
        .trade-time { font-size: 0.7rem; opacity: 0.45; white-space: nowrap; }
        .up   { color: #10b981; }
        .down { color: #ef4444; }
      `}</style>
    </div>
  );
}

// forecast dashboard

function ForecastDashboard({ user, profile, theme, setTheme, onLogout }) {
  const [form, setForm]                     = useState(initialForm);
  const [symbols, setSymbols]               = useState([]);
  const [forecast, setForecast]             = useState(null);
  const [loading, setLoading]               = useState(false);
  const [loadingSymbols, setLoadingSymbols] = useState(false);
  const [error, setError]                   = useState("");

  const [now, setNow]           = useState(new Date());
  const [location, setLocation] = useState({ city: "", region: "", status: "idle" });
  const [weather, setWeather]   = useState({ temp: null, wind: null, code: null });

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!navigator.geolocation) { setLocation((p) => ({ ...p, status: "blocked" })); return; }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        setLocation((p) => ({ ...p, status: "ready" }));
        try {
          const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/reverse?latitude=${latitude}&longitude=${longitude}&count=1`);
          const geo    = await geoRes.json();
          const place  = geo?.results?.[0];
          setLocation({ city: place?.name || "", region: place?.admin1 || "", status: "ready" });
        } catch { setLocation((p) => ({ ...p, status: "ready" })); }
        try {
          const wxRes   = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,weathercode,windspeed_10m`);
          const wx      = await wxRes.json();
          const current = wx?.current || {};
          setWeather({ temp: current.temperature_2m, wind: current.windspeed_10m, code: current.weathercode });
        } catch { setWeather({ temp: null, wind: null, code: null }); }
      },
      () => setLocation((p) => ({ ...p, status: "blocked" }))
    );
  }, []);

  useEffect(() => {
    async function loadSymbols() {
      if (form.data_source === "yfinance") { setSymbols([]); return; }
      setLoadingSymbols(true);
      try {
        const data = await getSymbols({ data_source: form.data_source, local_data_dir: form.local_data_dir || undefined });
        setSymbols(data.symbols || []);
      } catch { setSymbols([]); }
      finally { setLoadingSymbols(false); }
    }
    loadSymbols();
  }, [form.data_source, form.local_data_dir]);

  const chartData = useMemo(() => {
    if (!forecast) return [];
    const history   = forecast.history.map((p) => ({ date: p.date, history: p.value, forecast: null, high: null, low: null }));
    const predicted = forecast.forecast.map((p) => ({ date: p.date, history: null, forecast: p.value, high: p.high ?? null, low: p.low ?? null }));
    return [...history, ...predicted];
  }, [forecast]);

  const yDomain = useMemo(() => {
    if (chartData.length === 0) return ["auto", "auto"];
    const values = [];
    for (const row of chartData) {
      for (const key of ["history", "forecast", "high", "low"]) {
        const v = row[key];
        if (typeof v === "number" && Number.isFinite(v)) values.push(v);
      }
    }
    if (values.length === 0) return ["auto", "auto"];
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { const pad = Math.max(Math.abs(min) * 0.02, 1); return [min - pad, max + pad]; }
    const range = max - min, pad = Math.max(range * 0.18, Math.abs(max) * 0.003, 0.75);
    return [min - pad, max + pad];
  }, [chartData]);

  const forecastRows = useMemo(() => {
    if (!forecast) return [];
    return forecast.forecast.map((p, idx) => {
      const delta     = p.value - forecast.last_close;
      const changePct = forecast.last_close === 0 ? 0 : (delta / forecast.last_close) * 100;
      return { idx: idx + 1, date: p.date, value: p.value, high: p.high, low: p.low, delta, changePct };
    });
  }, [forecast]);

  const activeHorizon = horizons.find((h) => h.key === form.horizon);

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const result = await getForecast({
        ticker: form.ticker, horizon: form.horizon,
        data_source: form.data_source, local_data_dir: form.local_data_dir || null,
      });
      setForecast(result);
      saveLocal("last_forecast", result);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Forecast request failed");
      setForecast(null);
    } finally { setLoading(false); }
  }

  return (
    <div className="app-shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      <aside className="sidebar">
        <div className="brand">
          <div className="logo">K</div>
          <div><h1>Kavout</h1><p>Enterprise Forecasting Console</p></div>
        </div>

        <section className="side-block">
          <h2>Prediction Horizons</h2>
          {horizons.map((h) => (
            <button key={h.key} type="button"
              className={`side-horizon ${form.horizon === h.key ? "active" : ""}`}
              onClick={() => setForm((p) => ({ ...p, horizon: h.key }))}
            >
              <strong>{h.label}</strong>
              <span>{h.subtitle}</span>
            </button>
          ))}
        </section>

        <section className="side-block">
          <h2>Quick Companies</h2>
          {loadingSymbols && <p className="muted">Loading...</p>}
          <div className="symbol-list">
            {symbols.slice(0, 18).map((symbol) => (
              <button key={symbol} type="button" className="symbol-pill"
                onClick={() => setForm((p) => ({ ...p, ticker: symbol }))}
              >{symbol}</button>
            ))}
          </div>
        </section>

        {/* Logout button at the bottom of sidebar */}
        <div className="sidebar-footer">
          <button type="button" className="btn-logout" onClick={onLogout}>
            ⎋ &nbsp;Log Out
          </button>
        </div>

        <style>{`
          .sidebar-footer {
            margin-top: auto;
            padding: 16px;
            border-top: 1px solid var(--border, rgba(255,255,255,0.08));
          }
          .btn-logout {
            width: 100%;
            padding: 10px 14px;
            border-radius: 10px;
            border: 1.5px solid rgba(239,68,68,0.35);
            background: rgba(239,68,68,0.07);
            color: #ef4444;
            font-size: 0.82rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
          }
          .btn-logout:hover {
            background: rgba(239,68,68,0.15);
            border-color: #ef4444;
          }
        `}</style>
      </aside>

      <main className="main">
        <header className="hero">
          <div>
            <h2>Live-Style Multi-Horizon Forecasting</h2>
            <p>Search a company, choose 1D/15D/30D, and get stacked-model forecasts generated entirely by the backend.</p>
          </div>
          <div className="hero-badge">{activeHorizon?.label}</div>
        </header>

        <section className="status-bar">
          <div>
            <p className="muted">Welcome</p>
            <h3>{user.name}</h3>
            {profile?.city && <span className="subline">{profile.city}</span>}
          </div>
          <div>
            <p className="muted">Local Time</p>
            <h3>{formatTime(now)}</h3>
            <span className="subline">{formatDate(now)}</span>
          </div>
          <div>
            <p className="muted">Location</p>
            <h3>{location.city ? `${location.city}, ${location.region}` : "Bhubaneswar, IN"}</h3>
            <span className="subline">{location.status === "blocked" ? "Enable location services" : "GPS-based"}</span>
          </div>
          <div>
            <p className="muted">Weather</p>
            <h3>{weather.temp !== null ? `${weather.temp}°C` : "--"}</h3>
            <span className="subline">{weather.code !== null ? weatherLabel(weather.code) : "Waiting for data"}</span>
          </div>
          <div className="toggle">
            <span>Theme</span>
            <button type="button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
              {theme === "dark" ? "Night" : "Day"}
            </button>
          </div>
        </section>

        <section className="search-card">
          <form className="search-grid" onSubmit={onSubmit}>
            <label>
              <span>Company / Ticker</span>
              <input
                value={form.ticker}
                onChange={(e) => setForm((p) => ({ ...p, ticker: e.target.value.toUpperCase() }))}
                list="symbol-options" placeholder="Type RELIANCE, TCS, INFY..." required
              />
            </label>
            <label>
              <span>Data Source</span>
              <select value={form.data_source} onChange={(e) => setForm((p) => ({ ...p, data_source: e.target.value }))}>
                <option value="auto">auto</option>
                <option value="local">local</option>
                <option value="yfinance">yfinance</option>
              </select>
            </label>
            <label>
              <span>Local Data Folder (optional)</span>
              <input
                value={form.local_data_dir}
                onChange={(e) => setForm((p) => ({ ...p, local_data_dir: e.target.value }))}
                placeholder="backend/data"
              />
            </label>
            <button className="btn primary" type="submit" disabled={loading}>
              {loading ? "Forecasting..." : `Predict ${activeHorizon?.label}`}
            </button>
          </form>
        </section>

        {error && <div className="alert"><strong>Request failed:</strong> {error}</div>}

        <section className="metric-grid">
          <article className="metric">
            <p>Last Close</p>
            <h3>{forecast ? `₹${inr(forecast.last_close)}` : "--"}</h3>
          </article>
          <article className="metric">
            <p>Projected Close</p>
            <h3>{forecast ? `₹${inr(forecast.projected_close)}` : "--"}</h3>
          </article>
          <article className="metric">
            <p>Projected Change</p>
            <h3 className={forecast && forecast.projected_change_pct >= 0 ? "up" : "down"}>
              {forecast ? pct(forecast.projected_change_pct) : "--"}
            </h3>
          </article>
          <article className="metric">
            <p>Granularity</p>
            <h3>{forecast ? forecast.granularity : "--"}</h3>
          </article>
        </section>

        <PaperTradingPanel
          ticker={form.ticker}
          lastPrice={forecast?.last_close}
          forecastPrice={forecast?.projected_close}
          projectedChangePct={forecast?.projected_change_pct}
        />

        <section className="panel chart-panel">
          <div className="panel-head">
            <h3>Forecast Curve</h3>
            <span>
              {form.horizon === "1d"  && "Every minute for one trading day"}
              {form.horizon === "15d" && "Hour-wise trajectory for 15 days"}
              {form.horizon === "30d" && "Day-wise curve with high/low range"}
            </span>
          </div>
          {chartData.length === 0 ? (
            <div className="empty">Run prediction to view chart output.</div>
          ) : (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={420}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#dde5f0" />
                  <XAxis dataKey="date" minTickGap={24} />
                  <YAxis domain={yDomain} tickFormatter={(v) => `₹${Math.round(v)}`} width={88} />
                  <Tooltip formatter={(v) => `₹${inr(Number(v))}`} />
                  <Legend />
                  <Line type="linear" dataKey="history"  stroke="#0e7ce9" strokeWidth={2.4} dot={false} name="History" />
                  <Line type="linear" dataKey="forecast" stroke="#ef9d13" strokeWidth={3.1} dot={false} name="Forecast" />
                  {form.horizon === "30d" && (
                    <>
                      <Line type="linear" dataKey="high" stroke="#1ba97a" strokeWidth={1.8} strokeDasharray="6 4" dot={false} name="Projected High" />
                      <Line type="linear" dataKey="low"  stroke="#d94d4d" strokeWidth={1.8} strokeDasharray="6 4" dot={false} name="Projected Low"  />
                    </>
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="dual-grid">
          <article className="panel">
            <div className="panel-head"><h3>Stacking Weights</h3></div>
            {!forecast ? <div className="empty small">No forecast yet.</div> : (
              <ul className="stat-list">
                <li><span>DLinear</span>      <strong>{(forecast.stack_weights.dlinear * 100).toFixed(2)}%</strong></li>
                <li><span>Linear Model</span> <strong>{(forecast.stack_weights.linear  * 100).toFixed(2)}%</strong></li>
                <li><span>NLinear Model</span><strong>{(forecast.stack_weights.nlinear * 100).toFixed(2)}%</strong></li>
              </ul>
            )}
          </article>
          <article className="panel">
            <div className="panel-head"><h3>Model Error (MSE)</h3></div>
            {!forecast ? <div className="empty small">No forecast yet.</div> : (
              <ul className="stat-list">
                <li><span>DLinear</span>      <strong>{forecast.model_mse.dlinear.toFixed(8)}</strong></li>
                <li><span>Linear Model</span> <strong>{forecast.model_mse.linear.toFixed(8)}</strong></li>
                <li><span>NLinear Model</span><strong>{forecast.model_mse.nlinear.toFixed(8)}</strong></li>
                <li><span>Stacked</span>      <strong>{forecast.model_mse.stacked.toFixed(8)}</strong></li>
              </ul>
            )}
          </article>
        </section>

        <section className="panel table-panel">
          <div className="panel-head"><h3>Forecast Table</h3></div>
          {!forecast ? <div className="empty">Forecast rows appear after prediction.</div> : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th><th>Timestamp</th><th>Forecast (₹)</th>
                    {form.horizon === "30d" && <th>High / Low (₹)</th>}
                    <th>Delta (₹)</th><th>Delta %</th>
                  </tr>
                </thead>
                <tbody>
                  {forecastRows.map((row) => (
                    <tr key={row.date}>
                      <td>{row.idx}</td>
                      <td>{row.date}</td>
                      <td>{inr(row.value)}</td>
                      {form.horizon === "30d" && <td>{inr(row.high)} / {inr(row.low)}</td>}
                      <td className={row.delta     >= 0 ? "up" : "down"}>{row.delta     >= 0 ? "+" : ""}{inr(row.delta)}</td>
                      <td className={row.changePct >= 0 ? "up" : "down"}>{row.changePct >= 0 ? "+" : ""}{pct(row.changePct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <datalist id="symbol-options">
        {symbols.map((s) => <option key={s} value={s} />)}
      </datalist>
    </div>
  );
}

// ─── ROOT ──────────────────────────────────────────────────────────────────────
//
//  SESSION STRATEGY
//  ─────────────────
//  • `authed` lives only in React state (not initialised from localStorage).
//    This means every fresh page load always starts at the AuthPage — exactly
//    what you want.  localStorage is only used to remember registered users
//    and their profiles across signups/logins, not to auto-skip auth.
//
//  • After a successful login or completed signup, authed is set to true and
//    the dashboard renders.
//
//  • Logout clears authed (and the active-session keys) and returns to AuthPage.
//
// ──────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [theme,   setTheme]   = useState(() => loadLocal("theme", "light"));

  //  authed starts as FALSE every page load → always shows AuthPage first
  const [authed,  setAuthed]  = useState(false);
  const [user,    setUser]    = useState(null);
  const [profile, setProfile] = useState(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    saveLocal("theme", theme);
  }, [theme]);

  // LOGOUT 
  function handleLogout() {
    localStorage.removeItem("user");
    localStorage.removeItem("profile");
    setUser(null);
    setProfile(null);
    setAuthed(false);
  }

  // LOGIN 
  function handleLogin(email, password) {
    const users = loadLocal("users", []);
    const match = users.find((u) => u.email === email && u.password === password);
    if (!match) return { error: "Invalid email or password." };
    if (match.demo && match.demoExpiry && new Date(match.demoExpiry) < new Date()) {
      return { error: "Demo access expired. Please contact support to upgrade." };
    }

    const storedProfile = loadLocal(`profile_${match.email}`, null);

    setUser(match);
    saveLocal("user", match);

    if (storedProfile) {
      setProfile(storedProfile);
      saveLocal("profile", storedProfile);
      setAuthed(true);          
      return null;
    }

    // Has account but no profile yet → AuthPage will advance to step 2
    return { needsProfile: true, user: match };
  }

  // SIGNUP (AuthPage calls this twice)
  function handleSignup(name, email, password, demo, profileData) {
    const users = loadLocal("users", []);

    // First call — profileData === null: register user, ask for profile next
    if (profileData === null) {
      if (users.some((u) => u.email === email)) {
        return { error: "Email already registered. Please log in." };
      }
      const demoExpiry = demo
        ? new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString()
        : null;
      const newUser = { name, email, password, demo, demoExpiry };
      saveLocal("users", [...users, newUser]);
      saveLocal("user",  newUser);
      setUser(newUser);
      return { needsProfile: true, user: newUser }; 
    }

    // Second call — profileData is the completed profile object
    const resolvedUser =
      users.find((u) => u.email === email) ||
      loadLocal("user", null) ||
      { name, email, password, demo, demoExpiry: null };

    const nextProfile = { ...profileData, email: resolvedUser.email };
    setProfile(nextProfile);
    saveLocal("profile", nextProfile);
    saveLocal(`profile_${resolvedUser.email}`, nextProfile);

    setUser(resolvedUser);
    saveLocal("user", resolvedUser);
    setAuthed(true);        
    return null;
  }

  // routing 
  if (!authed) {
    return (
      <AuthPage
        onLogin={handleLogin}
        onSignup={handleSignup}
      />
    );
  }

  return (
    <ForecastDashboard
      user={user}
      profile={profile}
      theme={theme}
      setTheme={setTheme}
      onLogout={handleLogout}
    />
  );
}