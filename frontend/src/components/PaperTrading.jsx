import { useState } from "react";

export default function PaperTrading() {
  const [balance, setBalance] = useState(100000); // 1 Lakh starting cash
  const [portfolio, setPortfolio] = useState([]);

  return (
    <div className="trading-platform">
      <header className="hero">
        <div>
          <h2>Paper Trading Terminal</h2>
          <p>Execute trades with zero risk using virtual capital.</p>
        </div>
        <div className="hero-badge" style={{ background: 'var(--up)', color: 'white' }}>
          Virtual Balance: ₹{balance.toLocaleString()}
        </div>
      </header>

      <div className="dual-grid" style={{ marginTop: '20px' }}>
        <section className="panel">
          <h3>Quick Order</h3>
          <div className="search-grid" style={{ gridTemplateColumns: '1fr 1fr auto', marginTop: '15px' }}>
            <input placeholder="Ticker (e.g. RELIANCE)" />
            <input type="number" placeholder="Quantity" />
            <button className="btn primary">Buy</button>
          </div>
        </section>

        <section className="panel">
          <h3>Your Holdings</h3>
          {portfolio.length === 0 ? (
            <div className="empty small">No active positions.</div>
          ) : (
            /* Table for holdings */
            null
          )}
        </section>
      </div>
    </div>
  );
}