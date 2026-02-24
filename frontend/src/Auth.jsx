import { useEffect, useRef, useState } from "react";

/* tiny helpers */
function saveLocal(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}
function loadLocal(key, fallback = null) {
  const raw = localStorage.getItem(key);
  if (!raw) return fallback;
  try { return JSON.parse(raw); } catch { return fallback; }
}

/* animated ticker tape */
const TICKERS = [
  { sym: "RELIANCE", val: "₹2,847.30", chg: "+1.24%" },
  { sym: "TCS",      val: "₹3,912.55", chg: "+0.87%" },
  { sym: "INFY",     val: "₹1,548.20", chg: "-0.43%" },
  { sym: "HDFC",     val: "₹1,632.80", chg: "+2.11%" },
  { sym: "WIPRO",    val: "₹452.60",   chg: "+0.65%" },
  { sym: "BAJAJ",    val: "₹7,124.00", chg: "-1.02%" },
  { sym: "ADANI",    val: "₹2,390.15", chg: "+3.40%" },
  { sym: "NIFTY50",  val: "₹22,519",   chg: "+0.55%" },
  { sym: "SENSEX",   val: "₹74,119",   chg: "+0.49%" },
  { sym: "ONGC",     val: "₹261.45",   chg: "+1.78%" },
];

function TickerTape() {
  const items = [...TICKERS, ...TICKERS]; // duplicate for seamless loop
  return (
    <div className="ap-ticker-wrap">
      <div className="ap-ticker-track">
        {items.map((t, i) => (
          <span key={i} className="ap-ticker-item">
            <span className="ap-ticker-sym">{t.sym}</span>
            <span className="ap-ticker-val">{t.val}</span>
            <span className={`ap-ticker-chg ${t.chg.startsWith("+") ? "up" : "down"}`}>{t.chg}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* floating orbs canvas */
function OrbCanvas() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W = (canvas.width = canvas.offsetWidth);
    let H = (canvas.height = canvas.offsetHeight);
    const orbs = Array.from({ length: 6 }, (_, i) => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: 120 + Math.random() * 180,
      vx: (Math.random() - 0.5) * 0.28,
      vy: (Math.random() - 0.5) * 0.28,
      hue: [210, 180, 200, 165, 220, 190][i],
    }));

    let raf;
    function draw() {
      ctx.clearRect(0, 0, W, H);
      for (const o of orbs) {
        const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.r);
        g.addColorStop(0, `hsla(${o.hue},80%,62%,0.18)`);
        g.addColorStop(1, `hsla(${o.hue},80%,62%,0)`);
        ctx.beginPath();
        ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
        o.x += o.vx;
        o.y += o.vy;
        if (o.x < -o.r) o.x = W + o.r;
        if (o.x > W + o.r) o.x = -o.r;
        if (o.y < -o.r) o.y = H + o.r;
        if (o.y > H + o.r) o.y = -o.r;
      }
      raf = requestAnimationFrame(draw);
    }
    draw();

    const ro = new ResizeObserver(() => {
      W = canvas.width = canvas.offsetWidth;
      H = canvas.height = canvas.offsetHeight;
    });
    ro.observe(canvas);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={canvasRef} className="ap-orb-canvas" />;
}

/* grid dot pattern  */
function GridPattern() {
  return (
    <svg className="ap-grid-pattern" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="dots" x="0" y="0" width="28" height="28" patternUnits="userSpaceOnUse">
          <circle cx="1.5" cy="1.5" r="1.5" fill="currentColor" opacity="0.18" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#dots)" />
    </svg>
  );
}

/*  feature list for left panel */
const FEATURES = [
  {icon: "✔️", title: "Multi-Horizon Forecasts", desc: "1D, 15D, 30D predictions with minute/hour/day granularity" },
  {icon: "✔️", title: "Stacked AI Models",       desc: "DLinear, NLinear & Linear ensemble with live MSE tracking" },
  {icon: "✔️", title: "Paper Trading",            desc: "Risk-free virtual trades guided by AI price forecasts" },
  {icon: "✔️", title: "Live Market Context",      desc: "Real-time weather, location & time for informed decisions" },
];

/*  main auth page */
export default function AuthPage({ onLogin, onSignup }) {
  const [mode, setMode]     = useState("login");
  const [step, setStep]     = useState(1); // 1 = auth, 2 = profile
  const [error, setError]   = useState("");
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);
  const [pendingUser, setPendingUser] = useState(null);

  const [auth, setAuth] = useState({ name: "", email: "", password: "", demo: false });
  const [profile, setProfile] = useState({
    phone: "", city: "", experience: "Beginner", risk: "Moderate", capital: "100000",
  });

  function switchMode(m) {
    setMode(m);
    setError("");
  }

  async function handleAuth(e) {
    e.preventDefault();
    setError("");
    setLoading(true);

    await new Promise(r => setTimeout(r, 480)); // brief loading feel

    if (mode === "login") {
      const result = onLogin(auth.email, auth.password);
      if (result?.error) { setError(result.error); setLoading(false); return; }
    } else {
      if (!auth.name)     { setError("Please enter your full name."); setLoading(false); return; }
      if (!auth.email)    { setError("Email is required.");           setLoading(false); return; }
      if (auth.password.length < 6) { setError("Password must be at least 6 characters."); setLoading(false); return; }

      const result = onSignup(auth.name, auth.email, auth.password, auth.demo, null);
      if (result?.error) { setError(result.error); setLoading(false); return; }
      if (result?.needsProfile) {
        setPendingUser(result.user);
        setStep(2);
        setLoading(false);
        return;
      }
    }
    setLoading(false);
  }

  function handleProfile(e) {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      onSignup(auth.name, auth.email, auth.password, auth.demo, profile);
      setLoading(false);
    }, 400);
  }

  /* step 2: profile */
  if (step === 2) {
    return (
      <div className="ap-root">
        <OrbCanvas />
        <GridPattern />
        <TickerTape />
        <div className="ap-center">
          <div className="ap-card ap-card-profile">
            <div className="ap-card-header">
              <div className="ap-logo-mark">K</div>
              <div>
                <h2 className="ap-card-title">Complete your profile</h2>
                <p className="ap-card-sub">One quick step before entering the dashboard</p>
              </div>
            </div>

            <form onSubmit={handleProfile} className="ap-form">
              <div className="ap-grid-2">
                <div className="ap-field">
                  <label>Phone</label>
                  <input value={profile.phone} onChange={e => setProfile(p => ({ ...p, phone: e.target.value }))} placeholder="+91 9XXXXXXXXX" />
                </div>
                <div className="ap-field">
                  <label>City</label>
                  <input value={profile.city} onChange={e => setProfile(p => ({ ...p, city: e.target.value }))} placeholder="Mumbai, Delhi…" />
                </div>
              </div>
              <div className="ap-grid-2">
                <div className="ap-field">
                  <label>Experience</label>
                  <select value={profile.experience} onChange={e => setProfile(p => ({ ...p, experience: e.target.value }))}>
                    <option>Beginner</option><option>Intermediate</option><option>Advanced</option>
                  </select>
                </div>
                <div className="ap-field">
                  <label>Risk Profile</label>
                  <select value={profile.risk} onChange={e => setProfile(p => ({ ...p, risk: e.target.value }))}>
                    <option>Conservative</option><option>Moderate</option><option>Aggressive</option>
                  </select>
                </div>
              </div>
              <div className="ap-field">
                <label>Starting Capital (₹)</label>
                <input value={profile.capital} onChange={e => setProfile(p => ({ ...p, capital: e.target.value }))} placeholder="100000" />
              </div>
              <button className="ap-btn-primary" type="submit" disabled={loading}>
                {loading ? <span className="ap-spinner" /> : "Enter Dashboard →"}
              </button>
            </form>
          </div>
        </div>
        <AuthPageStyles />
      </div>
    );
  }

  /*  step 1: login / signup  */
  return (
    <div className="ap-root">
      <OrbCanvas />
      <GridPattern />
      <TickerTape />

      <div className="ap-split">

        {/* LEFT ─ brand panel */}
        <div className="ap-left">
          <div className="ap-left-inner">
            <div className="ap-brand-lockup">
              <div className="ap-logo-mark">K</div>
              <div>
                <h1 className="ap-brand-name">Kavout</h1>
                <p className="ap-brand-tagline">Enterprise-grade Indian equity forecasting</p>
              </div>
            </div>

            <div className="ap-headline">
              <h2>Predict markets.<br /><span className="ap-headline-accent">Trade smarter.</span></h2>
              <p className="ap-headline-body">
                Multi-model AI forecasts for NSE stocks with paper trading, live weather context, and portfolio analytics — all in one console.
              </p>
            </div>

            <div className="ap-features">
              {FEATURES.map((f, i) => (
                <div className="ap-feature-row" key={i} style={{ animationDelay: `${0.1 + i * 0.08}s` }}>
                  <div className="ap-feature-icon">{f.icon}</div>
                  <div>
                    <strong>{f.title}</strong>
                    <p>{f.desc}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="ap-trust-bar">
              <div><span>100K+</span><p>Forecasts</p></div>
              <div className="ap-trust-div" />
              <div><span>NSE</span><p>Listed stocks</p></div>
              <div className="ap-trust-div" />
              <div><span>3</span><p>AI models</p></div>
            </div>
          </div>
        </div>

        {/* RIGHT ─ form panel */}
        <div className="ap-right">
          <div className="ap-card">
            <div className="ap-card-header">
              <div className="ap-logo-mark ap-logo-sm">K</div>
              <div>
                <h2 className="ap-card-title">
                  {mode === "login" ? "Welcome back" : "Create account"}
                </h2>
                <p className="ap-card-sub">
                  {mode === "login" ? "Sign in to your console" : "Start trading with AI insights"}
                </p>
              </div>
            </div>

            {/* tab switcher */}
            <div className="ap-tab-row">
              <button type="button" className={`ap-tab ${mode === "login" ? "active" : ""}`} onClick={() => switchMode("login")}>
                <span>Log In</span>
              </button>
              <button type="button" className={`ap-tab ${mode === "signup" ? "active" : ""}`} onClick={() => switchMode("signup")}>
                <span>Sign Up</span>
              </button>
            </div>

            <form onSubmit={handleAuth} className="ap-form">
              {mode === "signup" && (
                <div className="ap-field ap-field-animated">
                  <label>Full Name</label>
                  <div className="ap-input-wrap">
                    <input
                      value={auth.name}
                      onChange={e => setAuth(p => ({ ...p, name: e.target.value }))}
                      placeholder="Your full name"
                      autoComplete="name"
                    />
                  </div>
                </div>
              )}

              <div className="ap-field">
                <label>Email address</label>
                <div className="ap-input-wrap">
                  <input
                    type="email"
                    value={auth.email}
                    onChange={e => setAuth(p => ({ ...p, email: e.target.value.toLowerCase() }))}
                    placeholder="name@company.com"
                    autoComplete="email"
                  />
                </div>
              </div>

              <div className="ap-field">
                <label>
                  Password
                  {mode === "login" && <button type="button" className="ap-forgot">Forgot?</button>}
                </label>
                <div className="ap-input-wrap">
                  <input
                    type={showPass ? "text" : "password"}
                    value={auth.password}
                    onChange={e => setAuth(p => ({ ...p, password: e.target.value }))}
                    placeholder={mode === "login" ? "Your password" : "Min. 6 characters"}
                    autoComplete={mode === "login" ? "current-password" : "new-password"}
                  />
                  <button type="button" className="ap-eye" onClick={() => setShowPass(s => !s)}>
                    {showPass ? "" : ""}
                  </button>
                </div>
              </div>

              {mode === "signup" && (
                <label className="ap-demo-toggle">
                  <div className="ap-toggle-wrap">
                    <input
                      type="checkbox"
                      checked={auth.demo}
                      onChange={e => setAuth(p => ({ ...p, demo: e.target.checked }))}
                    />
                    <span className="ap-toggle-slider" />
                  </div>
                  <div>
                    <strong>30-day demo session</strong>
                    <p>Access all features free for one month on this email</p>
                  </div>
                </label>
              )}

              {error && (
                <div className="ap-error">
                  <span>Error</span> {error}
                </div>
              )}

              <button className="ap-btn-primary" type="submit" disabled={loading}>
                {loading
                  ? <span className="ap-spinner" />
                  : mode === "login"
                    ? "Access Dashboard →"
                    : "Create Account →"
                }
              </button>

              {mode === "login" && (
                <p className="ap-switch-hint">
                  New to Kavout?{" "}
                  <button type="button" onClick={() => switchMode("signup")}>Create a free account</button>
                </p>
              )}
              {mode === "signup" && (
                <p className="ap-switch-hint">
                  Already have an account?{" "}
                  <button type="button" onClick={() => switchMode("login")}>Log in</button>
                </p>
              )}
            </form>

            <div className="ap-card-footer">
              <div className="ap-secure-badge">256-bit encrypted · Secure login</div>
            </div>
          </div>
        </div>
      </div>

      <AuthPageStyles />
    </div>
  );
}

/* scoped styles injected as a component */
function AuthPageStyles() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

      .ap-root {
        min-height: 100vh;
        position: relative;
        overflow: hidden;
        background: #050d18;
        font-family: 'Manrope', sans-serif;
        color: #e8f2ff;
      }

      /* ── canvas & bg ── */
      .ap-orb-canvas {
        position: fixed;
        inset: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
      }

      .ap-grid-pattern {
        position: fixed;
        inset: 0;
        width: 100%;
        height: 100%;
        color: #4a9eff;
        pointer-events: none;
        z-index: 0;
      }

      /* ── ticker tape ── */
      .ap-ticker-wrap {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 36px;
        background: rgba(5, 13, 24, 0.85);
        border-bottom: 1px solid rgba(74, 158, 255, 0.12);
        overflow: hidden;
        z-index: 10;
        backdrop-filter: blur(8px);
      }

      .ap-ticker-track {
        display: flex;
        align-items: center;
        height: 100%;
        gap: 0;
        animation: ticker-scroll 32s linear infinite;
        white-space: nowrap;
        width: max-content;
      }

      @keyframes ticker-scroll {
        from { transform: translateX(0); }
        to   { transform: translateX(-50%); }
      }

      .ap-ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 0 24px;
        border-right: 1px solid rgba(74, 158, 255, 0.1);
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
      }

      .ap-ticker-sym { color: rgba(200, 225, 255, 0.7); font-weight: 600; }
      .ap-ticker-val { color: #e8f2ff; font-weight: 600; }
      .ap-ticker-chg.up   { color: #34d399; }
      .ap-ticker-chg.down { color: #f87171; }

      /* ── split layout ── */
      .ap-split {
        position: relative;
        z-index: 5;
        min-height: 100vh;
        padding-top: 36px;
        display: grid;
        grid-template-columns: 1fr 520px;
        gap: 0;
      }

      /* ── left panel ── */
      .ap-left {
        display: flex;
        align-items: center;
        padding: 60px 56px 60px 72px;
        position: relative;
      }

      .ap-left::after {
        content: '';
        position: absolute;
        right: 0;
        top: 10%;
        bottom: 10%;
        width: 1px;
        background: linear-gradient(to bottom, transparent, rgba(74, 158, 255, 0.25) 30%, rgba(74, 158, 255, 0.25) 70%, transparent);
      }

      .ap-left-inner {
        display: flex;
        flex-direction: column;
        gap: 40px;
        max-width: 540px;
        animation: ap-fade-up 0.7s ease both;
      }

      .ap-brand-lockup {
        display: flex;
        align-items: center;
        gap: 14px;
      }

      .ap-logo-mark {
        width: 52px;
        height: 52px;
        border-radius: 16px;
        background: linear-gradient(135deg, #3b82f6, #10b981);
        display: grid;
        place-items: center;
        font-weight: 800;
        font-size: 1.05rem;
        color: #fff;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.45);
        flex-shrink: 0;
        letter-spacing: -0.02em;
        font-family: 'Manrope', sans-serif;
      }

      .ap-logo-sm { width: 40px; height: 40px; font-size: 0.88rem; border-radius: 12px; }

      .ap-brand-name {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        font-family: 'Manrope', sans-serif;
      }

      .ap-brand-ai { color: #4ea1ff; }

      .ap-brand-tagline {
        margin: 3px 0 0;
        font-size: 0.78rem;
        color: rgba(180, 210, 240, 0.55);
        font-weight: 500;
      }

      .ap-headline h2 {
        margin: 0 0 14px;
        font-size: clamp(2.2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.04em;
        font-family: 'Manrope', sans-serif;
        color: #f0f8ff;
      }

      .ap-headline-accent {
        background: linear-gradient(135deg, #4ea1ff, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }

      .ap-headline-body {
        margin: 0;
        font-size: 1rem;
        color: rgba(180, 210, 240, 0.65);
        line-height: 1.65;
        max-width: 440px;
      }

      /* ── features ── */
      .ap-features {
        display: flex;
        flex-direction: column;
        gap: 14px;
      }

      .ap-feature-row {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(74, 158, 255, 0.1);
        transition: background 0.2s, border-color 0.2s;
        animation: ap-fade-up 0.6s ease both;
      }

      .ap-feature-row:hover {
        background: rgba(74, 158, 255, 0.08);
        border-color: rgba(74, 158, 255, 0.22);
      }

      .ap-feature-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: rgba(59, 130, 246, 0.15);
        display: grid;
        place-items: center;
        font-size: 1rem;
        flex-shrink: 0;
      }

      .ap-feature-row strong {
        display: block;
        font-size: 0.88rem;
        color: #e0f0ff;
        margin-bottom: 3px;
        font-weight: 700;
      }

      .ap-feature-row p {
        margin: 0;
        font-size: 0.76rem;
        color: rgba(160, 195, 230, 0.6);
        line-height: 1.5;
      }

      /* ── trust bar ── */
      .ap-trust-bar {
        display: flex;
        align-items: center;
        gap: 28px;
      }

      .ap-trust-bar > div:not(.ap-trust-div) span {
        display: block;
        font-size: 1.35rem;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
        background: linear-gradient(135deg, #4ea1ff, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
      }

      .ap-trust-bar p {
        margin: 2px 0 0;
        font-size: 0.72rem;
        color: rgba(160, 195, 230, 0.5);
        font-weight: 600;
      }

      .ap-trust-div {
        width: 1px;
        height: 36px;
        background: rgba(74, 158, 255, 0.2);
      }

      /* ── right panel ── */
      .ap-right {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 40px 48px 40px 40px;
      }

      /* ── card ── */
      .ap-card {
        width: 100%;
        max-width: 420px;
        background: rgba(10, 20, 36, 0.75);
        border: 1px solid rgba(74, 158, 255, 0.18);
        border-radius: 24px;
        padding: 32px;
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        box-shadow:
          0 32px 80px rgba(0, 0, 0, 0.55),
          inset 0 1px 0 rgba(255, 255, 255, 0.06);
        animation: ap-fade-up 0.55s ease both 0.1s;
      }

      .ap-card-profile {
        max-width: 520px;
      }

      .ap-card-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 24px;
      }

      .ap-card-title {
        margin: 0;
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        font-family: 'Manrope', sans-serif;
        color: #f0f8ff;
      }

      .ap-card-sub {
        margin: 3px 0 0;
        font-size: 0.8rem;
        color: rgba(150, 190, 230, 0.6);
      }

      /* ── tab row ── */
      .ap-tab-row {
        display: flex;
        gap: 4px;
        margin-bottom: 24px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 4px;
        border: 1px solid rgba(74, 158, 255, 0.1);
      }

      .ap-tab {
        flex: 1;
        padding: 9px 12px;
        border-radius: 10px;
        border: none;
        background: transparent;
        color: rgba(150, 190, 230, 0.55);
        font-size: 0.86rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.22s;
        font-family: 'Manrope', sans-serif;
      }

      .ap-tab.active {
        background: linear-gradient(135deg, #1d4ed8, #0d9488);
        color: #fff;
        box-shadow: 0 4px 14px rgba(29, 78, 216, 0.4);
      }

      .ap-tab:not(.active):hover {
        color: rgba(200, 225, 255, 0.8);
        background: rgba(255,255,255,0.05);
      }

      /* ── form ── */
      .ap-form {
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .ap-field {
        display: flex;
        flex-direction: column;
        gap: 7px;
      }

      .ap-field-animated {
        animation: ap-slide-down 0.3s ease both;
      }

      @keyframes ap-slide-down {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      .ap-field label {
        font-size: 0.74rem;
        font-weight: 700;
        color: rgba(150, 190, 230, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }

      .ap-forgot {
        background: none;
        border: none;
        color: #4ea1ff;
        font-size: 0.72rem;
        font-weight: 600;
        cursor: pointer;
        padding: 0;
        text-transform: none;
        letter-spacing: 0;
        transition: color 0.2s;
      }

      .ap-forgot:hover { color: #7ec3ff; }

      .ap-input-wrap {
        position: relative;
        display: flex;
        align-items: center;
      }

      .ap-input-icon {
        position: absolute;
        left: 13px;
        font-size: 0.88rem;
        pointer-events: none;
        z-index: 1;
        opacity: 0.55;
      }

      .ap-input-wrap input,
      .ap-input-wrap select {
        width: 100%;
        padding: 12px 14px 12px 38px;
        background: rgba(255, 255, 255, 0.05);
        border: 1.5px solid rgba(74, 158, 255, 0.18);
        border-radius: 12px;
        color: #e8f2ff;
        font-size: 0.92rem;
        font-family: 'Manrope', sans-serif;
        transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
        outline: none;
      }

      .ap-input-wrap input::placeholder {
        color: rgba(130, 170, 210, 0.38);
      }

      .ap-input-wrap input:focus {
        border-color: #4ea1ff;
        background: rgba(78, 161, 255, 0.07);
        box-shadow: 0 0 0 4px rgba(78, 161, 255, 0.14);
      }

      .ap-eye {
        position: absolute;
        right: 12px;
        background: none;
        border: none;
        cursor: pointer;
        font-size: 0.9rem;
        opacity: 0.5;
        transition: opacity 0.2s;
        padding: 0;
        line-height: 1;
      }

      .ap-eye:hover { opacity: 0.9; }

      /* plain fields (profile step) */
      .ap-field input,
      .ap-field select {
        padding: 11px 14px;
        background: rgba(255, 255, 255, 0.05);
        border: 1.5px solid rgba(74, 158, 255, 0.18);
        border-radius: 12px;
        color: #e8f2ff;
        font-size: 0.9rem;
        font-family: 'Manrope', sans-serif;
        outline: none;
        transition: border-color 0.2s, box-shadow 0.2s;
        width: 100%;
      }

      .ap-field input:focus,
      .ap-field select:focus {
        border-color: #4ea1ff;
        box-shadow: 0 0 0 4px rgba(78, 161, 255, 0.14);
      }

      .ap-field select option {
        background: #0d1e32;
        color: #e8f2ff;
      }

      /* ── toggle switch (demo) ── */
      .ap-demo-toggle {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 14px;
        background: rgba(78, 161, 255, 0.06);
        border: 1px solid rgba(78, 161, 255, 0.15);
        border-radius: 12px;
        cursor: pointer;
      }

      .ap-toggle-wrap {
        position: relative;
        flex-shrink: 0;
        width: 40px;
        height: 22px;
        margin-top: 2px;
      }

      .ap-toggle-wrap input {
        position: absolute;
        opacity: 0;
        width: 0;
        height: 0;
      }

      .ap-toggle-slider {
        position: absolute;
        inset: 0;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255,255,255,0.15);
        transition: background 0.25s;
        cursor: pointer;
      }

      .ap-toggle-slider::after {
        content: '';
        position: absolute;
        top: 2px;
        left: 2px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: rgba(200, 225, 255, 0.6);
        transition: transform 0.25s, background 0.25s;
      }

      .ap-toggle-wrap input:checked + .ap-toggle-slider {
        background: linear-gradient(135deg, #3b82f6, #10b981);
        border-color: transparent;
      }

      .ap-toggle-wrap input:checked + .ap-toggle-slider::after {
        transform: translateX(18px);
        background: #fff;
      }

      .ap-demo-toggle strong {
        display: block;
        font-size: 0.84rem;
        color: #c8e0ff;
        margin-bottom: 3px;
      }

      .ap-demo-toggle p {
        margin: 0;
        font-size: 0.74rem;
        color: rgba(150, 190, 230, 0.55);
      }

      /* ── error ── */
      .ap-error {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.3);
        border-radius: 10px;
        font-size: 0.83rem;
        color: #fca5a5;
        font-weight: 600;
        animation: ap-shake 0.35s ease;
      }

      @keyframes ap-shake {
        0%,100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
      }

      /* ── primary button ── */
      .ap-btn-primary {
        padding: 14px 18px;
        border-radius: 13px;
        border: none;
        background: linear-gradient(135deg, #2563eb 0%, #0891b2 60%, #059669 100%);
        color: #fff;
        font-size: 0.95rem;
        font-weight: 800;
        font-family: 'Manrope', sans-serif;
        cursor: pointer;
        transition: all 0.22s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.4);
        letter-spacing: -0.01em;
      }

      .ap-btn-primary:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 14px 36px rgba(37, 99, 235, 0.52);
        background: linear-gradient(135deg, #3b82f6 0%, #0ea5e9 60%, #10b981 100%);
      }

      .ap-btn-primary:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
      }

      /* ── spinner ── */
      .ap-spinner {
        display: inline-block;
        width: 18px;
        height: 18px;
        border: 2.5px solid rgba(255,255,255,0.3);
        border-top-color: #fff;
        border-radius: 50%;
        animation: ap-spin 0.7s linear infinite;
      }

      @keyframes ap-spin { to { transform: rotate(360deg); } }

      /* ── switch hint ── */
      .ap-switch-hint {
        margin: 0;
        text-align: center;
        font-size: 0.8rem;
        color: rgba(150, 190, 230, 0.5);
      }

      .ap-switch-hint button {
        background: none;
        border: none;
        color: #4ea1ff;
        font-weight: 700;
        cursor: pointer;
        font-size: inherit;
        padding: 0;
        transition: color 0.2s;
        font-family: 'Manrope', sans-serif;
      }

      .ap-switch-hint button:hover { color: #7ec3ff; }

      /* ── card footer ── */
      .ap-card-footer {
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid rgba(74, 158, 255, 0.1);
        display: flex;
        justify-content: center;
      }

      .ap-secure-badge {
        font-size: 0.72rem;
        color: rgba(120, 170, 220, 0.45);
        font-weight: 600;
        letter-spacing: 0.03em;
      }

      /* ── grid 2 for profile ── */
      .ap-grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }

      /* ── center for profile page ── */
      .ap-center {
        position: relative;
        z-index: 5;
        min-height: 100vh;
        padding-top: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 80px 24px 48px;
      }

      /* ── animations ── */
      @keyframes ap-fade-up {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      /* ── responsive ── */
      @media (max-width: 1000px) {
        .ap-split {
          grid-template-columns: 1fr;
        }
        .ap-left {
          display: none;
        }
        .ap-right {
          min-height: calc(100vh - 36px);
          padding: 40px 24px;
        }
      }

      @media (max-width: 480px) {
        .ap-card {
          padding: 24px 20px;
          border-radius: 20px;
        }
        .ap-grid-2 {
          grid-template-columns: 1fr;
        }
        .ap-headline h2 {
          font-size: 2rem;
        }
      }
    `}</style>
  );
}