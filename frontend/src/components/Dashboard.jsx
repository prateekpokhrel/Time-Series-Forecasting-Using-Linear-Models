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

import { getForecast, getSymbols } from "../api";

const horizons = [
  { key: "1d", label: "1 Day" },
  { key: "15d", label: "15 Days" },
  { key: "30d", label: "30 Days" },
];

export default function Dashboard() {

  const [form, setForm] = useState({
    ticker: "RELIANCE",
    horizon: "1d",
    data_source: "auto",
  });

  const [forecast, setForecast] = useState(null);

  const [loading, setLoading] = useState(false);



  const inr = (v) =>
    new Intl.NumberFormat("en-IN", {
      minimumFractionDigits: 2,
    }).format(v || 0);


  const pct = (v) =>
    `${(v || 0).toFixed(2)}%`;



  useEffect(() => {

    async function load() {

      try {

        const data = await getSymbols({
          data_source: form.data_source,
        });

      } catch {}

    }

    load();

  }, []);



  const onSubmit = async (e) => {

    e.preventDefault();

    setLoading(true);

    try {

      const res = await getForecast(form);

      console.log(res);

      setForecast(res);

    }

    catch (err) {

      alert(err.message);

    }

    setLoading(false);

  };



  // correct fields from backend

  const prediction =
    forecast?.projected_close || 0;

  const change =
    forecast?.projected_change_pct || 0;

  const lastClose =
    forecast?.last_close || 0;



  // chart

  const chartData = useMemo(() => {

    if (!forecast?.forecast) return [];

    return forecast.forecast.map((p) => ({

      date: p.date,

      price: p.value,

    }));

  }, [forecast]);



  return (

    <>

      <header className="hero">

        <h2>Market Forecasting</h2>

        <div className="hero-badge">

          {form.horizon} Horizon

        </div>

      </header>



      <section className="search-card">

        <form className="search-grid" onSubmit={onSubmit}>

          <input

            value={form.ticker}

            onChange={(e) =>
              setForm({
                ...form,
                ticker: e.target.value.toUpperCase(),
              })
            }

          />



          <select

            value={form.horizon}

            onChange={(e) =>
              setForm({
                ...form,
                horizon: e.target.value,
              })
            }

          >

            {horizons.map((h) => (

              <option key={h.key} value={h.key}>

                {h.label}

              </option>

            ))}

          </select>



          <button className="btn primary">

            {loading ? "Processing..." : "Predict"}

          </button>

        </form>

      </section>



      {!forecast && (

        <div className="empty">

          Forecast visualization and metrics display here.

        </div>

      )}



      {forecast && (

        <>

          {/* Metrics */}

          <section className="metrics-grid">

            <div className="metric">

              <h4>Last Close</h4>

              <p>₹ {inr(lastClose)}</p>

            </div>



            <div className="metric">

              <h4>Projected Close</h4>

              <p>₹ {inr(prediction)}</p>

            </div>



            <div className="metric">

              <h4>Projected Change</h4>

              <p>{pct(change)}</p>

            </div>

          </section>



          {/* Chart */}

          <section className="chart-card">

            <ResponsiveContainer width="100%" height={400}>

              <ComposedChart data={chartData}>

                <CartesianGrid stroke="#eee" />

                <XAxis dataKey="date" />

                <YAxis />

                <Tooltip />

                <Legend />



                <Line

                  type="monotone"

                  dataKey="price"

                  stroke="#2563eb"

                  strokeWidth={3}

                />

              </ComposedChart>

            </ResponsiveContainer>

          </section>

        </>

      )}

    </>

  );

}