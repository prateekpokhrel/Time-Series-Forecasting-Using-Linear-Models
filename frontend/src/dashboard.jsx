<<<<<<< HEAD
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

import { getForecast, getSymbols } from "./api";


/* ---------------- UTILITIES ---------------- */

const horizons = [
  { key: "1d", label: "1 Day", subtitle: "Minute-wise intraday forecast" },
  { key: "15d", label: "15 Days", subtitle: "Hour-wise movement forecast" },
  { key: "30d", label: "30 Days", subtitle: "Day-wise curve with high/low" },
];

const initialForm = {
  ticker: "RELIANCE",
  horizon: "1d",
  data_source: "auto",
  local_data_dir: "",
};

function inr(value) {
  if (typeof value !== "number") return "--";
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function pct(value) {
  if (typeof value !== "number") return "--";
  return `${value.toFixed(2)}%`;
}

function formatDate(date) {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(date);
}

function formatTime(date) {
  return new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function weatherLabel(code) {
  const map = {0:"Clear",1:"Mostly Clear",2:"Partly Cloudy",3:"Overcast"};
  return map[code] || "Weather";
}

function saveLocal(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function loadLocal(key, fallback = null) {
  const raw = localStorage.getItem(key);
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}


/* ---------------- PAPER TRADING ---------------- */

export function PaperTradingPanel({ ticker, lastPrice }) {

  const [portfolio, setPortfolio] = useState(() =>
    loadLocal("paper_portfolio", { cash: 100000, positions: {}, trades: [] })
  );

  const [order, setOrder] = useState({
    side: "BUY",
    qty: "1",
    price: "",
  });

  useEffect(() => {
    saveLocal("paper_portfolio", portfolio);
  }, [portfolio]);

  useEffect(() => {
    if (!order.price && lastPrice) {
      setOrder(prev => ({
        ...prev,
        price: lastPrice.toFixed(2)
      }));
    }
  }, [lastPrice]);


  const position = portfolio.positions[ticker] || { qty: 0, avg: 0 };

  const pnl =
    position.qty * ((Number(order.price)||lastPrice) - position.avg);


  function executeTrade() {

    const qty = Number(order.qty);
    const price = Number(order.price);

    if (!qty || !price) return;

    let next = { ...portfolio };

    if (order.side === "BUY") {

      const cost = qty * price;

      if (next.cash < cost) return;

      next.cash -= cost;

      const pos = next.positions[ticker] || { qty:0, avg:0 };

      const newQty = pos.qty + qty;

      const newAvg =
        (pos.avg * pos.qty + cost) / newQty;

      next.positions[ticker] = {
        qty:newQty,
        avg:newAvg
      };

    } else {

      const pos = next.positions[ticker];

      if (!pos || pos.qty < qty) return;

      next.cash += qty * price;

      pos.qty -= qty;

      if (pos.qty === 0) delete next.positions[ticker];

    }


    next.trades.unshift({

      id:Date.now(),

      ticker,

      side:order.side,

      qty,

      price,

      time:new Date().toLocaleString()

    });


    setPortfolio(next);

  }


  return (

    <div className="panel">

      <h3>Paper Trading</h3>

      <p>Cash: Rs {inr(portfolio.cash)}</p>

      <p>Position: {position.qty}</p>

      <p>PnL: Rs {inr(pnl)}</p>


      <select

        value={order.side}

        onChange={e=>setOrder({...order,side:e.target.value})}

      >

        <option>BUY</option>

        <option>SELL</option>

      </select>


      <input

        value={order.qty}

        onChange={e=>setOrder({...order,qty:e.target.value})}

      />


      <input

        value={order.price}

        onChange={e=>setOrder({...order,price:e.target.value})}

      />


      <button onClick={executeTrade}>

        Execute

      </button>


    </div>

  );

}



/* ---------------- DASHBOARD ---------------- */

export default function Dashboard({

  user,

  profile,

  theme,

  setTheme,

}) {

  const [form,setForm]=useState(initialForm);

  const [forecast,setForecast]=useState(null);

  const [symbols,setSymbols]=useState([]);


  useEffect(()=>{

    getSymbols({}).then(res=>{

      setSymbols(res.symbols||[])

    })

  },[]);



  async function runForecast(){

    const result = await getForecast(form);

    setForecast(result);

  }


  const chartData = useMemo(()=>{

    if(!forecast) return [];

    return [

      ...forecast.history.map(x=>({

        date:x.date,

        history:x.value

      })),

      ...forecast.forecast.map(x=>({

        date:x.date,

        forecast:x.value

      }))

    ];

  },[forecast]);



  return (

    <div>

      <h2>Welcome {user.name}</h2>


      <button onClick={()=>

        setTheme(theme==="dark"?"light":"dark")

      }>

        Toggle Theme

      </button>



      <input

        value={form.ticker}

        onChange={e=>

          setForm({...form,ticker:e.target.value})

        }

      />


      <button onClick={runForecast}>

        Forecast

      </button>



      <PaperTradingPanel

        ticker={form.ticker}

        lastPrice={forecast?.last_close}

      />



      <ResponsiveContainer width="100%" height={400}>

        <ComposedChart data={chartData}>

          <CartesianGrid/>

          <XAxis dataKey="date"/>

          <YAxis/>

          <Tooltip/>

          <Legend/>

          <Line dataKey="history"/>

          <Line dataKey="forecast"/>

        </ComposedChart>

      </ResponsiveContainer>


    </div>

  );

=======
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

import { getForecast, getSymbols } from "./api";


/* ---------------- UTILITIES ---------------- */

const horizons = [
  { key: "1d", label: "1 Day", subtitle: "Minute-wise intraday forecast" },
  { key: "15d", label: "15 Days", subtitle: "Hour-wise movement forecast" },
  { key: "30d", label: "30 Days", subtitle: "Day-wise curve with high/low" },
];

const initialForm = {
  ticker: "RELIANCE",
  horizon: "1d",
  data_source: "auto",
  local_data_dir: "",
};

function inr(value) {
  if (typeof value !== "number") return "--";
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function pct(value) {
  if (typeof value !== "number") return "--";
  return `${value.toFixed(2)}%`;
}

function formatDate(date) {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(date);
}

function formatTime(date) {
  return new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function weatherLabel(code) {
  const map = {0:"Clear",1:"Mostly Clear",2:"Partly Cloudy",3:"Overcast"};
  return map[code] || "Weather";
}

function saveLocal(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function loadLocal(key, fallback = null) {
  const raw = localStorage.getItem(key);
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}


/* ---------------- PAPER TRADING ---------------- */

export function PaperTradingPanel({ ticker, lastPrice }) {

  const [portfolio, setPortfolio] = useState(() =>
    loadLocal("paper_portfolio", { cash: 100000, positions: {}, trades: [] })
  );

  const [order, setOrder] = useState({
    side: "BUY",
    qty: "1",
    price: "",
  });

  useEffect(() => {
    saveLocal("paper_portfolio", portfolio);
  }, [portfolio]);

  useEffect(() => {
    if (!order.price && lastPrice) {
      setOrder(prev => ({
        ...prev,
        price: lastPrice.toFixed(2)
      }));
    }
  }, [lastPrice]);


  const position = portfolio.positions[ticker] || { qty: 0, avg: 0 };

  const pnl =
    position.qty * ((Number(order.price)||lastPrice) - position.avg);


  function executeTrade() {

    const qty = Number(order.qty);
    const price = Number(order.price);

    if (!qty || !price) return;

    let next = { ...portfolio };

    if (order.side === "BUY") {

      const cost = qty * price;

      if (next.cash < cost) return;

      next.cash -= cost;

      const pos = next.positions[ticker] || { qty:0, avg:0 };

      const newQty = pos.qty + qty;

      const newAvg =
        (pos.avg * pos.qty + cost) / newQty;

      next.positions[ticker] = {
        qty:newQty,
        avg:newAvg
      };

    } else {

      const pos = next.positions[ticker];

      if (!pos || pos.qty < qty) return;

      next.cash += qty * price;

      pos.qty -= qty;

      if (pos.qty === 0) delete next.positions[ticker];

    }


    next.trades.unshift({

      id:Date.now(),

      ticker,

      side:order.side,

      qty,

      price,

      time:new Date().toLocaleString()

    });


    setPortfolio(next);

  }


  return (

    <div className="panel">

      <h3>Paper Trading</h3>

      <p>Cash: Rs {inr(portfolio.cash)}</p>

      <p>Position: {position.qty}</p>

      <p>PnL: Rs {inr(pnl)}</p>


      <select

        value={order.side}

        onChange={e=>setOrder({...order,side:e.target.value})}

      >

        <option>BUY</option>

        <option>SELL</option>

      </select>


      <input

        value={order.qty}

        onChange={e=>setOrder({...order,qty:e.target.value})}

      />


      <input

        value={order.price}

        onChange={e=>setOrder({...order,price:e.target.value})}

      />


      <button onClick={executeTrade}>

        Execute

      </button>


    </div>

  );

}



/* ---------------- DASHBOARD ---------------- */

export default function Dashboard({

  user,

  profile,

  theme,

  setTheme,

}) {

  const [form,setForm]=useState(initialForm);

  const [forecast,setForecast]=useState(null);

  const [symbols,setSymbols]=useState([]);


  useEffect(()=>{

    getSymbols({}).then(res=>{

      setSymbols(res.symbols||[])

    })

  },[]);



  async function runForecast(){

    const result = await getForecast(form);

    setForecast(result);

  }


  const chartData = useMemo(()=>{

    if(!forecast) return [];

    return [

      ...forecast.history.map(x=>({

        date:x.date,

        history:x.value

      })),

      ...forecast.forecast.map(x=>({

        date:x.date,

        forecast:x.value

      }))

    ];

  },[forecast]);



  return (

    <div>

      <h2>Welcome {user.name}</h2>


      <button onClick={()=>

        setTheme(theme==="dark"?"light":"dark")

      }>

        Toggle Theme

      </button>



      <input

        value={form.ticker}

        onChange={e=>

          setForm({...form,ticker:e.target.value})

        }

      />


      <button onClick={runForecast}>

        Forecast

      </button>



      <PaperTradingPanel

        ticker={form.ticker}

        lastPrice={forecast?.last_close}

      />



      <ResponsiveContainer width="100%" height={400}>

        <ComposedChart data={chartData}>

          <CartesianGrid/>

          <XAxis dataKey="date"/>

          <YAxis/>

          <Tooltip/>

          <Legend/>

          <Line dataKey="history"/>

          <Line dataKey="forecast"/>

        </ComposedChart>

      </ResponsiveContainer>


    </div>

  );

>>>>>>> cc5000d6a14516db49261019afca98d67f50e91d
}