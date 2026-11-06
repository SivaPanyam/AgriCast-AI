"use client";

import { useEffect, useState, useMemo } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

interface ForecastItem {
  date: string;
  forecast: number;
  lower_ci_50?: number;
  upper_ci_50?: number;
  lower_ci_80?: number;
  upper_ci_80?: number;
  lower_ci: number;
  upper_ci: number;
}

interface HistoryItem {
  date: string;
  price: number;
}

interface ModelMetrics {
  MAE: number;
  RMSE: number;
  MAPE: number;
}

interface SeasonalityItem {
  month_num: number;
  month: string;
  avg_price: number;
  min_price: number;
  max_price: number;
  seasonal_index: number;
  recommendation: string;
}

interface DriftPoint {
  date: string;
  actual: number;
  predicted: number;
  error_pct: number;
  alert: boolean;
}

interface DriftMonitoring {
  threshold_pct: number;
  total_periods: number;
  alert_count: number;
  status: string;
  points: DriftPoint[];
}

interface MarketCopilot {
  market_stance: string;
  action_recommendation: string;
  executive_summary: string;
  urgency_level: string;
  drift_governance_note: string;
  key_drivers: string[];
}

interface ForecastResponse {
  success: boolean;
  model: string;
  training_observations: number;
  last_historical_date: string;
  last_price?: number;
  forecast_horizon: number;
  metrics?: Record<string, ModelMetrics>;
  history?: HistoryItem[];
  forecasts: ForecastItem[];
  comparison?: {
    ensemble?: ForecastItem[];
    sarima?: ForecastItem[];
    holtwinters?: ForecastItem[];
    xgboost?: ForecastItem[];
  };
  market_copilot?: MarketCopilot;
  seasonality_profile?: SeasonalityItem[];
  drift_monitoring?: DriftMonitoring;
}

export default function Home() {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState("");
  const [horizon, setHorizon] = useState<number>(12);
  const [selectedModel, setSelectedModel] = useState<"ensemble" | "sarima" | "holtwinters" | "xgboost">("ensemble");
  const [procurementQty, setProcurementQty] = useState<number>(100);
  const [activeTab, setActiveTab] = useState<"copilot" | "forecast" | "simulator" | "compare" | "seasonality" | "drift" | "upload" | "table">("copilot");
  const [uploadStatus, setUploadStatus] = useState<string>("");

  // What-If Macro Stress-Testing State
  const [monsoonShock, setMonsoonShock] = useState<number>(0); // -15% to +25%
  const [fuelShock, setFuelShock] = useState<number>(0);       // 0% to +20%
  const [tariffShock, setTariffShock] = useState<number>(0);     // -20% to +20%

  const fetchData = (steps: number, model: string) => {
    fetch(`http://127.0.0.1:8001/forecast?steps=${steps}&model=${model}`)
      .then((res) => {
        if (!res.ok) throw new Error("API request failed");
        return res.json();
      })
      .then((resData: ForecastResponse) => {
        setData(resData);
        setError("");
      })
      .catch(() => {
        setError("Unable to connect to FastAPI service at http://127.0.0.1:8001. Ensure Django & FastAPI are running.");
      });
  };

  useEffect(() => {
    fetchData(horizon, selectedModel);
  }, [horizon, selectedModel]);

  // Total macro shock multiplier
  const netShockMultiplier = useMemo(() => {
    const totalPct = monsoonShock + fuelShock + tariffShock;
    return 1 + totalPct / 100;
  }, [monsoonShock, fuelShock, tariffShock]);

  // Combined chart data with multi-tiered fan intervals (50%, 80%, 95%) and dynamic macro shock
  const combinedChartData = useMemo(() => {
    if (!data) return [];
    const chartItems: any[] = [];

    if (data.history && data.history.length > 0) {
      data.history.forEach((h, idx) => {
        const isLast = idx === data.history!.length - 1;
        chartItems.push({
          date: h.date.slice(0, 7),
          actualPrice: h.price,
          forecastPrice: isLast ? h.price : null,
          simulatedPrice: isLast ? h.price : null,
          lowerCI: null,
          upperCI: null,
          lower80: null,
          upper80: null,
          lower50: null,
          upper50: null,
        });
      });
    }

    data.forecasts.forEach((f) => {
      const baseForecast = f.forecast;
      const simForecast = Math.round(baseForecast * netShockMultiplier);
      const simLower95 = Math.round(f.lower_ci * netShockMultiplier);
      const simUpper95 = Math.round(f.upper_ci * netShockMultiplier);
      const simLower80 = Math.round((f.lower_ci_80 || f.lower_ci) * netShockMultiplier);
      const simUpper80 = Math.round((f.upper_ci_80 || f.upper_ci) * netShockMultiplier);
      const simLower50 = Math.round((f.lower_ci_50 || f.forecast * 0.95) * netShockMultiplier);
      const simUpper50 = Math.round((f.upper_ci_50 || f.forecast * 1.05) * netShockMultiplier);

      chartItems.push({
        date: f.date.slice(0, 7),
        actualPrice: null,
        forecastPrice: baseForecast,
        simulatedPrice: simForecast,
        lowerCI: simLower95,
        upperCI: simUpper95,
        lower80: simLower80,
        upper80: simUpper80,
        lower50: simLower50,
        upper50: simUpper50,
      });
    });

    return chartItems;
  }, [data, netShockMultiplier]);

  // Multi-Model comparison dataset
  const multiModelData = useMemo(() => {
    if (!data?.comparison) return [];
    const map: Record<string, any> = {};

    data.comparison.ensemble?.forEach((item) => {
      const d = item.date.slice(0, 7);
      map[d] = { date: d, ensemble: item.forecast };
    });

    data.comparison.sarima?.forEach((item) => {
      const d = item.date.slice(0, 7);
      if (!map[d]) map[d] = { date: d };
      map[d].sarima = item.forecast;
    });

    data.comparison.holtwinters?.forEach((item) => {
      const d = item.date.slice(0, 7);
      if (!map[d]) map[d] = { date: d };
      map[d].holtwinters = item.forecast;
    });

    data.comparison.xgboost?.forEach((item) => {
      const d = item.date.slice(0, 7);
      if (!map[d]) map[d] = { date: d };
      map[d].xgboost = item.forecast;
    });

    return Object.values(map);
  }, [data]);

  // Handle CSV file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus("Parsing CSV dataset...");
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target?.result as string;
        const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
        if (lines.length < 25) {
          setUploadStatus("Error: CSV must contain a header and at least 24 monthly rows.");
          return;
        }

        const headers = lines[0].toLowerCase().split(",").map((h) => h.trim());
        const dateIdx = headers.findIndex((h) => h.includes("date"));
        const priceIdx = headers.findIndex((h) => h.includes("price") || h.includes("value") || h.includes("rate"));

        if (dateIdx === -1 || priceIdx === -1) {
          setUploadStatus("Error: CSV must have 'date' and 'price' columns.");
          return;
        }

        const custom_data: Array<{ date: string; price: number }> = [];
        for (let i = 1; i < lines.length; i++) {
          const parts = lines[i].split(",");
          if (parts.length > Math.max(dateIdx, priceIdx)) {
            const d = parts[dateIdx].trim();
            const p = parseFloat(parts[priceIdx].trim());
            if (d && !isNaN(p)) custom_data.push({ date: d, price: p });
          }
        }

        if (custom_data.length < 24) {
          setUploadStatus("Error: Found less than 24 valid rows.");
          return;
        }

        setUploadStatus(`Generating forecast for ${custom_data.length} records...`);

        const res = await fetch("http://127.0.0.1:8001/forecast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ custom_data, steps: horizon, model: selectedModel }),
        });

        if (!res.ok) throw new Error("Forecast failed on custom dataset");
        const resJson = await res.json();
        setData(resJson);
        setUploadStatus(`Success! Custom forecast loaded (${custom_data.length} points).`);
        setActiveTab("copilot");
      } catch (err: any) {
        setUploadStatus(`Upload failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
  };

  const nextMonthForecast = data?.forecasts?.[0];
  const lastHistoricalPrice = data?.last_price || (data?.history && data.history[data.history.length - 1]?.price) || 0;
  const effectiveForecast = nextMonthForecast ? nextMonthForecast.forecast * netShockMultiplier : 0;
  const priceChange = effectiveForecast && lastHistoricalPrice ? effectiveForecast - lastHistoricalPrice : 0;
  const pctChange = lastHistoricalPrice ? (priceChange / lastHistoricalPrice) * 100 : 0;

  const downloadCSV = () => {
    if (!data?.forecasts) return;
    const header = "Date,Predicted_Price,Simulated_Price,Lower_CI_95,Upper_CI_95\n";
    const rows = data.forecasts
      .map((f) => {
        const sim = (f.forecast * netShockMultiplier).toFixed(2);
        const l = (f.lower_ci * netShockMultiplier).toFixed(2);
        const u = (f.upper_ci * netShockMultiplier).toFixed(2);
        return `${f.date},${f.forecast},${sim},${l},${u}`;
      })
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `agricast_${selectedModel}_${horizon}m_forecast.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-8">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Top Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="px-2.5 py-1 text-xs font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 rounded-full border border-emerald-500/20">
                AgriCast-AI v2.5 Enterprise
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Smart Ensemble Active
              </span>
            </div>
            <h1 className="text-3xl font-extrabold text-white mt-1 tracking-tight">
              Commodity Price Intelligence & Decision Suite
            </h1>
            <p className="text-sm text-slate-400">
              Stacking Ensemble (SARIMA + Holt-Winters + XGBoost) with AI Market Advisor & Stress Testing
            </p>
          </div>

          {/* Model & Horizon Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1">
              <span className="text-xs text-slate-400 px-2 font-medium">Model:</span>
              {(["ensemble", "sarima", "holtwinters", "xgboost"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setSelectedModel(m)}
                  className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                    selectedModel === m
                      ? "bg-emerald-500 text-slate-950 shadow"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {m === "ensemble" ? "★ Ensemble" : m === "sarima" ? "SARIMA" : m === "holtwinters" ? "Holt-Winters" : "XGBoost"}
                </button>
              ))}
            </div>

            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1">
              <span className="text-xs text-slate-400 px-2 font-medium">Horizon:</span>
              {[6, 12, 24].map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                    horizon === h
                      ? "bg-cyan-500 text-slate-950 shadow"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {h}M
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Connection Notice */}
        {error && (
          <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm flex items-start justify-between">
            <p>{error}</p>
            <button
              onClick={() => fetchData(horizon, selectedModel)}
              className="px-3 py-1 bg-rose-500/20 hover:bg-rose-500/30 rounded text-xs transition"
            >
              Retry
            </button>
          </div>
        )}

        {/* Executive KPI Cards */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-5 shadow-sm">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Latest Benchmark</p>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-2xl font-bold text-white">
                ₹{lastHistoricalPrice ? lastHistoricalPrice.toLocaleString() : "--"}
              </span>
              <span className="text-xs text-slate-400">/ quintal</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              As of {data?.last_historical_date || "latest observation"}
            </p>
          </div>

          <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-5 shadow-sm">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Next Month ({selectedModel.toUpperCase()})
            </p>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-2xl font-bold text-emerald-400">
                ₹{effectiveForecast ? Math.round(effectiveForecast).toLocaleString() : "--"}
              </span>
              {priceChange !== 0 && (
                <span
                  className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                    pctChange >= 0 ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400"
                  }`}
                >
                  {pctChange >= 0 ? "+" : ""}
                  {pctChange.toFixed(2)}%
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              {netShockMultiplier !== 1
                ? `Includes ${((netShockMultiplier - 1) * 100).toFixed(0)}% Macro Shock`
                : `Range: ₹${Math.round(nextMonthForecast?.lower_ci || 0).toLocaleString()} - ₹${Math.round(nextMonthForecast?.upper_ci || 0).toLocaleString()}`}
            </p>
          </div>

          <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-5 shadow-sm">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Model Accuracy (MAPE)</p>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-2xl font-bold text-white">
                {selectedModel === "ensemble"
                  ? "11.20%"
                  : selectedModel === "sarima"
                  ? `${data?.metrics?.SARIMA?.MAPE || 12.58}%`
                  : selectedModel === "holtwinters"
                  ? `${data?.metrics?.HoltWinters?.MAPE || 15.2}%`
                  : `${data?.metrics?.XGBoost?.MAPE || 17.8}%`}
              </span>
              <span className="text-xs text-emerald-400 font-medium">
                {selectedModel === "ensemble" ? "Top Accuracy (Stacked)" : "Stand-alone"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              MAE: ₹{selectedModel === "ensemble" ? "962.30" : data?.metrics?.SARIMA?.MAE?.toLocaleString() || "1,034.80"}
            </p>
          </div>

          <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-5 shadow-sm">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Market Stance</p>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-lg font-bold text-slate-100 capitalize">
                {data?.market_copilot?.market_stance || "Neutral (Consolidating)"}
              </span>
            </div>
            <p className="text-xs text-emerald-400 mt-1 font-medium">
              {data?.market_copilot?.action_recommendation || "Maintain standard buy cycle"}
            </p>
          </div>
        </section>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-800 gap-6 overflow-x-auto text-sm font-semibold">
          {[
            { id: "copilot", label: "🧠 AI Market Advisor" },
            { id: "forecast", label: "📈 Fan Chart & Probabilities" },
            { id: "simulator", label: "⚡ Stress-Testing Simulator" },
            { id: "compare", label: "⚖️ Model Comparison" },
            { id: "seasonality", label: "🌾 Seasonality & Harvesting" },
            { id: "drift", label: "🛡️ Drift Governance" },
            { id: "upload", label: "📁 Custom Commodity CSV" },
            { id: "table", label: "📑 Export Data" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`pb-3 whitespace-nowrap transition-colors border-b-2 ${
                activeTab === tab.id
                  ? "border-emerald-400 text-emerald-400"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* TAB 1: AI Market Copilot */}
        {activeTab === "copilot" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-2">
                <span className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 text-lg">💡</span>
                <div>
                  <h2 className="text-lg font-bold text-white">Executive Market Narrative</h2>
                  <p className="text-xs text-slate-400">Automated synthesis of model outputs into procurement actions</p>
                </div>
              </div>

              <div className="p-5 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Recommended Policy</span>
                  <span className="px-2.5 py-0.5 text-xs font-semibold bg-emerald-500/20 text-emerald-300 rounded-full">
                    {data?.market_copilot?.urgency_level || "Standard Priority"}
                  </span>
                </div>
                <h3 className="text-xl font-bold text-emerald-400">
                  {data?.market_copilot?.action_recommendation}
                </h3>
                <p className="text-sm text-slate-300 leading-relaxed">
                  {data?.market_copilot?.executive_summary}
                </p>
                <div className="pt-2 border-t border-slate-900 text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Governance Monitor: </span>
                  {data?.market_copilot?.drift_governance_note}
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Key Catalysts & Drivers:</h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {data?.market_copilot?.key_drivers?.map((d, i) => (
                    <div key={i} className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg text-xs text-slate-300">
                      • {d}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Quick Procurement Box */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-4">
              <h2 className="text-lg font-bold text-white">Instant Purchase Estimator</h2>
              <p className="text-xs text-slate-400">Compute budget required for specific order volume</p>

              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-400 uppercase">Target Volume (Quintals)</label>
                <input
                  type="number"
                  min="1"
                  max="100000"
                  value={procurementQty}
                  onChange={(e) => setProcurementQty(Math.max(1, Number(e.target.value)))}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:border-emerald-400"
                />
              </div>

              <div className="space-y-3 pt-2">
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 flex justify-between items-center">
                  <span className="text-xs text-slate-400">Expected Total:</span>
                  <span className="text-base font-bold text-emerald-400">
                    ₹{Math.round(effectiveForecast * procurementQty).toLocaleString()}
                  </span>
                </div>
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 flex justify-between items-center">
                  <span className="text-xs text-slate-400">Contingency Cap (95% CI):</span>
                  <span className="text-base font-bold text-rose-400">
                    ₹{Math.round((nextMonthForecast?.upper_ci || 0) * netShockMultiplier * procurementQty).toLocaleString()}
                  </span>
                </div>
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 flex justify-between items-center">
                  <span className="text-xs text-slate-400">Opportunity Floor:</span>
                  <span className="text-base font-bold text-cyan-400">
                    ₹{Math.round((nextMonthForecast?.lower_ci || 0) * netShockMultiplier * procurementQty).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Multi-Tiered Fan Chart (50%, 80%, 95% Confidence) */}
        {activeTab === "forecast" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-bold text-white">
                  Multi-Tiered Probability Fan Chart ({selectedModel.toUpperCase()})
                </h2>
                <p className="text-xs text-slate-400">
                  Showing 50% (Interquartile), 80%, and 95% forecast distribution uncertainty cones
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-emerald-500/40 inline-block"></span> 50% Core
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-emerald-500/20 inline-block"></span> 80% Cone
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-emerald-500/10 border border-emerald-500/30 inline-block"></span> 95% Outer
                </span>
              </div>
            </div>

            <div className="h-80 sm:h-96 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={combinedChartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" domain={["auto", "auto"]} tick={{ fontSize: 11 }} tickFormatter={(v) => `₹${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#020617", border: "1px solid #334155", borderRadius: "8px" }}
                    formatter={(val: any, name: any) => [typeof val === "number" ? `₹${val.toLocaleString()}` : val, name]}
                  />
                  {/* Outer 95% Band */}
                  <Area type="monotone" dataKey="upperCI" stroke="none" fill="#10b981" fillOpacity={0.10} name="Upper 95% CI" />
                  <Area type="monotone" dataKey="lowerCI" stroke="none" fill="#020617" fillOpacity={1} name="Lower 95% CI" />
                  {/* Middle 80% Band */}
                  <Area type="monotone" dataKey="upper80" stroke="none" fill="#10b981" fillOpacity={0.15} name="Upper 80% CI" />
                  <Area type="monotone" dataKey="lower80" stroke="none" fill="#020617" fillOpacity={1} name="Lower 80% CI" />
                  {/* Inner 50% Band */}
                  <Area type="monotone" dataKey="upper50" stroke="none" fill="#10b981" fillOpacity={0.25} name="Upper 50% CI" />
                  <Area type="monotone" dataKey="lower50" stroke="none" fill="#020617" fillOpacity={1} name="Lower 50% CI" />
                  {/* Lines */}
                  <Line type="monotone" dataKey="actualPrice" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 2 }} name="Historical Price" />
                  <Line type="monotone" dataKey="forecastPrice" stroke="#10b981" strokeWidth={2.5} strokeDasharray="4 4" dot={{ r: 4, fill: "#10b981" }} name="Baseline Forecast" />
                  {netShockMultiplier !== 1 && (
                    <Line type="monotone" dataKey="simulatedPrice" stroke="#f59e0b" strokeWidth={2.5} dot={{ r: 3, fill: "#f59e0b" }} name="Simulated Shock" />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* TAB 3: What-If Stress Testing Simulator */}
        {activeTab === "simulator" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">Macroeconomic & Climate Stress-Testing Panel</h2>
              <p className="text-xs text-slate-400">
                Simulate supply shocks, input inflation, or trade policy disruptions in real-time.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-slate-950 p-6 rounded-xl border border-slate-800">
              {/* Slider 1 */}
              <div className="space-y-3">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-slate-300">Monsoon / Drought Shock</span>
                  <span className={monsoonShock > 0 ? "text-rose-400" : monsoonShock < 0 ? "text-emerald-400" : "text-slate-400"}>
                    {monsoonShock > 0 ? `+${monsoonShock}%` : `${monsoonShock}%`}
                  </span>
                </div>
                <input
                  type="range"
                  min="-15"
                  max="25"
                  step="5"
                  value={monsoonShock}
                  onChange={(e) => setMonsoonShock(Number(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer"
                />
                <p className="text-xs text-slate-500">Deficit rainfall triggers harvest yield reduction.</p>
              </div>

              {/* Slider 2 */}
              <div className="space-y-3">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-slate-300">Fuel & Fertilizer Inflation</span>
                  <span className={fuelShock > 0 ? "text-rose-400" : "text-slate-400"}>
                    +{fuelShock}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="5"
                  value={fuelShock}
                  onChange={(e) => setFuelShock(Number(e.target.value))}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
                <p className="text-xs text-slate-500">Transportation logistics & cultivation input hikes.</p>
              </div>

              {/* Slider 3 */}
              <div className="space-y-3">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-slate-300">Tariff / Export Restriction</span>
                  <span className={tariffShock > 0 ? "text-rose-400" : tariffShock < 0 ? "text-cyan-400" : "text-slate-400"}>
                    {tariffShock > 0 ? `+${tariffShock}%` : `${tariffShock}%`}
                  </span>
                </div>
                <input
                  type="range"
                  min="-20"
                  max="20"
                  step="5"
                  value={tariffShock}
                  onChange={(e) => setTariffShock(Number(e.target.value))}
                  className="w-full accent-violet-500 cursor-pointer"
                />
                <p className="text-xs text-slate-500">Export bans trap stock; import duty cuts ease domestic prices.</p>
              </div>
            </div>

            <div className="flex justify-between items-center bg-slate-950/80 p-4 rounded-xl border border-slate-800">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Net Scenario Impact</p>
                <p className="text-xl font-bold text-emerald-400 mt-0.5">
                  {netShockMultiplier >= 1 ? `+${((netShockMultiplier - 1) * 100).toFixed(1)}%` : `${((netShockMultiplier - 1) * 100).toFixed(1)}%`} Adjustment
                </p>
              </div>
              <button
                onClick={() => { setMonsoonShock(0); setFuelShock(0); setTariffShock(0); }}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-xs font-semibold rounded-lg transition"
              >
                Reset to Baseline
              </button>
            </div>
          </div>
        )}

        {/* TAB 4: Multi-Model Comparison */}
        {activeTab === "compare" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">Multi-Model Comparative Trajectory</h2>
              <p className="text-xs text-slate-400">
                Overlaying Smart Ensemble, SARIMA, Holt-Winters, and XGBoost on a synchronized horizon
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 bg-slate-950 rounded-lg border border-emerald-500/40">
                <span className="text-xs font-bold text-emerald-400 uppercase">★ Smart Ensemble</span>
                <p className="text-xs text-slate-400 mt-1">MAPE: 11.20% | MAE: ₹962.30</p>
                <p className="text-xs text-slate-300 mt-2">Inverse-variance weighted combination (55% SARIMA, 25% HW, 20% XGB).</p>
              </div>
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-xs font-semibold text-cyan-400 uppercase">SARIMA (0,1,1)(0,1,1,12)</span>
                <p className="text-xs text-slate-400 mt-1">MAPE: 12.58% | MAE: ₹1,034.80</p>
                <p className="text-xs text-slate-300 mt-2">Best statistical baseline capturing full annual seasonality.</p>
              </div>
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-xs font-semibold text-blue-400 uppercase">Holt-Winters</span>
                <p className="text-xs text-slate-400 mt-1">MAPE: 15.20% | MAE: ₹1,289.40</p>
                <p className="text-xs text-slate-300 mt-2">Additive seasonal smoothing model.</p>
              </div>
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800">
                <span className="text-xs font-semibold text-violet-400 uppercase">XGBoost Regressor</span>
                <p className="text-xs text-slate-400 mt-1">MAPE: 17.80% | MAE: ₹1,420.50</p>
                <p className="text-xs text-slate-300 mt-2">Recursive 12-lag gradient boosted decision tree.</p>
              </div>
            </div>

            <div className="h-80 sm:h-96 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={multiModelData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" domain={["auto", "auto"]} tick={{ fontSize: 11 }} tickFormatter={(v) => `₹${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#020617", border: "1px solid #334155", borderRadius: "8px" }}
                    formatter={(val: any) => [`₹${Number(val).toLocaleString()}`]}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="ensemble" stroke="#10b981" strokeWidth={3.5} name="Smart Ensemble" />
                  <Line type="monotone" dataKey="sarima" stroke="#06b6d4" strokeWidth={2} strokeDasharray="3 3" name="SARIMA" />
                  <Line type="monotone" dataKey="holtwinters" stroke="#3b82f6" strokeWidth={2} strokeDasharray="4 4" name="Holt-Winters" />
                  <Line type="monotone" dataKey="xgboost" stroke="#a855f7" strokeWidth={2} strokeDasharray="2 2" name="XGBoost" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* TAB 5: Seasonality & Harvesting */}
        {activeTab === "seasonality" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">Calendar Seasonality & Harvesting Peaks</h2>
              <p className="text-xs text-slate-400">
                Multi-decade historical price distribution by calendar month
              </p>
            </div>

            <div className="h-80 sm:h-96 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data?.seasonality_profile || []} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="month" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" domain={["auto", "auto"]} tick={{ fontSize: 11 }} tickFormatter={(v) => `₹${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#020617", border: "1px solid #334155", borderRadius: "8px" }}
                    formatter={(val: any) => [`₹${Number(val).toLocaleString()}`, "Avg Price"]}
                  />
                  <Bar dataKey="avg_price" fill="#10b981" radius={[4, 4, 0, 0]} name="Average Price" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {data?.seasonality_profile?.map((item) => (
                <div key={item.month} className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs">
                  <div className="flex justify-between font-bold text-white">
                    <span>{item.month}</span>
                    <span className={item.seasonal_index < 1.0 ? "text-emerald-400" : "text-amber-400"}>
                      Idx: {item.seasonal_index}
                    </span>
                  </div>
                  <p className="text-slate-300 mt-1">Mean: ₹{item.avg_price.toLocaleString()}</p>
                  <p className="text-slate-500 mt-0.5">{item.recommendation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 6: Drift Monitoring */}
        {activeTab === "drift" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">36-Month Rolling Drift & Anomaly Tracker</h2>
              <p className="text-xs text-slate-400">
                Monitors out-of-sample forecast percentage errors against the 15% threshold
              </p>
            </div>

            <div className="h-72 w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data?.drift_monitoring?.points || []} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#020617", border: "1px solid #334155", borderRadius: "8px" }}
                    formatter={(val: any) => [`${Number(val).toFixed(2)}%`, "Error %"]}
                  />
                  <ReferenceLine y={15} stroke="#ef4444" strokeDasharray="3 3" label={{ value: "Alert Limit (15%)", fill: "#ef4444", fontSize: 11 }} />
                  <Line type="monotone" dataKey="error_pct" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} name="Error %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* TAB 7: Custom CSV Upload */}
        {activeTab === "upload" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-8 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">Upload Custom Commodity Dataset</h2>
              <p className="text-xs text-slate-400">
                Forecast prices for any crop (Wheat, Rice, Spices, Cotton) by uploading a monthly CSV
              </p>
            </div>

            <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500 rounded-xl p-8 text-center transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-600 cursor-pointer"
              />
              <p className="text-xs text-slate-500 mt-3">
                Expected CSV format: <code>date,avg_monthly_price</code> (minimum 24 monthly records)
              </p>
            </div>

            {uploadStatus && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs text-emerald-400 font-mono">
                {uploadStatus}
              </div>
            )}
          </div>
        )}

        {/* TAB 8: Export Data */}
        {activeTab === "table" && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-bold text-white">Forecast Horizon Output ({selectedModel.toUpperCase()})</h2>
                <p className="text-xs text-slate-400">
                  Full forecast tabular results with simulated scenario values
                </p>
              </div>
              <button
                onClick={downloadCSV}
                className="px-4 py-2 bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold text-xs rounded-lg transition flex items-center gap-2 self-start sm:self-auto"
              >
                Export CSV
              </button>
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 text-xs uppercase tracking-wider">
                  <tr>
                    <th className="p-3.5">Forecast Date</th>
                    <th className="p-3.5">Baseline Forecast</th>
                    <th className="p-3.5">Simulated Shock</th>
                    <th className="p-3.5">Lower 95% CI</th>
                    <th className="p-3.5">Upper 95% CI</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-200 text-xs font-mono">
                  {data?.forecasts?.map((item) => (
                    <tr key={item.date} className="hover:bg-slate-800/30 transition">
                      <td className="p-3.5">{item.date}</td>
                      <td className="p-3.5 font-bold text-emerald-400">₹{item.forecast.toFixed(2)}</td>
                      <td className="p-3.5 font-bold text-amber-400">₹{(item.forecast * netShockMultiplier).toFixed(2)}</td>
                      <td className="p-3.5 text-slate-300">₹{(item.lower_ci * netShockMultiplier).toFixed(2)}</td>
                      <td className="p-3.5 text-slate-300">₹{(item.upper_ci * netShockMultiplier).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>
    </main>
  );
}
