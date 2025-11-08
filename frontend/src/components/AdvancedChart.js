import { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";
import { Card } from "@/components/ui/card";
import { TrendingUp, BarChart3 } from "lucide-react";

const AdvancedChart = ({ marketData, currentMarket }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const candlestickSeriesRef = useRef();
  const volumeSeriesRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current || !marketData || marketData.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: "#334155",
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: "#06b6d4",
          width: 1,
          style: 2,
        },
        horzLine: {
          color: "#06b6d4",
          width: 1,
          style: 2,
        },
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#ef4444",
      borderUpColor: "#10b981",
      borderDownColor: "#ef4444",
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    });

    candlestickSeriesRef.current = candlestickSeries;

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: "#06b6d4",
      priceFormat: {
        type: "volume",
      },
      priceScaleId: "",
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    volumeSeriesRef.current = volumeSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current || !marketData || marketData.length === 0) {
      return;
    }

    // Format data for candlestick chart
    const candleData = marketData.map((item, idx) => ({
      time: Math.floor(Date.parse(item.timestamp) / 1000),
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    // Format data for volume
    const volumeData = marketData.map((item, idx) => ({
      time: Math.floor(Date.parse(item.timestamp) / 1000),
      value: item.volume,
      color: item.close >= item.open ? "#10b98140" : "#ef444440",
    }));

    candlestickSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [marketData]);

  // Calculate stats
  const getStats = () => {
    if (!marketData || marketData.length === 0) return null;

    const latest = marketData[marketData.length - 1];
    const first = marketData[0];
    const change = latest.close - first.close;
    const changePct = (change / first.close) * 100;

    return {
      price: latest.close,
      high: Math.max(...marketData.map((d) => d.high)),
      low: Math.min(...marketData.map((d) => d.low)),
      change,
      changePct,
      volume: latest.volume,
    };
  };

  const stats = getStats();

  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="advanced-chart">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-lg font-semibold text-slate-200">{currentMarket || "BTC/USD"}</h3>
            {stats && (
              <div className="flex items-center gap-3 mt-1">
                <span className="text-2xl font-bold text-white">
                  ${stats.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                </span>
                <span
                  className={`text-sm font-semibold flex items-center gap-1 ${
                    stats.change >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {stats.change >= 0 ? <TrendingUp className="w-4 h-4" /> : ""}
                  {stats.change >= 0 ? "+" : ""}
                  {stats.changePct.toFixed(2)}%
                </span>
              </div>
            )}
          </div>
        </div>
        {stats && (
          <div className="flex gap-6 text-sm">
            <div>
              <div className="text-slate-400 text-xs">High</div>
              <div className="text-emerald-400 font-semibold">${stats.high.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-slate-400 text-xs">Low</div>
              <div className="text-red-400 font-semibold">${stats.low.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-slate-400 text-xs">Volume</div>
              <div className="text-cyan-400 font-semibold">
                {(stats.volume / 1000000).toFixed(2)}M
              </div>
            </div>
          </div>
        )}
      </div>

      <div
        ref={chartContainerRef}
        style={{ position: "relative", height: "400px" }}
      />

      <div className="mt-3 flex items-center justify-center gap-4 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-emerald-500 rounded"></div>
          <span>Bullish Candle</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span>Bearish Candle</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-cyan-500 rounded"></div>
          <span>Volume</span>
        </div>
      </div>
    </Card>
  );
};

export default AdvancedChart;
