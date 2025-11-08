import { Card } from "@/components/ui/card";
import { ComposedChart, Line, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Area } from "recharts";
import { TrendingUp, TrendingDown, BarChart3 } from "lucide-react";

const AdvancedChart = ({ marketData, currentMarket }) => {
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

  // Prepare data for candlestick visualization
  const chartData = (marketData || []).slice(-50).map((item, idx) => ({
    time: idx,
    price: item.close,
    high: item.high,
    low: item.low,
    open: item.open,
    volume: item.volume / 100000, // Scale down for visibility
    color: item.close >= item.open ? "#10b981" : "#ef4444",
    wickTop: item.high,
    wickBottom: item.low,
    candleTop: Math.max(item.open, item.close),
    candleBottom: Math.min(item.open, item.close),
  }));

  // Custom candlestick shape
  const CandleStick = (props) => {
    const { x, y, width, height, payload } = props;
    if (!payload) return null;

    const wickX = x + width / 2;
    const wickWidth = 2;
    const candleWidth = Math.max(width * 0.6, 4);
    const candleX = x + (width - candleWidth) / 2;

    // Scale factor for price to pixel conversion
    const priceRange = Math.max(...chartData.map(d => d.high)) - Math.min(...chartData.map(d => d.low));
    const chartHeight = 300;
    const scale = chartHeight / priceRange;

    const wickTopY = y - (payload.wickTop - payload.price) * scale;
    const wickBottomY = y + (payload.price - payload.wickBottom) * scale;
    const candleTopY = y - (payload.candleTop - payload.price) * scale;
    const candleBottomY = y + (payload.price - payload.candleBottom) * scale;
    const candleHeight = Math.abs(candleBottomY - candleTopY);

    return (
      <g>
        {/* Wick */}
        <line
          x1={wickX}
          y1={wickTopY}
          x2={wickX}
          y2={wickBottomY}
          stroke={payload.color}
          strokeWidth={wickWidth}
        />
        {/* Candle body */}
        <rect
          x={candleX}
          y={Math.min(candleTopY, candleBottomY)}
          width={candleWidth}
          height={Math.max(candleHeight, 2)}
          fill={payload.color}
          stroke={payload.color}
          strokeWidth={1}
        />
      </g>
    );
  };

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
                  ${stats.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                <span
                  className={`text-sm font-semibold flex items-center gap-1 ${
                    stats.change >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {stats.change >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
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
                {(stats.volume / 1000000).toFixed(1)}M
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="h-80">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>Loading market data...</p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="time" 
                stroke="#475569"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }}
              />
              <YAxis 
                stroke="#475569"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                domain={['auto', 'auto']}
                label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
                formatter={(value, name) => {
                  if (name === 'volume') return [(value * 100000 / 1000000).toFixed(2) + 'M', 'Volume'];
                  return ['$' + Number(value).toFixed(2), 'Price'];
                }}
              />
              {/* Volume bars at bottom */}
              <Bar 
                dataKey="volume" 
                fill="#06b6d480"
                radius={[2, 2, 0, 0]}
              />
              {/* Price line with area fill */}
              <Area 
                type="monotone" 
                dataKey="price" 
                stroke="#06b6d4" 
                strokeWidth={2}
                fill="url(#colorPrice)"
              />
              {/* High/Low range */}
              <Line 
                type="monotone" 
                dataKey="high" 
                stroke="#10b98140" 
                strokeWidth={1}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="low" 
                stroke="#ef444440" 
                strokeWidth={1}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="mt-4 flex items-center justify-center gap-6 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-cyan-500 rounded"></div>
          <span>Price Line</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-emerald-500 opacity-50 rounded"></div>
          <span>High Range</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 opacity-50 rounded"></div>
          <span>Low Range</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-cyan-500 opacity-50 rounded"></div>
          <span>Volume</span>
        </div>
      </div>
    </Card>
  );
};

export default AdvancedChart;
