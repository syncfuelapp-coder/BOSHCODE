import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { BarChart3 } from "lucide-react";

const ChartView = ({ marketData, currentMarket }) => {
  const chartData = marketData.slice(-30).map((item, idx) => ({
    time: idx,
    price: item.close,
    high: item.high,
    low: item.low
  }));
  
  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="chart-view">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          {currentMarket || "BTC/USD"}
        </h3>
        <div className="text-sm text-slate-400">
          Last 30 candles
        </div>
      </div>
      
      <div className="h-64">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500">
            Waiting for market data...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="time" 
                stroke="#475569"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
              />
              <YAxis 
                stroke="#475569"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                domain={['auto', 'auto']}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
                formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
              />
              <Area 
                type="monotone" 
                dataKey="price" 
                stroke="#06b6d4" 
                strokeWidth={2}
                fill="url(#colorPrice)"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
};

export default ChartView;
