import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

const SentimentGauge = ({ sentiment, headlines }) => {
  const rotation = (sentiment + 1) * 90; // -1 to 1 maps to 0-180 degrees
  
  const getSentimentLabel = () => {
    if (sentiment > 0.3) return "BULLISH";
    if (sentiment < -0.3) return "BEARISH";
    return "NEUTRAL";
  };
  
  const getSentimentColor = () => {
    if (sentiment > 0.3) return "text-emerald-400";
    if (sentiment < -0.3) return "text-red-400";
    return "text-slate-400";
  };
  
  const getSentimentIcon = () => {
    if (sentiment > 0.3) return <TrendingUp className="w-5 h-5" />;
    if (sentiment < -0.3) return <TrendingDown className="w-5 h-5" />;
    return <Minus className="w-5 h-5" />;
  };
  
  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="sentiment-gauge">
      <h3 className="text-lg font-semibold text-slate-200 mb-4">Sentiment Analysis</h3>
      
      {/* Gauge */}
      <div className="relative w-48 h-24 mx-auto mb-6">
        <svg width="192" height="96" viewBox="0 0 192 96" className="overflow-visible">
          {/* Background arc */}
          <path
            d="M 16 80 A 80 80 0 0 1 176 80"
            fill="none"
            stroke="#334155"
            strokeWidth="12"
            strokeLinecap="round"
          />
          {/* Colored arc */}
          <path
            d="M 16 80 A 80 80 0 0 1 176 80"
            fill="none"
            stroke="url(#gradient)"
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray="251.2"
            strokeDashoffset={251.2 - (rotation / 180) * 251.2}
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="50%" stopColor="#64748b" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
          </defs>
        </svg>
        
        {/* Needle */}
        <motion.div
          className="absolute bottom-0 left-1/2 origin-bottom"
          style={{
            width: "3px",
            height: "70px",
            background: "linear-gradient(to top, #06b6d4, #3b82f6)",
            marginLeft: "-1.5px",
            borderRadius: "2px"
          }}
          animate={{ rotate: rotation - 90 }}
          transition={{ type: "spring", stiffness: 100, damping: 15 }}
        />
        
        {/* Center dot */}
        <div className="absolute bottom-0 left-1/2 w-3 h-3 bg-cyan-400 rounded-full transform -translate-x-1/2" />
      </div>
      
      {/* Sentiment Label */}
      <div className="text-center mb-6">
        <div className={`text-2xl font-bold flex items-center justify-center gap-2 ${getSentimentColor()}`} data-testid="sentiment-label">
          {getSentimentIcon()}
          {getSentimentLabel()}
        </div>
        <div className="text-slate-400 text-sm mt-1" data-testid="sentiment-score">
          Score: {sentiment?.toFixed(2) || "0.00"}
        </div>
      </div>
      
      {/* Headlines */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Recent News</h4>
        {headlines.length === 0 ? (
          <p className="text-sm text-slate-500 italic">No headlines yet...</p>
        ) : (
          headlines.map((headline, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/30"
              data-testid={`sentiment-headline-${idx}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <div className="text-xs text-cyan-400 mb-1">{headline.source}</div>
                  <div className="text-sm text-slate-300 leading-tight">{headline.headline}</div>
                </div>
                <div className={`text-xs font-bold ${
                  headline.sentiment > 0 ? "text-emerald-400" : headline.sentiment < 0 ? "text-red-400" : "text-slate-400"
                }`}>
                  {headline.sentiment > 0 ? "+" : ""}{headline.sentiment?.toFixed(1)}
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </Card>
  );
};

export default SentimentGauge;
