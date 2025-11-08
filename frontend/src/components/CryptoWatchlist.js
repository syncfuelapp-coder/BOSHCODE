import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Eye, TrendingUp, TrendingDown, ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const CryptoWatchlist = ({ recommendations }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const displayCount = isExpanded ? recommendations.length : 5;
  const visibleCryptos = recommendations.slice(0, displayCount);

  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-4" data-testid="crypto-watchlist">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-semibold text-slate-200">
            Watching {recommendations.length} Cryptos
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 text-xs">
            Smart Filter
          </Badge>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </div>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="mt-3 space-y-2 max-h-60 overflow-y-auto">
              {visibleCryptos.map((crypto, idx) => (
                <motion.div
                  key={crypto.symbol}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex items-center justify-between p-2 bg-slate-800/30 rounded border border-slate-700/30 hover:border-cyan-500/30 transition-all"
                >
                  <div className="flex items-center gap-2 flex-1">
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-slate-200">{crypto.symbol}</span>
                      <span className="text-xs text-slate-500">{crypto.name}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      {crypto.sentiment >= 0 ? (
                        <TrendingUp className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <TrendingDown className="w-3 h-3 text-red-400" />
                      )}
                      <span className={`text-xs font-semibold ${
                        crypto.sentiment >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}>
                        {crypto.sentiment >= 0 ? "+" : ""}{crypto.sentiment.toFixed(2)}
                      </span>
                    </div>
                    
                    <Badge 
                      className={`text-xs ${
                        crypto.recommendation === "STRONG BUY" 
                          ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                          : crypto.recommendation === "BUY"
                          ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/30"
                          : crypto.recommendation === "HOLD"
                          ? "bg-slate-500/20 text-slate-400 border-slate-500/30"
                          : "bg-red-500/20 text-red-400 border-red-500/30"
                      }`}
                    >
                      {crypto.opportunity_score.toFixed(0)}
                    </Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {!isExpanded && (
        <div className="mt-2 text-xs text-slate-500 text-center">
          Click to expand watchlist
        </div>
      )}
    </Card>
  );
};

export default CryptoWatchlist;
