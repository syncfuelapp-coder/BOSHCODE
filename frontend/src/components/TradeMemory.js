import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { History, TrendingUp, TrendingDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const TradeMemory = ({ trades }) => {
  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="trade-memory">
      <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <History className="w-5 h-5 text-cyan-400" />
        Trade Memory
      </h3>
      
      <div className="space-y-2">
        {trades.length === 0 ? (
          <p className="text-sm text-slate-500 italic">No trades yet...</p>
        ) : (
          <AnimatePresence>
            {trades.slice(0, 5).map((trade, idx) => (
              <motion.div
                key={trade.id}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ delay: idx * 0.05 }}
                className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/30"
                data-testid={`trade-${idx}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant={trade.result === "WIN" ? "default" : "destructive"}
                      className={trade.result === "WIN" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-red-500/20 text-red-400 border-red-500/30"}
                    >
                      {trade.action}
                    </Badge>
                    <span className="text-sm text-slate-300 font-medium">{trade.symbol}</span>
                  </div>
                  <div className={`text-sm font-bold flex items-center gap-1 ${
                    trade.profit_loss > 0 ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {trade.profit_loss > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                    {trade.profit_loss > 0 ? "+" : ""}£{trade.profit_loss?.toFixed(2)}
                  </div>
                </div>
                <div className="text-xs text-slate-400">
                  @${trade.price} | Confidence: {trade.ai_confidence?.toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 mt-1 truncate">
                  {trade.reason}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </Card>
  );
};

export default TradeMemory;
