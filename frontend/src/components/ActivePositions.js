import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, TrendingUp, TrendingDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const ActivePositions = ({ positions, cryptoData }) => {
  const calculatePnL = (symbol, position) => {
    // Get current price from crypto data
    if (cryptoData && cryptoData[symbol] && cryptoData[symbol].market_data) {
      const marketData = cryptoData[symbol].market_data;
      if (marketData.length > 0) {
        const currentPrice = marketData[marketData.length - 1].close;
        const pnl = (currentPrice - position.entry_price) * position.amount;
        const pnlPct = ((currentPrice - position.entry_price) / position.entry_price) * 100;
        return { pnl, pnlPct, currentPrice };
      }
    }
    return { pnl: 0, pnlPct: 0, currentPrice: position.entry_price };
  };

  const positionsArray = Object.entries(positions || {});

  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="active-positions">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-400" />
          Active Positions
        </h3>
        <Badge variant="outline" className="border-purple-500/50 text-purple-400">
          {positionsArray.length} / 5
        </Badge>
      </div>
      
      <div className="space-y-3">
        {positionsArray.length === 0 ? (
          <div className="text-sm text-slate-500 italic text-center py-6">
            No open positions
            <div className="text-xs mt-2">Bot will open positions on strong signals</div>
          </div>
        ) : (
          <AnimatePresence>
            {positionsArray.map(([symbol, position], idx) => {
              const { pnl, pnlPct, currentPrice } = calculatePnL(symbol, position);
              const isProfit = pnl >= 0;
              
              return (
                <motion.div
                  key={symbol}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`bg-slate-800/30 rounded-lg p-4 border ${
                    isProfit ? "border-emerald-500/30" : "border-red-500/30"
                  }`}
                  data-testid={`position-${idx}`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-white font-bold text-lg">{symbol}</span>
                        <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
                          OPEN
                        </Badge>
                      </div>
                      <div className="text-xs text-slate-400">
                        Entry: ${position.entry_price.toFixed(6)} | Amount: {position.amount.toFixed(6)}
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        Invested: £{position.invested.toFixed(2)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-bold flex items-center gap-1 ${
                        isProfit ? "text-emerald-400" : "text-red-400"
                      }`}>
                        {isProfit ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                        {isProfit ? "+" : ""}£{pnl.toFixed(2)}
                      </div>
                      <div className={`text-sm ${isProfit ? "text-emerald-400" : "text-red-400"}`}>
                        {isProfit ? "+" : ""}{pnlPct.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between pt-3 border-t border-slate-700/30">
                    <div className="text-xs text-slate-400">
                      Current: ${currentPrice.toFixed(6)}
                    </div>
                    <div className="text-xs text-slate-500">
                      Opened: {new Date(position.opened_at).toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        )}
      </div>
      
      {positionsArray.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700/30 text-xs text-slate-500 text-center">
          Auto-closing at ±5% profit or -3% loss
        </div>
      )}
    </Card>
  );
};

export default ActivePositions;
