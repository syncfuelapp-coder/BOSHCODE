import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, Database } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { motion } from "framer-motion";

const MLStats = ({ mlStats }) => {
  if (!mlStats) {
    return null;
  }

  const isActive = mlStats.status === "Active Learning";

  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="ml-stats">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          AI Learning System
        </h3>
        <Badge 
          variant="outline" 
          className={isActive ? "border-emerald-500/50 text-emerald-400" : "border-yellow-500/50 text-yellow-400"}
        >
          {mlStats.status}
        </Badge>
      </div>
      
      {!isActive ? (
        <div className="space-y-3">
          <div className="text-sm text-slate-400 mb-2">
            {mlStats.phase || "Learning Mode: The bot needs more trades to train the ML model."}
          </div>
          <div className="flex items-center gap-3">
            <Database className="w-4 h-4 text-slate-400" />
            <div className="flex-1">
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Trades Collected</span>
                <span>{10 - (mlStats.trades_needed || 10)} / 10</span>
              </div>
              <Progress value={(10 - (mlStats.trades_needed || 10)) * 10} className="h-2" />
            </div>
          </div>
          <div className="text-xs text-slate-500 italic">
            {mlStats.trades_needed || 10} more trades needed to activate self-learning
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800/30 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Total Trades</div>
              <div className="text-xl font-bold text-slate-200">
                {mlStats.total_trades_learned || 0}
              </div>
            </div>
            <div className="bg-slate-800/30 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">ML Accuracy</div>
              <div className="text-xl font-bold text-emerald-400">
                {((mlStats.accuracy || 0) * 100).toFixed(0)}%
              </div>
            </div>
            {mlStats.improvement !== undefined && (
              <div className="bg-slate-800/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Improvement</div>
                <div className={`text-xl font-bold ${mlStats.improvement >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {mlStats.improvement >= 0 ? '+' : ''}{mlStats.improvement}%
                </div>
              </div>
            )}
          </div>
          
          {mlStats.feature_importance && Object.keys(mlStats.feature_importance).length > 0 && (
            <div>
              <div className="text-xs text-slate-400 mb-2 flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                Feature Importance
              </div>
              <div className="space-y-2">
                {Object.entries(mlStats.feature_importance)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([feature, importance], idx) => (
                    <motion.div
                      key={feature}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      className="flex items-center gap-2"
                    >
                      <div className="text-xs text-slate-400 w-20 capitalize">
                        {feature.replace(/_/g, ' ')}
                      </div>
                      <div className="flex-1">
                        <Progress value={importance * 100} className="h-2" />
                      </div>
                      <div className="text-xs text-cyan-400 w-10 text-right">
                        {(importance * 100).toFixed(0)}%
                      </div>
                    </motion.div>
                  ))}
              </div>
            </div>
          )}
          
          <div className="text-xs text-slate-500 italic bg-purple-500/10 rounded p-2 border border-purple-500/20">
            🧠 AI is actively learning from each trade to improve predictions
          </div>
        </div>
      )}
    </Card>
  );
};

export default MLStats;
