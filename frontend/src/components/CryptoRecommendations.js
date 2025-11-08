import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrendingUp, Sparkles, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { motion } from "framer-motion";

const CryptoRecommendations = ({ recommendations, onSelectCrypto, currentMarket }) => {
  const getRecommendationColor = (recommendation) => {
    if (recommendation === "STRONG BUY") return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    if (recommendation === "BUY") return "bg-cyan-500/20 text-cyan-400 border-cyan-500/30";
    if (recommendation === "HOLD") return "bg-slate-500/20 text-slate-400 border-slate-500/30";
    return "bg-red-500/20 text-red-400 border-red-500/30";
  };

  const getScoreColor = (score) => {
    if (score >= 80) return "text-emerald-400";
    if (score >= 60) return "text-cyan-400";
    if (score >= 40) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="crypto-recommendations">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          AI Crypto Recommendations
        </h3>
        <Badge variant="outline" className="border-cyan-500/50 text-cyan-400">
          Live Analysis
        </Badge>
      </div>
      
      <div className="space-y-3">
        {recommendations.length === 0 ? (
          <div className="text-sm text-slate-500 italic text-center py-4">
            Loading recommendations...
          </div>
        ) : (
          recommendations.map((rec, idx) => (
            <motion.div
              key={rec.symbol}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`bg-slate-800/30 rounded-lg p-4 border ${
                currentMarket === rec.symbol ? "border-cyan-500/50" : "border-slate-700/30"
              } hover:border-cyan-500/30 transition-all cursor-pointer`}
              onClick={() => onSelectCrypto(rec.symbol)}
              data-testid={`crypto-recommendation-${idx}`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-white font-bold">{rec.name}</span>
                    <span className="text-xs text-slate-400">{rec.symbol}</span>
                    {currentMarket === rec.symbol && (
                      <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 text-xs">
                        Active
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge className={getRecommendationColor(rec.recommendation)}>
                      {rec.recommendation}
                    </Badge>
                    <div className="flex items-center gap-1 text-sm">
                      {rec.sentiment > 0 ? (
                        <ArrowUpRight className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <ArrowDownRight className="w-3 h-3 text-red-400" />
                      )}
                      <span className={rec.sentiment > 0 ? "text-emerald-400" : "text-red-400"}>
                        {rec.sentiment > 0 ? "+" : ""}{rec.sentiment.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-400 mb-1">Opportunity</div>
                  <div className={`text-2xl font-bold ${getScoreColor(rec.opportunity_score)}`}>
                    {rec.opportunity_score}
                  </div>
                </div>
              </div>
              
              {rec.headlines && rec.headlines.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-700/30">
                  <div className="text-xs text-slate-500 mb-1">Latest News:</div>
                  <div className="text-xs text-slate-400 line-clamp-2">
                    {rec.headlines[0]?.headline}
                  </div>
                </div>
              )}
            </motion.div>
          ))
        )}
      </div>
      
      <div className="mt-4 pt-4 border-t border-slate-700/30 text-xs text-slate-500 text-center">
        Powered by AI analysis of worldwide crypto news
      </div>
    </Card>
  );
};

export default CryptoRecommendations;
