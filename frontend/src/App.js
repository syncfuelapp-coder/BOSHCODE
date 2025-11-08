import { useState, useEffect } from "react";
import "@/App.css";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { Play, Square, Settings, RefreshCw, TrendingUp, TrendingDown, Activity, DollarSign, Target, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import SentimentGauge from "@/components/SentimentGauge";
import ChartView from "@/components/ChartView";
import TradeMemory from "@/components/TradeMemory";
import LiveConsole from "@/components/LiveConsole";
import CryptoRecommendations from "@/components/CryptoRecommendations";
import MLStats from "@/components/MLStats";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [botStatus, setBotStatus] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [settings, setSettings] = useState({
    mode: "demo",
    risk_per_trade: 2,
    current_market: "BTC/USD",
    ai_mode_enabled: true,
    sentiment_weight: 0.5
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [marketData, setMarketData] = useState([]);

  const fetchBotStatus = async () => {
    try {
      const response = await axios.get(`${API}/bot/status`);
      setBotStatus(response.data);
      setIsRunning(response.data.running);
    } catch (error) {
      console.error("Error fetching bot status:", error);
    }
  };

  const fetchMarketData = async () => {
    try {
      const response = await axios.get(`${API}/market/data`);
      setMarketData(response.data.data);
    } catch (error) {
      console.error("Error fetching market data:", error);
    }
  };

  useEffect(() => {
    fetchBotStatus();
    fetchMarketData();
    const interval = setInterval(() => {
      fetchBotStatus();
      if (isRunning) {
        fetchMarketData();
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleStart = async () => {
    try {
      await axios.post(`${API}/bot/start`);
      toast.success("Bot started successfully!");
      setIsRunning(true);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to start bot");
    }
  };

  const handleStop = async () => {
    try {
      await axios.post(`${API}/bot/stop`);
      toast.info("Bot stopped");
      setIsRunning(false);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to stop bot");
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API}/bot/reset`);
      toast.success("Bot reset to £100 balance");
      fetchBotStatus();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to reset bot");
    }
  };

  const handleSaveSettings = async () => {
    try {
      await axios.put(`${API}/bot/settings`, settings);
      toast.success("Settings saved");
      setSettingsOpen(false);
    } catch (error) {
      toast.error("Failed to save settings");
    }
  };

  const getProfitColor = (value) => {
    if (value > 0) return "text-emerald-400";
    if (value < 0) return "text-red-400";
    return "text-slate-400";
  };

  const getConfidenceColor = (value) => {
    if (value >= 70) return "from-emerald-500 to-green-600";
    if (value >= 40) return "from-cyan-500 to-blue-600";
    return "from-slate-500 to-gray-600";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Toaster position="top-right" />
      
      {/* Top Bar */}
      <div className="border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <motion.div
                animate={{ rotate: isRunning ? 360 : 0 }}
                transition={{ duration: 2, repeat: isRunning ? Infinity : 0, ease: "linear" }}
              >
                <Zap className="w-8 h-8 text-cyan-400" />
              </motion.div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                AetherBot
              </h1>
              <Badge 
                variant="outline" 
                className={`ml-2 ${
                  botStatus?.mode === "demo" 
                    ? "border-cyan-500/50 text-cyan-400" 
                    : "border-emerald-500/50 text-emerald-400"
                }`}
                data-testid="bot-mode-badge"
              >
                {botStatus?.mode?.toUpperCase() || "DEMO"}
              </Badge>
            </div>
            
            <div className="flex items-center gap-6">
              <div className="text-right">
                <div className="text-xs text-slate-400">Balance</div>
                <div className="text-xl font-bold text-slate-200" data-testid="bot-balance">
                  £{botStatus?.balance?.toFixed(2) || "100.00"}
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-xs text-slate-400">Profit/Loss</div>
                <div className={`text-xl font-bold ${getProfitColor(botStatus?.profit_loss || 0)}`} data-testid="bot-profit-loss">
                  {botStatus?.profit_loss >= 0 ? "+" : ""}£{botStatus?.profit_loss?.toFixed(2) || "0.00"}
                  <span className="text-sm ml-2">({botStatus?.profit_loss_pct?.toFixed(2) || "0.00"}%)</span>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-xs text-slate-400">AI Confidence</div>
                <div className="flex items-center gap-2">
                  <div className={`text-xl font-bold bg-gradient-to-r ${getConfidenceColor(botStatus?.ai_confidence || 0)} bg-clip-text text-transparent`} data-testid="ai-confidence">
                    {botStatus?.ai_confidence?.toFixed(0) || "0"}%
                  </div>
                  <Activity className="w-4 h-4 text-cyan-400" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Chart View */}
            <ChartView marketData={marketData} currentMarket={botStatus?.current_market} />
            
            {/* Live Console */}
            <LiveConsole logs={botStatus?.trade_logs || []} />
          </div>
          
          {/* Right Column */}
          <div className="space-y-6">
            {/* Performance Panel */}
            <Card className="bg-slate-900/50 border-slate-800/50 backdrop-blur-xl p-6" data-testid="performance-panel">
              <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                Performance
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Win Rate</span>
                  <span className="text-emerald-400 font-bold" data-testid="win-rate">{botStatus?.win_rate?.toFixed(0) || "0"}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Trades</span>
                  <span className="text-slate-200 font-bold" data-testid="trades-executed">{botStatus?.trades_executed || 0}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Wins</span>
                  <span className="text-emerald-400 font-bold">{botStatus?.wins || 0}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Losses</span>
                  <span className="text-red-400 font-bold">{botStatus?.losses || 0}</span>
                </div>
              </div>
            </Card>
            
            {/* Sentiment Gauge */}
            <SentimentGauge 
              sentiment={botStatus?.sentiment_score || 0} 
              headlines={botStatus?.sentiment_headlines || []}
            />
            
            {/* Trade Memory */}
            <TradeMemory trades={botStatus?.recent_trades || []} />
          </div>
        </div>
        
        {/* Control Bar */}
        <motion.div 
          className="fixed bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur-xl border-t border-slate-800/50 py-4"
          initial={{ y: 100 }}
          animate={{ y: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
          <div className="container mx-auto px-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                {!isRunning ? (
                  <Button
                    onClick={handleStart}
                    className="bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white font-bold px-6"
                    data-testid="start-bot-button"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Start Bot
                  </Button>
                ) : (
                  <Button
                    onClick={handleStop}
                    variant="destructive"
                    className="bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 font-bold px-6"
                    data-testid="stop-bot-button"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    Stop Bot
                  </Button>
                )}
                
                <Button
                  onClick={handleReset}
                  variant="outline"
                  className="border-slate-700 text-slate-300 hover:bg-slate-800"
                  disabled={isRunning}
                  data-testid="reset-bot-button"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 rounded-lg">
                  <Switch 
                    checked={settings.ai_mode_enabled}
                    onCheckedChange={(checked) => {
                      setSettings({...settings, ai_mode_enabled: checked});
                      axios.put(`${API}/bot/settings`, {...settings, ai_mode_enabled: checked});
                    }}
                    data-testid="ai-mode-toggle"
                  />
                  <span className="text-sm text-slate-300">AI Mode</span>
                </div>
                
                <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" className="border-slate-700 text-slate-300 hover:bg-slate-800" data-testid="settings-button">
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="bg-slate-900 border-slate-800 text-slate-200">
                    <DialogHeader>
                      <DialogTitle className="text-xl font-bold">Bot Settings</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-6 py-4">
                      <div>
                        <label className="text-sm text-slate-400 mb-2 block">Risk per Trade (%)</label>
                        <div className="flex items-center gap-4">
                          <Slider
                            value={[settings.risk_per_trade]}
                            onValueChange={([value]) => setSettings({...settings, risk_per_trade: value})}
                            min={0.5}
                            max={10}
                            step={0.5}
                            className="flex-1"
                          />
                          <span className="text-cyan-400 font-bold w-12 text-right">{settings.risk_per_trade}%</span>
                        </div>
                      </div>
                      
                      <div>
                        <label className="text-sm text-slate-400 mb-2 block">Sentiment Weight</label>
                        <div className="flex items-center gap-4">
                          <Slider
                            value={[settings.sentiment_weight]}
                            onValueChange={([value]) => setSettings({...settings, sentiment_weight: value})}
                            min={0}
                            max={1}
                            step={0.1}
                            className="flex-1"
                          />
                          <span className="text-cyan-400 font-bold w-12 text-right">{settings.sentiment_weight.toFixed(1)}</span>
                        </div>
                      </div>
                      
                      <Button 
                        onClick={handleSaveSettings}
                        className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                        data-testid="save-settings-button"
                      >
                        Save Settings
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default App;
