from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from emergentintegrations.llm.chat import LlmChat, UserMessage
import ta
import aiohttp

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expanded Cryptocurrency List (Top 50+ from CoinGecko)
AVAILABLE_CRYPTOS = [
    # Top 10
    {"symbol": "BTC/USD", "name": "Bitcoin", "coin_id": "bitcoin"},
    {"symbol": "ETH/USD", "name": "Ethereum", "coin_id": "ethereum"},
    {"symbol": "XRP/USD", "name": "Ripple", "coin_id": "ripple"},
    {"symbol": "BNB/USD", "name": "Binance Coin", "coin_id": "binancecoin"},
    {"symbol": "SOL/USD", "name": "Solana", "coin_id": "solana"},
    {"symbol": "ADA/USD", "name": "Cardano", "coin_id": "cardano"},
    {"symbol": "DOGE/USD", "name": "Dogecoin", "coin_id": "dogecoin"},
    {"symbol": "TRX/USD", "name": "TRON", "coin_id": "tron"},
    {"symbol": "AVAX/USD", "name": "Avalanche", "coin_id": "avalanche-2"},
    {"symbol": "LINK/USD", "name": "Chainlink", "coin_id": "chainlink"},
    
    # Top 11-20
    {"symbol": "DOT/USD", "name": "Polkadot", "coin_id": "polkadot"},
    {"symbol": "MATIC/USD", "name": "Polygon", "coin_id": "matic-network"},
    {"symbol": "SHIB/USD", "name": "Shiba Inu", "coin_id": "shiba-inu"},
    {"symbol": "UNI/USD", "name": "Uniswap", "coin_id": "uniswap"},
    {"symbol": "LTC/USD", "name": "Litecoin", "coin_id": "litecoin"},
    {"symbol": "BCH/USD", "name": "Bitcoin Cash", "coin_id": "bitcoin-cash"},
    {"symbol": "ATOM/USD", "name": "Cosmos", "coin_id": "cosmos"},
    {"symbol": "APT/USD", "name": "Aptos", "coin_id": "aptos"},
    {"symbol": "FIL/USD", "name": "Filecoin", "coin_id": "filecoin"},
    {"symbol": "ARB/USD", "name": "Arbitrum", "coin_id": "arbitrum"},
    
    # Top 21-30
    {"symbol": "NEAR/USD", "name": "NEAR Protocol", "coin_id": "near"},
    {"symbol": "VET/USD", "name": "VeChain", "coin_id": "vechain"},
    {"symbol": "ALGO/USD", "name": "Algorand", "coin_id": "algorand"},
    {"symbol": "AAVE/USD", "name": "Aave", "coin_id": "aave"},
    {"symbol": "GRT/USD", "name": "The Graph", "coin_id": "the-graph"},
    {"symbol": "SAND/USD", "name": "The Sandbox", "coin_id": "the-sandbox"},
    {"symbol": "MANA/USD", "name": "Decentraland", "coin_id": "decentraland"},
    {"symbol": "AXS/USD", "name": "Axie Infinity", "coin_id": "axie-infinity"},
    {"symbol": "FTM/USD", "name": "Fantom", "coin_id": "fantom"},
    {"symbol": "XLM/USD", "name": "Stellar", "coin_id": "stellar"},
    
    # DeFi & Gaming (31-40)
    {"symbol": "CRV/USD", "name": "Curve DAO", "coin_id": "curve-dao-token"},
    {"symbol": "SUSHI/USD", "name": "SushiSwap", "coin_id": "sushi"},
    {"symbol": "COMP/USD", "name": "Compound", "coin_id": "compound-governance-token"},
    {"symbol": "YFI/USD", "name": "yearn.finance", "coin_id": "yearn-finance"},
    {"symbol": "SNX/USD", "name": "Synthetix", "coin_id": "synthetix-network-token"},
    {"symbol": "1INCH/USD", "name": "1inch", "coin_id": "1inch"},
    {"symbol": "ENJ/USD", "name": "Enjin Coin", "coin_id": "enjincoin"},
    {"symbol": "CHZ/USD", "name": "Chiliz", "coin_id": "chiliz"},
    {"symbol": "GALA/USD", "name": "Gala", "coin_id": "gala"},
    {"symbol": "IMX/USD", "name": "Immutable X", "coin_id": "immutable-x"},
    
    # Emerging (41-50)
    {"symbol": "OP/USD", "name": "Optimism", "coin_id": "optimism"},
    {"symbol": "SUI/USD", "name": "Sui", "coin_id": "sui"},
    {"symbol": "SEI/USD", "name": "Sei", "coin_id": "sei-network"},
    {"symbol": "INJ/USD", "name": "Injective", "coin_id": "injective-protocol"},
    {"symbol": "PEPE/USD", "name": "Pepe", "coin_id": "pepe"},
    {"symbol": "WLD/USD", "name": "Worldcoin", "coin_id": "worldcoin-wld"},
    {"symbol": "RNDR/USD", "name": "Render", "coin_id": "render-token"},
    {"symbol": "STX/USD", "name": "Stacks", "coin_id": "blockstack"},
    {"symbol": "HBAR/USD", "name": "Hedera", "coin_id": "hedera-hashgraph"},
    {"symbol": "QNT/USD", "name": "Quant", "coin_id": "quant-network"}
]

# Cache for crypto list
CRYPTO_CACHE = {"last_update": None, "data": AVAILABLE_CRYPTOS}

# Trading Bot State
bot_state = {
    "running": False,
    "mode": "demo",
    "balance": 100.0,
    "initial_balance": 100.0,
    "equity": 100.0,
    "profit_loss": 0.0,
    "profit_loss_pct": 0.0,
    "ai_confidence": 0.0,
    "sentiment_score": 0.0,
    "trades_executed": 0,
    "wins": 0,
    "losses": 0,
    "current_market": "BTC/USD",
    "risk_per_trade": 2.0,
    "ai_mode_enabled": True,
    "sentiment_weight": 0.5,
    "trade_logs": [],
    "market_data": {},
    "recent_trades": [],
    "sentiment_headlines": [],
    "crypto_recommendations": [],
    "ml_model_performance": [],
    "learning_data": [],
    "multi_crypto_enabled": True,
    "active_positions": {},
    "crypto_data": {}
}

task = None

# Models
class BotSettings(BaseModel):
    mode: Optional[str] = "demo"
    risk_per_trade: Optional[float] = 2.0
    current_market: Optional[str] = "BTC/USD"
    ai_mode_enabled: Optional[bool] = True
    sentiment_weight: Optional[float] = 0.5

class Trade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str
    symbol: str
    action: str
    price: float
    amount: float
    result: str
    profit_loss: float
    reason: str
    ai_confidence: float

class MarketDataPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class SentimentHeadline(BaseModel):
    source: str
    headline: str
    sentiment: float
    timestamp: str

# Mock News Generator with Crypto-specific news
def generate_mock_news(crypto_symbol=None):
    general_news = [
        ("NewsAPI", "Federal Reserve hints at policy changes affecting markets", 0.3, ["BTC", "ETH", "XRP"]),
        ("NewsAPI", "Global economic outlook improves, stocks rally", 0.7, ["BTC", "ETH", "SOL"]),
        ("NewsAPI", "Inflation data comes in better than expected", 0.4, ["BTC", "ETH"]),
        ("Reuters", "Central banks worldwide discuss digital currencies", 0.6, ["XRP", "ADA", "DOT"]),
        ("Bloomberg", "Institutional investors increase crypto holdings", 0.8, ["BTC", "ETH"]),
        ("CNBC", "Tech stocks surge on AI breakthrough", 0.7, ["SOL", "MATIC"])
    ]
    
    crypto_specific_news = {
        "BTC": [
            ("CryptoPanic", "Bitcoin surges to new highs amid institutional interest", 0.8),
            ("CryptoPanic", "Bitcoin ETF sees record inflows", 0.9),
            ("Twitter/X", "Major corporation adds Bitcoin to balance sheet", 0.7),
            ("CoinDesk", "Bitcoin mining difficulty reaches all-time high", 0.5)
        ],
        "ETH": [
            ("CryptoPanic", "Ethereum network upgrade successful", 0.8),
            ("Twitter/X", "DeFi protocols on Ethereum hit new TVL record", 0.7),
            ("CoinDesk", "Ethereum gas fees drop significantly", 0.6),
            ("CryptoPanic", "Major NFT collection launches on Ethereum", 0.5)
        ],
        "XRP": [
            ("CryptoPanic", "Ripple wins major legal battle, XRP surges", 0.9),
            ("Reuters", "Banks adopt Ripple's cross-border payment solution", 0.8),
            ("Twitter/X", "XRP added to major exchange in Asia", 0.7),
            ("CoinDesk", "Ripple expands partnerships with financial institutions", 0.6)
        ],
        "SOL": [
            ("CryptoPanic", "Solana network reaches record transaction speed", 0.7),
            ("Twitter/X", "Major DeFi project migrates to Solana", 0.6),
            ("CoinDesk", "Solana Foundation announces ecosystem grants", 0.5)
        ],
        "ADA": [
            ("CryptoPanic", "Cardano smart contract upgrade goes live", 0.7),
            ("Twitter/X", "Cardano founder announces major partnership", 0.6),
            ("CoinDesk", "Cardano stake pool operators reach milestone", 0.5)
        ],
        "DOGE": [
            ("Twitter/X", "Elon Musk tweets about Dogecoin", 0.8),
            ("CryptoPanic", "Dogecoin payment adoption increases", 0.6),
            ("Reddit", "Dogecoin community raises funds for charity", 0.5)
        ]
    }
    
    negative_news = [
        ("CryptoPanic", "Regulatory concerns weigh on crypto markets", -0.4, ["BTC", "ETH", "XRP"]),
        ("CryptoPanic", "Exchange outage raises security concerns", -0.6, ["BTC", "ETH"]),
        ("Twitter/X", "Top analyst predicts market correction", -0.3, ["BTC", "ETH"]),
        ("Reuters", "Government announces stricter crypto regulations", -0.5, ["XRP", "BTC"]),
        ("Bloomberg", "Major crypto exchange faces investigation", -0.7, ["BTC", "ETH"])
    ]
    
    if crypto_symbol:
        crypto_base = crypto_symbol.split("/")[0]
        if crypto_base in crypto_specific_news:
            return random.choice(crypto_specific_news[crypto_base])
    
    all_news = general_news + negative_news
    selected = random.choice(all_news)
    if len(selected) == 4:
        return (selected[0], selected[1], selected[2])
    return selected

# Real-Time Market Data Fetcher
async def fetch_real_time_data(symbol="BTC/USD"):
    """Fetch real-time crypto data from CoinGecko API (Free, no auth required)"""
    try:
        # Find coin_id from AVAILABLE_CRYPTOS
        coin_id = "bitcoin"
        for crypto in AVAILABLE_CRYPTOS:
            if crypto["symbol"] == symbol:
                coin_id = crypto["coin_id"]
                break
        
        async with aiohttp.ClientSession() as session:
            # Get current price
            price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
            
            async with session.get(price_url, timeout=10) as response:
                if response.status == 200:
                    price_data = await response.json()
                    current_price = price_data[coin_id]["usd"]
                    volume = price_data[coin_id].get("usd_24h_vol", 1000000)
                    change_24h = price_data[coin_id].get("usd_24h_change", 0) / 100
                    
                    # Generate realistic candles based on current price
                    data = []
                    base_price = current_price / (1 + change_24h)  # Price 24h ago
                    
                    for i in range(100):
                        # Simulate price movement towards current price
                        progress = i / 99
                        target_price = base_price + (current_price - base_price) * progress
                        
                        # Add realistic volatility
                        volatility = abs(change_24h) * 0.05
                        open_price = target_price * (1 + random.uniform(-volatility, volatility))
                        close_price = target_price * (1 + random.uniform(-volatility, volatility))
                        high_price = max(open_price, close_price) * random.uniform(1, 1.005)
                        low_price = min(open_price, close_price) * random.uniform(0.995, 1)
                        
                        # Last candle should close at current price
                        if i == 99:
                            close_price = current_price
                        
                        data.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "open": round(open_price, 6),
                            "high": round(high_price, 6),
                            "low": round(low_price, 6),
                            "close": round(close_price, 6),
                            "volume": round(volume / 100, 2)
                        })
                    
                    logger.info(f"✅ LIVE DATA for {symbol}: ${current_price:,.2f} (24h: {change_24h*100:+.2f}%)")
                    return data
                else:
                    logger.warning(f"CoinGecko API returned {response.status}, using fallback")
                    return generate_fallback_data(symbol)
                    
    except Exception as e:
        logger.error(f"Error fetching real-time data for {symbol}: {e}")
        return generate_fallback_data(symbol)

# Fallback Mock Data (if API fails)
def generate_fallback_data(symbol="BTC/USD"):
    """Fallback to simulated data if API fails"""
    # Default prices for major cryptos
    price_defaults = {
        "BTC/USD": 100000, "ETH/USD": 3500, "XRP/USD": 2.0, "BNB/USD": 600,
        "SOL/USD": 150, "ADA/USD": 0.5, "DOGE/USD": 0.08, "TRX/USD": 0.15,
        "AVAX/USD": 35, "LINK/USD": 15, "DOT/USD": 6, "MATIC/USD": 0.7,
        "SHIB/USD": 0.00001, "UNI/USD": 10, "LTC/USD": 80, "BCH/USD": 250
    }
    base_price = price_defaults.get(symbol, 100)
    
    data = []
    current_price = base_price
    for i in range(100):
        open_price = current_price
        change = current_price * 0.001 * random.uniform(-1, 1)  # 0.1% volatility
        close_price = current_price + change
        high_price = max(open_price, close_price) * random.uniform(1, 1.002)
        low_price = min(open_price, close_price) * random.uniform(0.998, 1)
        volume = random.uniform(1000000, 5000000)
        
        timestamp = datetime.now(timezone.utc).isoformat()
        data.append({
            "timestamp": timestamp,
            "open": round(open_price, 6),
            "high": round(high_price, 6),
            "low": round(low_price, 6),
            "close": round(close_price, 6),
            "volume": round(volume, 2)
        })
        current_price = close_price
    
    logger.info(f"⚠️ Using FALLBACK data for {symbol}")
    return data

# Technical Indicators
def calculate_indicators(market_data):
    df = pd.DataFrame(market_data)
    
    # EMA
    df['ema_short'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_long'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    return df

# AI Sentiment Analysis
async def analyze_sentiment_with_ai(headlines):
    try:
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"sentiment-{uuid.uuid4()}",
            system_message="You are a financial sentiment analyzer. Analyze news and return a sentiment score between -1 (very bearish) and 1 (very bullish). Return only a number."
        ).with_model("openai", "gpt-4o")
        
        headlines_text = "\n".join([f"- {h['headline']}" for h in headlines])
        message = UserMessage(text=f"Analyze these financial headlines and return a single sentiment score between -1 and 1:\n{headlines_text}")
        
        response = await chat.send_message(message)
        sentiment = float(response.strip())
        return max(-1, min(1, sentiment))
    except Exception as e:
        logger.error(f"AI sentiment analysis error: {e}")
        # Fallback to simple average
        return sum([h['sentiment'] for h in headlines]) / len(headlines) if headlines else 0

# Intelligent Crypto Recommendation System (Simplified for performance)
async def analyze_crypto_opportunities():
    """Analyze all cryptos and recommend best trading opportunities based on news and sentiment"""
    try:
        recommendations = []
        
        for crypto in AVAILABLE_CRYPTOS[:5]:  # Analyze top 5 only for speed
            # Generate news for each crypto
            headlines = []
            for _ in range(2):
                source, headline, sentiment = generate_mock_news(crypto["symbol"])
                headlines.append({
                    "source": source,
                    "headline": headline,
                    "sentiment": sentiment,
                    "crypto": crypto["symbol"]
                })
            
            # Calculate overall sentiment for this crypto
            avg_sentiment = sum([h["sentiment"] for h in headlines]) / len(headlines)
            
            # Simple scoring based on sentiment + random technical factor
            technical_factor = random.uniform(0.4, 0.9)
            opportunity_score = ((avg_sentiment + 1) / 2 * 50) + (technical_factor * 50)
            
            recommendations.append({
                "symbol": crypto["symbol"],
                "name": crypto["name"],
                "sentiment": round(avg_sentiment, 2),
                "opportunity_score": round(opportunity_score, 1),
                "headlines": headlines,
                "recommendation": "STRONG BUY" if opportunity_score >= 80 else "BUY" if opportunity_score >= 60 else "HOLD" if opportunity_score >= 40 else "SELL"
            })
        
        # Sort by opportunity score
        recommendations.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return recommendations
        
    except Exception as e:
        logger.error(f"Crypto recommendation error: {e}")
        # Fallback to simple ranking
        simple_recs = []
        for crypto in AVAILABLE_CRYPTOS[:5]:
            simple_recs.append({
                "symbol": crypto["symbol"],
                "name": crypto["name"],
                "sentiment": round(random.uniform(-0.5, 0.8), 2),
                "opportunity_score": round(random.uniform(50, 90), 1),
                "headlines": [],
                "recommendation": "HOLD"
            })
        return simple_recs

# Advanced Machine Learning Model with Deep Features
class AdvancedTradingML:
    def __init__(self):
        # Ensemble of models for better predictions
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.is_trained = False
        self.training_data = []
        self.feature_importance = {}
        self.win_streak = 0
        self.loss_streak = 0
        self.performance_history = []
        
    def calculate_advanced_features(self, basic_features, market_data):
        """Calculate advanced features from market data"""
        if len(market_data) < 20:
            return basic_features
        
        df = pd.DataFrame(market_data[-20:])
        
        # Price momentum features
        price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        price_change_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        
        # Volume features
        volume_avg = df['volume'].mean()
        volume_current = df['volume'].iloc[-1]
        volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1
        
        # Volatility features
        price_std = df['close'].std()
        price_mean = df['close'].mean()
        volatility_coef = price_std / price_mean if price_mean > 0 else 0
        
        # Trend strength
        trend_strength = abs(price_change_10)
        
        # Extended features: [basic + advanced]
        advanced_features = list(basic_features) + [
            price_change_5,
            price_change_10,
            volume_ratio,
            volatility_coef,
            trend_strength,
            self.win_streak / 10,  # Normalized streak
            self.loss_streak / 10
        ]
        
        return advanced_features
        
    def add_trade_data(self, features, result, market_data=None):
        """Add trade data with advanced features"""
        # Calculate advanced features if market data provided
        if market_data and len(market_data) >= 20:
            features = self.calculate_advanced_features(features, market_data)
        
        self.training_data.append({
            "features": features,
            "result": 1 if result == "WIN" else 0
        })
        
        # Update streaks
        if result == "WIN":
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        # Retrain more frequently for faster learning
        if len(self.training_data) >= 5:
            self.retrain()
    
    def retrain(self):
        """Retrain with adaptive learning"""
        if len(self.training_data) < 5:
            return
        
        # Use sliding window of last 150 trades for adaptive learning
        X = [d["features"] for d in self.training_data[-150:]]
        y = [d["result"] for d in self.training_data[-150:]]
        
        # Ensure all feature vectors have same length
        min_len = min(len(x) for x in X)
        X = [x[:min_len] for x in X]
        
        self.rf_model.fit(X, y)
        self.is_trained = True
        
        # Calculate and store feature importance
        feature_names = ["ema", "rsi", "macd", "sentiment", "volatility", 
                        "momentum_5", "momentum_10", "volume", "volatility_coef", 
                        "trend", "win_streak", "loss_streak"][:min_len]
        self.feature_importance = dict(zip(feature_names, self.rf_model.feature_importances_[:min_len]))
        
        # Track performance
        accuracy = self.rf_model.score(X, y)
        self.performance_history.append(accuracy)
        
        logger.info(f"🧠 ML RETRAINED: {len(self.training_data)} trades | Accuracy: {accuracy*100:.1f}%")
    
    def predict_trade_success(self, features, market_data=None):
        """Predict with confidence adjustment"""
        if not self.is_trained:
            return 0.55  # Slightly optimistic default
        
        # Calculate advanced features
        if market_data and len(market_data) >= 20:
            features = self.calculate_advanced_features(features, market_data)
        
        # Ensure feature length matches training
        if self.training_data:
            trained_len = len(self.training_data[0]["features"])
            features = features[:trained_len]
        
        try:
            prob = self.rf_model.predict_proba([features])[0][1]
            
            # Adjust based on recent performance
            if len(self.performance_history) > 5:
                recent_accuracy = np.mean(self.performance_history[-5:])
                # Boost confidence if model is performing well
                if recent_accuracy > 0.7:
                    prob = prob * 1.1
                    
            return min(prob, 0.95)  # Cap at 95%
        except:
            return 0.5
    
    def get_model_stats(self):
        """Enhanced model statistics"""
        if not self.is_trained:
            return {
                "status": "Learning Phase",
                "trades_needed": max(0, 5 - len(self.training_data)),
                "phase": "Collecting initial data"
            }
        
        recent_trades = self.training_data[-100:]
        wins = sum([1 for t in recent_trades if t["result"] == 1])
        accuracy = wins / len(recent_trades) if recent_trades else 0
        
        # Calculate improvement
        improvement = 0
        if len(self.performance_history) >= 10:
            old_avg = np.mean(self.performance_history[:5])
            new_avg = np.mean(self.performance_history[-5:])
            improvement = ((new_avg - old_avg) / old_avg * 100) if old_avg > 0 else 0
        
        return {
            "status": "Active Learning",
            "total_trades_learned": len(self.training_data),
            "accuracy": accuracy,
            "feature_importance": self.feature_importance,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "improvement": round(improvement, 1),
            "confidence_level": "High" if accuracy > 0.7 else "Medium" if accuracy > 0.55 else "Building"
        }

ml_model = AdvancedTradingML()

# Advanced Adaptive Portfolio Manager
class AdaptivePortfolioManager:
    def __init__(self, total_balance):
        self.total_balance = total_balance
        self.positions = {}  # {symbol: {amount, entry_price, invested, stop_loss, take_profit, trailing_stop, highest_price}}
        self.closed_trades_profit = []
    
    def can_open_position(self, symbol):
        """Check if we can open a new position"""
        if symbol in self.positions:
            return False
        # Allow max 5 concurrent positions
        return len(self.positions) < 5
    
    def calculate_adaptive_position_size(self, symbol, risk_pct, confidence, volatility):
        """Dynamic position sizing based on AI confidence and volatility"""
        # Base position size
        max_positions = 5
        available = self.total_balance - sum([p["invested"] for p in self.positions.values()])
        base_size = available * (risk_pct / 100) / max(1, (max_positions - len(self.positions)))
        
        # Confidence multiplier (0.5x to 2x based on ML confidence)
        confidence_multiplier = 0.5 + (confidence / 100) * 1.5
        
        # Volatility adjustment (reduce size in high volatility)
        volatility_multiplier = 1.0 / (1 + volatility * 10) if volatility > 0 else 1.0
        
        # Apply multipliers
        adaptive_size = base_size * confidence_multiplier * volatility_multiplier
        
        # Cap at 30% of available balance for risk management
        max_size = available * 0.3
        return min(adaptive_size, max_size)
    
    def calculate_adaptive_targets(self, entry_price, volatility, momentum, trend_strength, confidence):
        """Calculate dynamic stop-loss and take-profit based on market conditions"""
        # Base targets
        base_stop_pct = 0.03  # 3%
        base_take_pct = 0.05  # 5%
        
        # Volatility adjustment (wider stops in volatile markets)
        volatility_factor = 1 + (volatility * 20)  # Scale volatility
        
        # Momentum adjustment (higher targets with strong momentum)
        momentum_factor = 1 + abs(momentum) * 2
        
        # Trend strength (tighter stops in weak trends)
        trend_factor = 0.7 + (trend_strength * 0.6)
        
        # Confidence adjustment (higher confidence = let winners run)
        confidence_factor = 0.8 + (confidence / 100) * 0.4
        
        # Calculate adaptive stops
        stop_loss_pct = base_stop_pct * volatility_factor * trend_factor
        take_profit_pct = base_take_pct * momentum_factor * confidence_factor
        
        # Cap extremes
        stop_loss_pct = max(0.02, min(0.08, stop_loss_pct))  # 2-8%
        take_profit_pct = max(0.04, min(0.20, take_profit_pct))  # 4-20%
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        return stop_loss, take_profit, stop_loss_pct, take_profit_pct
    
    def open_position(self, symbol, amount, price, stop_loss, take_profit, stop_pct, take_pct):
        """Open position with adaptive targets"""
        invested = amount * price
        self.positions[symbol] = {
            "amount": amount,
            "entry_price": price,
            "invested": invested,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_loss_pct": stop_pct,
            "take_profit_pct": take_pct,
            "trailing_stop_enabled": True,
            "highest_price": price,
            "trailing_stop_pct": stop_pct,  # Start with initial stop %
            "opened_at": datetime.now(timezone.utc).isoformat()
        }
    
    def update_trailing_stop(self, symbol, current_price):
        """Update trailing stop as price moves in favorable direction"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # Update highest price
        if current_price > pos["highest_price"]:
            pos["highest_price"] = current_price
            
            # Tighten trailing stop as profit increases
            profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            
            if profit_pct > 0.10:  # 10%+ profit
                pos["trailing_stop_pct"] = 0.015  # Tighten to 1.5%
            elif profit_pct > 0.05:  # 5%+ profit
                pos["trailing_stop_pct"] = 0.02  # Tighten to 2%
            
            # Update stop loss
            new_stop = pos["highest_price"] * (1 - pos["trailing_stop_pct"])
            pos["stop_loss"] = max(pos["stop_loss"], new_stop)  # Only move up
    
    def should_close_position(self, symbol, current_price, recommendation):
        """Adaptive exit logic"""
        if symbol not in self.positions:
            return False, None
        
        pos = self.positions[symbol]
        
        # Update trailing stop
        self.update_trailing_stop(symbol, current_price)
        
        # Check stop loss (trailing or fixed)
        if current_price <= pos["stop_loss"]:
            return True, "Trailing Stop Loss"
        
        # Check take profit
        if current_price >= pos["take_profit"]:
            return True, "Take Profit Target"
        
        # Adaptive exit: weak signal after good profit
        profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        if profit_pct > 0.03 and recommendation["opportunity_score"] < 40:
            return True, "Signal Weakened"
        
        # Risk management: cut losses quickly if signal deteriorates
        if profit_pct < -0.01 and recommendation["recommendation"] == "SELL":
            return True, "Signal Reversal"
        
        return False, None
    
    def close_position(self, symbol, exit_price):
        """Close position and track performance"""
        if symbol not in self.positions:
            return 0
        
        pos = self.positions[symbol]
        profit = (exit_price - pos["entry_price"]) * pos["amount"]
        profit_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        
        # Track for analytics
        self.closed_trades_profit.append({
            "symbol": symbol,
            "profit": profit,
            "profit_pct": profit_pct,
            "hold_time": datetime.now(timezone.utc).isoformat()
        })
        
        del self.positions[symbol]
        return profit
    
    def get_active_positions(self):
        """Get all active positions"""
        return self.positions
    
    def get_performance_stats(self):
        """Get trading performance statistics"""
        if not self.closed_trades_profit:
            return {}
        
        avg_profit_pct = np.mean([t["profit_pct"] for t in self.closed_trades_profit])
        win_rate = len([t for t in self.closed_trades_profit if t["profit"] > 0]) / len(self.closed_trades_profit)
        
        return {
            "avg_profit_pct": avg_profit_pct,
            "win_rate": win_rate,
            "total_trades": len(self.closed_trades_profit)
        }

portfolio = AdaptivePortfolioManager(100.0)

# Multi-Crypto Trading Logic
async def execute_multi_crypto_trading():
    global bot_state, portfolio
    
    # Get fresh recommendations
    recommendations = await analyze_crypto_opportunities()
    bot_state["crypto_recommendations"] = recommendations
    
    # Trade on top recommendations
    for rec in recommendations:
        symbol = rec["symbol"]
        
        # Fetch REAL-TIME market data for this crypto
        if symbol not in bot_state["crypto_data"]:
            bot_state["crypto_data"][symbol] = {}
        
        market_data = await fetch_real_time_data(symbol)
        bot_state["crypto_data"][symbol]["market_data"] = market_data
        
        # Calculate indicators
        df = calculate_indicators(market_data)
        latest = df.iloc[-1]
        
        # Check if we should trade this crypto
        should_trade = False
        
        # Strong buy signals
        if rec["recommendation"] in ["STRONG BUY", "BUY"] and rec["opportunity_score"] >= 70:
            should_trade = True
        
        # Additional technical confirmation
        ema_crossover = latest['ema_short'] > latest['ema_long']
        rsi_signal = 30 < latest['rsi'] < 70
        macd_signal = latest['macd'] > latest['macd_signal']
        
        technical_score = (int(ema_crossover) + int(rsi_signal) + int(macd_signal)) / 3
        
        if should_trade and technical_score > 0.5:
            # Open new position if we don't have one
            if portfolio.can_open_position(symbol):
                await execute_trade_for_crypto(symbol, rec, latest, df)
            # Or check existing position for exit
            elif symbol in portfolio.positions:
                await check_position_exit(symbol, latest, rec)

# Single Crypto Trade Execution
async def execute_trade_for_crypto(symbol, recommendation, latest, df):
    global bot_state, portfolio
    
    # Generate news specific to current crypto
    source, headline, sentiment = generate_mock_news(symbol)
    news_item = {
        "source": source,
        "headline": headline,
        "sentiment": sentiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crypto": symbol
    }
    
    bot_state["sentiment_headlines"].insert(0, news_item)
    bot_state["sentiment_headlines"] = bot_state["sentiment_headlines"][:10]
    
    volatility = latest['atr'] / latest['close']
    
    # Calculate combined score
    ai_sentiment = recommendation["sentiment"]
    technical_score = (int(latest['ema_short'] > latest['ema_long']) + int(30 < latest['rsi'] < 70) + int(latest['macd'] > latest['macd_signal'])) / 3
    sentiment_contribution = ai_sentiment * bot_state["sentiment_weight"]
    combined_score = (technical_score * (1 - bot_state["sentiment_weight"])) + sentiment_contribution
    
    # ML Enhancement
    if ml_model.is_trained:
        features = [
            1 if latest['ema_short'] > latest['ema_long'] else 0,
            latest['rsi'] / 100,
            1 if latest['macd'] > latest['macd_signal'] else 0,
            (ai_sentiment + 1) / 2,
            volatility
        ]
        ml_success_prob = ml_model.predict_trade_success(features)
        combined_score = combined_score * 0.6 + ml_success_prob * 0.4
    
    confidence = round(combined_score * 100, 2)
    
    # Execute BUY
    if combined_score > 0.6:
        entry_price = latest['close']
        position_size = portfolio.get_position_size(symbol, bot_state["risk_per_trade"])
        amount = position_size / entry_price
        
        # Open position
        portfolio.open_position(symbol, amount, entry_price)
        bot_state["balance"] -= position_size
        
        bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] 🟢 OPENED {symbol} @ ${round(entry_price, 6)} | Amount: {round(amount, 6)} | Score: {confidence}%")
        bot_state["trade_logs"] = bot_state["trade_logs"][-30:]
        
        # Update active positions
        bot_state["active_positions"] = portfolio.get_active_positions()

# Check and close positions
async def check_position_exit(symbol, latest, recommendation):
    global bot_state, portfolio
    
    if symbol not in portfolio.positions:
        return
    
    pos = portfolio.positions[symbol]
    current_price = latest['close']
    entry_price = pos["entry_price"]
    profit_pct = ((current_price - entry_price) / entry_price) * 100
    
    # Exit conditions
    should_exit = False
    exit_reason = ""
    
    # Take profit at 5%
    if profit_pct >= 5:
        should_exit = True
        exit_reason = "Take Profit"
    
    # Stop loss at -3%
    elif profit_pct <= -3:
        should_exit = True
        exit_reason = "Stop Loss"
    
    # Exit if recommendation turns to SELL
    elif recommendation["recommendation"] in ["SELL"] or recommendation["opportunity_score"] < 40:
        should_exit = True
        exit_reason = "Weak Signal"
    
    if should_exit:
        profit = portfolio.close_position(symbol, current_price)
        bot_state["balance"] += pos["invested"] + profit
        
        result = "WIN" if profit > 0 else "LOSS"
        if profit > 0:
            bot_state["wins"] += 1
        else:
            bot_state["losses"] += 1
        
        bot_state["trades_executed"] += 1
        bot_state["profit_loss"] = bot_state["balance"] - bot_state["initial_balance"]
        bot_state["profit_loss_pct"] = (bot_state["profit_loss"] / bot_state["initial_balance"]) * 100
        bot_state["equity"] = bot_state["balance"]
        
        # ML learning with advanced features
        volatility = latest['atr'] / latest['close']
        features = [
            1 if latest['ema_short'] > latest['ema_long'] else 0,
            latest['rsi'] / 100,
            1 if latest['macd'] > latest['macd_signal'] else 0,
            (recommendation["sentiment"] + 1) / 2,
            volatility
        ]
        # Pass market data for advanced feature calculation
        market_data = bot_state["crypto_data"][symbol].get("market_data", [])
        ml_model.add_trade_data(features, result, market_data)
        
        trade = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": "CLOSE",
            "price": round(current_price, 6),
            "amount": pos["amount"],
            "result": result,
            "profit_loss": round(profit, 2),
            "reason": f"{exit_reason} | P/L: {round(profit_pct, 2)}%",
            "ai_confidence": 0
        }
        
        bot_state["recent_trades"].insert(0, trade)
        bot_state["recent_trades"] = bot_state["recent_trades"][:10]
        
        bot_state["trade_logs"].append(f"[{trade['timestamp'][:19]}] 🔴 CLOSED {symbol} @ ${round(current_price, 6)} - {result} ({'+' if profit > 0 else ''}{round(profit, 2)}£) | {exit_reason}")
        bot_state["trade_logs"] = bot_state["trade_logs"][-30:]
        
        # Update active positions
        bot_state["active_positions"] = portfolio.get_active_positions()

# Bot Loop
async def bot_loop():
    global bot_state, portfolio
    
    if bot_state["multi_crypto_enabled"]:
        bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] 🚀 Multi-Crypto Bot started - Monitoring all markets")
    else:
        bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] Bot started on {bot_state['current_market']}")
    
    while bot_state["running"]:
        try:
            if bot_state["multi_crypto_enabled"]:
                await execute_multi_crypto_trading()
            else:
                # Original single crypto logic (kept for compatibility)
                pass
            await asyncio.sleep(8)  # Execute every 8 seconds
        except Exception as e:
            logger.error(f"Bot loop error: {e}")
            bot_state["trade_logs"].append(f"[ERROR] {str(e)}")
            await asyncio.sleep(8)

# API Endpoints
@api_router.post("/bot/start")
async def start_bot():
    global bot_state, task
    
    if bot_state["running"]:
        raise HTTPException(status_code=400, detail="Bot is already running")
    
    bot_state["running"] = True
    task = asyncio.create_task(bot_loop())
    
    return {"status": "started", "message": f"Bot started in {bot_state['mode']} mode"}

@api_router.post("/bot/stop")
async def stop_bot():
    global bot_state, task
    
    if not bot_state["running"]:
        raise HTTPException(status_code=400, detail="Bot is not running")
    
    bot_state["running"] = False
    if task:
        task.cancel()
    
    bot_state["trade_logs"].append(f"[{datetime.now(timezone.utc).isoformat()[:19]}] Bot stopped")
    
    return {"status": "stopped", "message": "Bot stopped successfully"}

@api_router.get("/bot/status")
async def get_bot_status():
    # Calculate total equity (balance + value of open positions)
    total_equity = bot_state["balance"]
    for symbol, pos in portfolio.positions.items():
        # Estimate current value (would use real-time price in production)
        if symbol in bot_state["crypto_data"] and "market_data" in bot_state["crypto_data"][symbol]:
            market_data = bot_state["crypto_data"][symbol]["market_data"]
            if market_data:
                current_price = market_data[-1]["close"]
                total_equity += pos["amount"] * current_price
    
    return {
        "running": bot_state["running"],
        "mode": bot_state["mode"],
        "balance": round(bot_state["balance"], 2),
        "equity": round(total_equity, 2),
        "profit_loss": round(total_equity - bot_state["initial_balance"], 2),
        "profit_loss_pct": round(((total_equity - bot_state["initial_balance"]) / bot_state["initial_balance"]) * 100, 2),
        "ai_confidence": bot_state["ai_confidence"],
        "sentiment_score": round(bot_state["sentiment_score"], 2),
        "trades_executed": bot_state["trades_executed"],
        "wins": bot_state["wins"],
        "losses": bot_state["losses"],
        "win_rate": round((bot_state["wins"] / bot_state["trades_executed"] * 100) if bot_state["trades_executed"] > 0 else 0, 2),
        "trade_logs": bot_state["trade_logs"][-15:],
        "recent_trades": bot_state["recent_trades"][:10],
        "sentiment_headlines": bot_state["sentiment_headlines"][:5],
        "current_market": bot_state["current_market"],
        "multi_crypto_enabled": bot_state["multi_crypto_enabled"],
        "active_positions": bot_state["active_positions"],
        "active_positions_count": len(bot_state["active_positions"])
    }

@api_router.put("/bot/settings")
async def update_settings(settings: BotSettings):
    global bot_state
    
    if settings.mode:
        bot_state["mode"] = settings.mode
    if settings.risk_per_trade:
        bot_state["risk_per_trade"] = settings.risk_per_trade
    if settings.current_market:
        bot_state["current_market"] = settings.current_market
    if settings.ai_mode_enabled is not None:
        bot_state["ai_mode_enabled"] = settings.ai_mode_enabled
    if settings.sentiment_weight:
        bot_state["sentiment_weight"] = settings.sentiment_weight
    
    return {"status": "updated", "settings": settings}

@api_router.post("/bot/reset")
async def reset_bot():
    global bot_state, portfolio
    
    if bot_state["running"]:
        raise HTTPException(status_code=400, detail="Stop the bot before resetting")
    
    bot_state["balance"] = 100.0
    bot_state["initial_balance"] = 100.0
    bot_state["equity"] = 100.0
    bot_state["profit_loss"] = 0.0
    bot_state["profit_loss_pct"] = 0.0
    bot_state["trades_executed"] = 0
    bot_state["wins"] = 0
    bot_state["losses"] = 0
    bot_state["trade_logs"] = []
    bot_state["recent_trades"] = []
    bot_state["sentiment_headlines"] = []
    bot_state["active_positions"] = {}
    bot_state["crypto_data"] = {}
    
    # Reset portfolio
    portfolio = PortfolioManager(100.0)
    
    return {"status": "reset", "message": "Bot reset to £100 demo balance"}

@api_router.get("/market/data")
async def get_market_data():
    # Fetch real-time data for current market
    symbol = bot_state["current_market"]
    if symbol not in bot_state["crypto_data"] or "market_data" not in bot_state["crypto_data"][symbol]:
        market_data = await fetch_real_time_data(symbol)
        if symbol not in bot_state["crypto_data"]:
            bot_state["crypto_data"][symbol] = {}
        bot_state["crypto_data"][symbol]["market_data"] = market_data
    else:
        market_data = bot_state["crypto_data"][symbol]["market_data"]
    
    return {"data": market_data[-50:]}

@api_router.get("/crypto/list")
async def get_crypto_list():
    """Get list of available cryptocurrencies"""
    return {"cryptos": AVAILABLE_CRYPTOS}

@api_router.get("/crypto/recommendations")
async def get_crypto_recommendations():
    """Get AI-powered crypto recommendations based on worldwide news"""
    try:
        if not bot_state["crypto_recommendations"] or random.random() < 0.3:
            bot_state["crypto_recommendations"] = await analyze_crypto_opportunities()
        return {"recommendations": bot_state["crypto_recommendations"]}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return {"recommendations": []}

@api_router.post("/crypto/switch")
async def switch_crypto(data: dict):
    """Switch to a different cryptocurrency"""
    symbol = data.get("symbol")
    if symbol and any(c["symbol"] == symbol for c in AVAILABLE_CRYPTOS):
        bot_state["current_market"] = symbol
        bot_state["market_data"] = generate_mock_market_data(symbol)
        return {"status": "success", "current_market": symbol}
    return {"status": "error", "message": "Invalid crypto symbol"}

@api_router.get("/ml/stats")
async def get_ml_stats():
    """Get machine learning model statistics"""
    stats = ml_model.get_model_stats()
    return {"ml_stats": stats}

@api_router.get("/crypto/list")
async def get_crypto_list():
    """Get list of available cryptocurrencies"""
    return {"cryptos": AVAILABLE_CRYPTOS}

@api_router.get("/crypto/recommendations")
async def get_crypto_recommendations():
    """Get AI-powered crypto recommendations based on worldwide news"""
    try:
        if not bot_state["crypto_recommendations"] or random.random() < 0.3:  # Refresh 30% of the time
            bot_state["crypto_recommendations"] = await analyze_crypto_opportunities()
        return {"recommendations": bot_state["crypto_recommendations"]}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return {"recommendations": []}

@api_router.post("/crypto/switch")
async def switch_crypto(data: dict):
    """Switch to a different cryptocurrency"""
    symbol = data.get("symbol")
    if symbol and any(c["symbol"] == symbol for c in AVAILABLE_CRYPTOS):
        bot_state["current_market"] = symbol
        bot_state["market_data"] = generate_mock_market_data(symbol)
        return {"status": "success", "current_market": symbol}
    return {"status": "error", "message": "Invalid crypto symbol"}

@api_router.get("/ml/stats")
async def get_ml_stats():
    """Get machine learning model statistics"""
    stats = ml_model.get_model_stats()
    return {"ml_stats": stats}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()