"""
Coinbase Integration Module
Prepared for live trading with Coinbase API
"""
import os
import hmac
import hashlib
import time
import json
from typing import Dict, Optional
import aiohttp

class CoinbaseTrader:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get('COINBASE_API_KEY')
        self.api_secret = api_secret or os.environ.get('COINBASE_API_SECRET')
        self.base_url = "https://api.coinbase.com/v2"
        self.is_demo = not (self.api_key and self.api_secret)
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = ""):
        """Generate HMAC signature for Coinbase API"""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def get_account_balance(self) -> Dict:
        """Get Coinbase account balances"""
        if self.is_demo:
            return {"USD": 100.0, "demo": True}
        
        try:
            timestamp = str(int(time.time()))
            path = "/accounts"
            signature = self._generate_signature(timestamp, "GET", path)
            
            headers = {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-VERSION": "2021-01-01"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}{path}", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return {"error": f"Status {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """
        Place order on Coinbase
        
        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "buy" or "sell"
            amount: Amount to trade
            price: Limit price (None for market order)
        """
        if self.is_demo:
            return {
                "order_id": f"demo_{int(time.time())}",
                "status": "filled",
                "demo": True
            }
        
        try:
            timestamp = str(int(time.time()))
            path = "/orders"
            
            order_data = {
                "product_id": symbol,
                "side": side,
                "type": "market" if price is None else "limit",
                "size": str(amount)
            }
            
            if price:
                order_data["price"] = str(price)
            
            body = json.dumps(order_data)
            signature = self._generate_signature(timestamp, "POST", path, body)
            
            headers = {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-VERSION": "2021-01-01",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}{path}",
                    headers=headers,
                    data=body
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return {"error": f"Status {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            # Convert symbol format: BTC/USD -> BTC-USD
            coinbase_symbol = symbol.replace("/", "-")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/prices/{coinbase_symbol}/spot"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data["data"]["amount"])
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Check order status"""
        if self.is_demo:
            return {"status": "filled", "demo": True}
        
        try:
            timestamp = str(int(time.time()))
            path = f"/orders/{order_id}"
            signature = self._generate_signature(timestamp, "GET", path)
            
            headers = {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-VERSION": "2021-01-01"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}{path}", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return {"error": f"Status {response.status}"}
        except Exception as e:
            return {"error": str(e)}

# Initialize Coinbase trader (will be in demo mode until API keys provided)
coinbase_trader = CoinbaseTrader()
