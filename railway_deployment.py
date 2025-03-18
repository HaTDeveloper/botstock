"""
Simplified Saudi Stock Bot for Railway Deployment

This is a lightweight version of the Saudi Stock Bot that removes heavy dependencies
to ensure it works properly on Railway's free tier.
"""
import os
import json
import time
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#################################################
# DATA API MODULE IMPLEMENTATION
#################################################

class ApiClient:
    """Simplified API client for accessing Yahoo Finance data."""
    
    def __init__(self):
        """Initialize the API client with default configuration."""
        self.base_url = "https://query1.finance.yahoo.com/v8/finance"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        logger.info("ApiClient initialized")
    
    def get_stock_data(self, symbol, interval='1d', range='1mo'):
        """Get basic stock data for a symbol."""
        # Ensure Saudi stock exchange suffix if not present
        if not symbol.endswith('.SR') and not symbol.startswith('^'):
            symbol = f"{symbol}.SR"
            
        # Prepare the endpoint URL
        endpoint = f"/chart/{symbol}"
        url = f"{self.base_url}{endpoint}"
        
        # Prepare query parameters
        params = {
            'interval': interval,
            'range': range,
            'includeAdjustedClose': True
        }
        
        # Make the API request
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stock data: {e}")
            return {"error": str(e)}

#################################################
# SAUDI STOCK BOT IMPLEMENTATION
#################################################

class SaudiStockBot:
    """Simplified bot for Saudi stock market data."""
    
    def __init__(self, webhook_url=None):
        """Initialize the Saudi Stock Bot."""
        self.api_client = ApiClient()
        self.webhook_url = webhook_url
        self.saudi_market_index = "^TASI"  # Tadawul All Share Index
        logger.info("Saudi Stock Bot initialized")
        if not webhook_url:
            logger.warning("Discord webhook URL not set. Skipping webhook notification.")
        
    def get_stock_data(self, symbol, interval='1d', range='1mo'):
        """Get stock data for a Saudi stock."""
        return self.api_client.get_stock_data(symbol, interval, range)
    
    def get_market_overview(self):
        """Get a simplified overview of the Saudi stock market."""
        # Get Tadawul All Share Index data
        tasi_data = self.get_stock_data(self.saudi_market_index, range='1mo')
        
        # Get data for major Saudi stocks (limited to 2 for simplicity)
        major_stocks = ['2222.SR', '1010.SR']
        stocks_data = {}
        
        for stock in major_stocks:
            try:
                stocks_data[stock] = self.get_stock_data(stock, range='5d')
            except Exception as e:
                logger.error(f"Error fetching data for {stock}: {e}")
                stocks_data[stock] = {"error": str(e)}
        
        # Compile market overview
        return {
            "market_index": tasi_data,
            "major_stocks": stocks_data,
            "timestamp": int(time.time())
        }

#################################################
# RAILWAY DEPLOYMENT WEB SERVER
#################################################

# Initialize Flask app
app = Flask(__name__)

# Get webhook URL from environment variable
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# Initialize Saudi Stock Bot
bot = SaudiStockBot(webhook_url=WEBHOOK_URL)

@app.route('/')
def home():
    """Home endpoint to verify the service is running."""
    return jsonify({
        "status": "online",
        "service": "Saudi Stock Bot (Simplified)",
        "version": "1.0.0",
        "timestamp": int(time.time())
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": int(time.time())
    })

@app.route('/api/market/overview', methods=['GET'])
def market_overview():
    """Get Saudi market overview."""
    try:
        data = bot.get_market_overview()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def stock_data(symbol):
    """Get data for a specific stock."""
    try:
        interval = request.args.get('interval', '1d')
        range_param = request.args.get('range', '1mo')
        data = bot.get_stock_data(symbol, interval, range_param)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting stock data for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port)
