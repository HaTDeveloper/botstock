"""
Combined Saudi Stock Bot with Railway Deployment

This is a single-file implementation that combines:
1. Data API functionality
2. Saudi Stock Bot implementation
3. Railway deployment web server

This approach eliminates the need for separate module imports.
"""
import os
import json
import time
import logging
import requests
import schedule
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """
    Client for accessing Yahoo Finance API data for stock analysis.
    
    This class provides methods to fetch stock data including:
    - Stock charts and historical price data
    - Stock holder information
    - Stock insights and technical analysis
    """
    
    def __init__(self):
        """Initialize the API client with default configuration."""
        self.base_url = "https://query1.finance.yahoo.com/v8/finance"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)  AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        logger.info("ApiClient initialized")
    
    def call_api(self, api_name, query=None):
        """
        Call the specified API with the given query parameters.
        
        Args:
            api_name (str): The name of the API to call (e.g., 'YahooFinance/get_stock_chart')
            query (dict, optional): Query parameters for the API call
            
        Returns:
            dict: The API response data
        """
        if query is None:
            query = {}
            
        # Parse the API name to determine the endpoint
        if api_name == "YahooFinance/get_stock_chart":
            return self._get_stock_chart(query)
        elif api_name == "YahooFinance/get_stock_holders":
            return self._get_stock_holders(query)
        elif api_name == "YahooFinance/get_stock_insights":
            return self._get_stock_insights(query)
        else:
            logger.error(f"Unknown API: {api_name}")
            raise ValueError(f"Unknown API: {api_name}")
    
    def _get_stock_chart(self, query):
        """
        Get stock chart data from Yahoo Finance.
        
        Args:
            query (dict): Query parameters including:
                - symbol: The stock symbol (required)
                - interval: Data interval (e.g., '1d', '1wk', '1mo')
                - range: Data range (e.g., '1d', '1mo', '1y')
                
        Returns:
            dict: Stock chart data including price history
        """
        # Validate required parameters
        if 'symbol' not in query:
            raise ValueError("Symbol is required for stock chart data")
        if 'interval' not in query:
            query['interval'] = '1d'  # Default interval
        if 'range' not in query and 'period1' not in query and 'period2' not in query:
            query['range'] = '1mo'  # Default range
            
        # Prepare the endpoint URL
        endpoint = f"/chart/{query.pop('symbol')}"
        url = f"{self.base_url}{endpoint}"
        
        # Make the API request
        try:
            response = requests.get(url, headers=self.headers, params=query)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stock chart: {e}")
            return {"error": str(e)}
    
    def _get_stock_holders(self, query):
        """
        Get stock holder information from Yahoo Finance.
        
        Args:
            query (dict): Query parameters including:
                - symbol: The stock symbol (required)
                - region: Region code (optional)
                
        Returns:
            dict: Stock holder information
        """
        # Validate required parameters
        if 'symbol' not in query:
            raise ValueError("Symbol is required for stock holder data")
            
        # Prepare the endpoint URL
        symbol = query.pop('symbol')
        endpoint = f"/quoteSummary/{symbol}"
        url = f"{self.base_url}{endpoint}"
        
        # Add modules parameter for holder information
        query['modules'] = 'insiderHolders'
        
        # Make the API request
        try:
            response = requests.get(url, headers=self.headers, params=query)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stock holders: {e}")
            return {"error": str(e)}
    
    def _get_stock_insights(self, query):
        """
        Get stock insights and technical analysis from Yahoo Finance.
        
        Args:
            query (dict): Query parameters including:
                - symbol: The stock symbol (required)
                
        Returns:
            dict: Stock insights and technical analysis
        """
        # Validate required parameters
        if 'symbol' not in query:
            raise ValueError("Symbol is required for stock insights")
            
        # Prepare the endpoint URL
        symbol = query.pop('symbol')
        endpoint = f"/finance/insights"
        url = f"https://query2.finance.yahoo.com/v1{endpoint}"
        
        # Add symbol parameter
        query['symbol'] = symbol
        
        # Make the API request
        try:
            response = requests.get(url, headers=self.headers, params=query) 
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stock insights: {e}")
            return {"error": str(e)}
    
    def get_saudi_stock_data(self, symbol, interval='1d', range='1mo'):
        """
        Convenience method to get Saudi stock data with proper formatting.
        
        Args:
            symbol (str): Saudi stock symbol (e.g., '2222.SR' for SABIC)
            interval (str): Data interval (default: '1d')
            range (str): Data range (default: '1mo')
            
        Returns:
            dict: Formatted stock data for Saudi market
        """
        # Ensure Saudi stock exchange suffix if not present
        if not symbol.endswith('.SR'):
            symbol = f"{symbol}.SR"
            
        # Get the stock chart data
        chart_data = self.call_api('YahooFinance/get_stock_chart', {
            'symbol': symbol,
            'interval': interval,
            'range': range,
            'includeAdjustedClose': True
        })
        
        # Get stock insights if available
        try:
            insights_data = self.call_api('YahooFinance/get_stock_insights', {
                'symbol': symbol
            })
        except Exception as e:
            logger.warning(f"Could not fetch insights for {symbol}: {e}")
            insights_data = {"error": str(e)}
        
        # Combine and return the data
        return {
            "chart": chart_data,
            "insights": insights_data,
            "timestamp": int(time.time())
        }

#################################################
# SAUDI STOCK BOT IMPLEMENTATION
#################################################

class SaudiStockBot:
    """
    Bot for analyzing Saudi stock market data and providing recommendations.
    
    This class provides functionality to:
    - Retrieve stock data from Yahoo Finance
    - Calculate technical indicators
    - Generate trading signals
    - Send notifications via webhooks
    """
    
    def __init__(self, webhook_url=None):
        """
        Initialize the Saudi Stock Bot.
        
        Args:
            webhook_url (str, optional): URL for webhook notifications
        """
        self.api_client = ApiClient()
        self.webhook_url = webhook_url
        self.saudi_market_index = "^TASI"  # Tadawul All Share Index
        self.cache = {}
        self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        logger.info("Saudi Stock Bot initialized")
        
    def get_stock_data(self, symbol, interval='1d', range='1mo', force_refresh=False):
        """
        Get stock data for a Saudi stock.
        
        Args:
            symbol (str): Stock symbol (e.g., '2222' for SABIC)
            interval (str): Data interval (default: '1d')
            range (str): Data range (default: '1mo')
            force_refresh (bool): Force refresh cache
            
        Returns:
            dict: Stock data including price history and insights
        """
        # Format symbol for Saudi market if needed
        if not symbol.endswith('.SR') and not symbol.startswith('^'):
            symbol = f"{symbol}.SR"
            
        # Check cache if not forcing refresh
        cache_key = f"{symbol}_{interval}_{range}"
        current_time = time.time()
        
        if not force_refresh and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_expiry:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
        
        # Get fresh data from API
        logger.info(f"Fetching fresh data for {symbol}")
        data = self.api_client.get_saudi_stock_data(symbol, interval, range)
        
        # Update cache
        self.cache[cache_key] = (data, current_time)
        
        return data
    
    def get_market_overview(self):
        """
        Get an overview of the Saudi stock market.
        
        Returns:
            dict: Market overview data
        """
        # Get Tadawul All Share Index data
        tasi_data = self.get_stock_data(self.saudi_market_index, range='1mo')
        
        # Get data for major Saudi stocks
        major_stocks = ['2222.SR', '1010.SR', '2350.SR', '1150.SR', '2001.SR']
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
    
    def calculate_technical_indicators(self, symbol, range='3mo'):
        """
        Calculate technical indicators for a stock.
        
        Args:
            symbol (str): Stock symbol
            range (str): Data range for calculations
            
        Returns:
            dict: Technical indicators
        """
        # Get stock data
        data = self.get_stock_data(symbol, range=range)
        
        # Extract price data
        try:
            chart_result = data['chart']['chart']['result'][0]
            timestamps = chart_result['timestamp']
            quotes = chart_result['indicators']['quote'][0]
            
            # Convert to pandas DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('date', inplace=True)
            
            # Calculate indicators
            # Moving Averages
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['MA200'] = df['close'].rolling(window=200).mean()
            
            # Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['MA20_std'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['MA20'] + (df['MA20_std'] * 2)
            df['lower_band'] = df['MA20'] - (df['MA20_std'] * 2)
            
            # Generate signals
            df['MA_signal'] = np.where(df['MA20'] > df['MA50'], 1, -1)
            df['RSI_signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
            df['MACD_crossover'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
            
            # Compile results
            latest = df.iloc[-1].to_dict()
            
            return {
                'symbol': symbol,
                'last_price': latest['close'],
                'indicators': {
                    'moving_averages': {
                        'MA20': latest['MA20'],
                        'MA50': latest['MA50'],
                        'MA200': latest['MA200'],
                        'signal': latest['MA_signal']
                    },
                    'rsi': {
                        'value': latest['RSI'],
                        'signal': latest['RSI_signal']
                    },
                    'macd': {
                        'value': latest['MACD'],
                        'signal_line': latest['MACD_signal'],
                        'histogram': latest['MACD'] - latest['MACD_signal'],
                        'signal': latest['MACD_crossover']
                    },
                    'bollinger_bands': {
                        'upper': latest['upper_band'],
                        'middle': latest['MA20'],
                        'lower': latest['lower_band']
                    }
                },
                'dataframe': df.tail(30).to_dict(orient='records'),
                'timestamp': int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def generate_recommendation(self, symbol):
        """
        Generate trading recommendation for a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Trading recommendation
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(symbol)
        
        if 'error' in indicators:
            return {"error": indicators['error'], "symbol": symbol}
        
        # Get stock insights
        try:
            data = self.get_stock_data(symbol)
            insights = data.get('insights', {})
        except Exception as e:
            logger.error(f"Error fetching insights for {symbol}: {e}")
            insights = {"error": str(e)}
        
        # Determine recommendation based on technical indicators
        ma_signal = indicators['indicators']['moving_averages']['signal']
        rsi_signal = indicators['indicators']['rsi']['signal']
        macd_signal = indicators['indicators']['macd']['signal']
        
        # Simple scoring system
        score = ma_signal + rsi_signal + macd_signal
        
        if score >= 2:
            recommendation = "Strong Buy"
        elif score > 0:
            recommendation = "Buy"
        elif score == 0:
            recommendation = "Hold"
        elif score > -2:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
        
        # Generate recommendation text
        rsi_value = indicators['indicators']['rsi']['value']
        last_price = indicators['last_price']
        
        if recommendation in ["Strong Buy", "Buy"]:
            reason = f"Technical indicators are bullish. RSI at {rsi_value:.2f} suggests the stock is not overbought."
            if ma_signal > 0:
                reason += " Short-term moving average is above long-term moving average, indicating upward momentum."
            if macd_signal > 0:
                reason += " MACD is above signal line, confirming bullish trend."
        elif recommendation == "Hold":
            reason = f"Mixed technical signals. RSI at {rsi_value:.2f} is in neutral territory."
            reason += " Consider waiting for clearer signals before taking a position."
        else:
            reason = f"Technical indicators are bearish. RSI at {rsi_value:.2f} suggests caution."
            if ma_signal < 0:
                reason += " Short-term moving average is below long-term moving average, indicating downward momentum."
            if macd_signal < 0:
                reason += " MACD is below signal line, confirming bearish trend."
        
        return {
            "symbol": symbol,
            "last_price": last_price,
            "recommendation": recommendation,
            "reason": reason,
            "score": score,
            "technical_indicators": indicators['indicators'],
            "insights": insights,
            "timestamp": int(time.time())
        }
    
    def send_webhook_notification(self, data):
        """
        Send notification via webhook.
        
        Args:
            data (dict): Data to send
            
        Returns:
            bool: Success status
        """
        if not self.webhook_url:
            logger.warning("No webhook URL configured")
            return False
        
        try:
            response = requests.post(
                self.webhook_url,
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"Webhook notification sent successfully: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    def generate_daily_report(self, symbols=None):
        """
        Generate a daily report for specified Saudi stocks.
        
        Args:
            symbols (list, optional): List of stock symbols to include
            
        Returns:
            dict: Daily report data
        """
        if symbols is None:
            # Default to major Saudi stocks
            symbols = ['2222', '1010', '2350', '1150', '2001']
        
        # Get market overview
        market_overview = self.get_market_overview()
        
        # Generate recommendations for each stock
        recommendations = {}
        for symbol in symbols:
            try:
                recommendations[symbol] = self.generate_recommendation(symbol)
            except Exception as e:
                logger.error(f"Error generating recommendation for {symbol}: {e}")
                recommendations[symbol] = {"error": str(e), "symbol": symbol}
        
        # Compile report
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market_overview": market_overview,
            "recommendations": recommendations,
            "timestamp": int(time.time())
        }
        
        # Send webhook notification if configured
        if self.webhook_url:
            self.send_webhook_notification({
                "type": "daily_report",
                "content": report
            })
        
        return report

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
        "service": "Saudi Stock Bot",
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

@app.route('/api/stock/<symbol>/indicators', methods=['GET'])
def technical_indicators(symbol):
    """Get technical indicators for a specific stock."""
    try:
        range_param = request.args.get('range', '3mo')
        data = bot.calculate_technical_indicators(symbol, range_param)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/recommendation', methods=['GET'])
def recommendation(symbol):
    """Get trading recommendation for a specific stock."""
    try:
        data = bot.generate_recommendation(symbol)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error generating recommendation for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/report/daily', methods=['GET'])
def daily_report():
    """Generate daily report for Saudi stocks."""
    try:
        symbols = request.args.get('symbols')
        if symbols:
            symbols = symbols.split(',')
        data = bot.generate_daily_report(symbols)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    """Handle incoming webhook requests."""
    try:
        data = request.json
        logger.info(f"Received webhook: {data}")
        
        # Process webhook data based on type
        if 'type' in data:
            if data['type'] == 'generate_report':
                symbols = data.get('symbols')
                report = bot.generate_daily_report(symbols)
                return jsonify({"status": "success", "report": report})
            elif data['type'] == 'analyze_stock':
                symbol = data.get('symbol')
                if not symbol:
                    return jsonify({"error": "Symbol is required"}), 400
                recommendation = bot.generate_recommendation(symbol)
                return jsonify({"status": "success", "recommendation": recommendation})
        
        return jsonify({"status": "success", "message": "Webhook received"})
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500

def run_scheduled_tasks():
    """Run scheduled tasks in a separate thread."""
    def run_scheduler():
        # Schedule daily report generation (after market close)
        schedule.every().day.at("16:30").do(lambda: bot.generate_daily_report())
        
        # Schedule market overview update (every hour during trading hours)
        def update_market_during_trading():
            now = datetime.now()
            # Saudi market trading hours: Sunday-Thursday, 10:00-15:00
            if now.weekday() not in [4, 5]:  # Not Friday or Saturday
                if 10 <= now.hour < 15:
                    bot.get_market_overview()
        
        schedule.every(60).minutes.do(update_market_during_trading)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("Scheduled tasks started")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Start scheduled tasks
    run_scheduled_tasks()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port)
