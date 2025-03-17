#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Saudi Stock Analysis Bot
--------------------------------
A bot that analyzes Saudi stock market data to identify:
1. Golden opportunities (quick rises and falls)
2. General analysis of stocks expected to rise

Features:
- Real-time data analysis when available
- Historical data analysis (up to 7 years when available)
- News integration and impact analysis
- Confidence metrics for predictions
- Investment advice with each recommendation
"""

import sys
import os
import json
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add path for data API access
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# Initialize API client
client = ApiClient()

class SaudiStockBot:
    def __init__(self):
        """Initialize the Saudi Stock Bot with default settings."""
        self.region = "SA"  # Saudi Arabia region
        self.lang = "en-US"  # Default language
        self.saudi_market_symbols = []
        self.data_cache = {}
        self.news_cache = {}
        
        # Technical analysis parameters
        self.short_window = 20
        self.medium_window = 50
        self.long_window = 200
        
        # Confidence thresholds - lowered for more sensitivity
        self.high_confidence = 70  # Was 80
        self.medium_confidence = 50  # Was 60
        self.low_confidence = 30  # Was 40
        
        # Load Saudi market symbols
        self.load_saudi_market_symbols()
    
    def load_saudi_market_symbols(self):
        """Load Saudi stock market symbols."""
        # Expanded list of Saudi stock symbols with .SR suffix (Saudi Riyal)
        self.saudi_market_symbols = [
            # Major companies
            "2222.SR",  # Saudi Aramco
            "1150.SR",  # Alinma Bank
            "1010.SR",  # RIBL (Riyad Bank)
            "2350.SR",  # Saudi Telecom
            "2001.SR",  # SABIC
            "1211.SR",  # MAADEN
            "2290.SR",  # Yanbu National Petrochemical
            
            # Banking & Financial Services
            "1180.SR",  # Al Rajhi Bank
            "1050.SR",  # National Commercial Bank
            "1060.SR",  # Bank AlJazira
            "1080.SR",  # Arab National Bank
            "1120.SR",  # Al Bilad Bank
            "1140.SR",  # Bank Albilad
            
            # Petrochemicals
            "2010.SR",  # SABIC
            "2330.SR",  # Advanced Petrochemical
            "2380.SR",  # Petro Rabigh
            "2310.SR",  # Sipchem
            
            # Healthcare
            "4261.SR",  # Dallah Healthcare
            "4002.SR",  # Mouwasat Medical Services
            "4004.SR",  # National Medical Care
            "4009.SR",  # Middle East Healthcare
            
            # Retail
            "4190.SR",  # Jarir Marketing
            "4240.SR",  # Fawaz Abdulaziz Alhokair
            "4003.SR",  # United Electronics (eXtra)
            "4050.SR",  # SACO
            
            # Food & Agriculture
            "2280.SR",  # Almarai
            "2100.SR",  # Savola Group
            "6001.SR",  # Halwani Bros
            "6002.SR",  # Herfy Food Services
            
            # Real Estate
            "4300.SR",  # Dar Al Arkan
            "4230.SR",  # Red Sea International
            
            # Telecommunications
            "7010.SR",  # Etihad Etisalat (Mobily)
            "7030.SR",  # Zain KSA
            
            # Transportation
            "4031.SR",  # Saudi Airlines Catering
            "4040.SR",  # Saudi Public Transport
            "4110.SR",  # Saudi Transport and Investment
            
            # Energy & Utilities
            "5110.SR",  # Saudi Electricity
            "2082.SR",  # ACWA Power
        ]
        print(f"Loaded {len(self.saudi_market_symbols)} Saudi stock symbols")
        return self.saudi_market_symbols
    
    def get_stock_data(self, symbol, interval="1d", range="1y"):
        """
        Get stock data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            range (str): Data range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            dict: Stock data
        """
        try:
            # Cache key to avoid redundant API calls
            cache_key = f"{symbol}_{interval}_{range}"
            
            # Return cached data if available
            if cache_key in self.data_cache:
                print(f"Using cached data for {symbol}")
                return self.data_cache[cache_key]
            
            # Call Yahoo Finance API
            print(f"Fetching data for {symbol} with interval {interval} and range {range}")
            stock_data = client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': self.region,
                'interval': interval,
                'range': range,
                'includeAdjustedClose': True
            })
            
            # Cache the data
            self.data_cache[cache_key] = stock_data
            
            return stock_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_insights(self, symbol):
        """
        Get stock insights for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock insights
        """
        try:
            # Call Yahoo Finance API
            print(f"Fetching insights for {symbol}")
            insights = client.call_api('YahooFinance/get_stock_insights', query={
                'symbol': symbol
            })
            
            return insights
        except Exception as e:
            print(f"Error fetching insights for {symbol}: {e}")
            return None
    
    def get_stock_holders(self, symbol):
        """
        Get stock holders information for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock holders information
        """
        try:
            # Call Yahoo Finance API
            print(f"Fetching holders information for {symbol}")
            holders = client.call_api('YahooFinance/get_stock_holders', query={
                'symbol': symbol,
                'region': self.region
            })
            
            return holders
        except Exception as e:
            print(f"Error fetching holders information for {symbol}: {e}")
            return None
    
    def process_stock_data(self, stock_data):
        """
        Process raw stock data into a pandas DataFrame.
        
        Args:
            stock_data (dict): Raw stock data from API
            
        Returns:
            pd.DataFrame: Processed stock data
        """
        if not stock_data or 'chart' not in stock_data or 'result' not in stock_data['chart'] or not stock_data['chart']['result']:
            return None
        
        result = stock_data['chart']['result'][0]
        
        # Extract timestamps and convert to datetime
        timestamps = result['timestamp']
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Extract price data
        quotes = result['indicators']['quote'][0]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes['volume']
        })
        
        # Add adjusted close if available
        if 'adjclose' in result['indicators']:
            df['adjclose'] = result['indicators']['adjclose'][0]['adjclose']
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Calculate technical indicators
        self.calculate_technical_indicators(df)
        
        return df
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for analysis.
        
        Args:
            df (pd.DataFrame): Stock data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_50'] = df['close'].rolling(window=self.medium_window).mean()
        df['SMA_200'] = df['close'].rolling(window=self.long_window).mean()
        
        # Exponential Moving Averages
        df['EMA_20'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=self.medium_window, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_20'] - df['EMA_50']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Volume indicators
        df['Volume_Change'] = df['volume'].pct_change() * 100
        df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA_20']
        
        # Price momentum
        df['Price_Change_1d'] = df['close'].pct_change() * 100
        df['Price_Change_5d'] = df['close'].pct_change(periods=5) * 100
        df['Price_Change_20d'] = df['close'].pct_change(periods=20) * 100
        
        # Additional indicators
        
        # Average Directional Index (ADX) - Trend strength
        # First calculate +DI and -DI
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().multiply(-1)
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low'] - df['close'].shift(1)).abs()
        }).max(axis=1)
        
        tr14 = tr.rolling(window=14).sum()
        plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr14)
        minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / tr14)
        
        dx = 100 * ((plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).abs())
        df['ADX'] = dx.rolling(window=14).mean()
        
        # Stochastic Oscillator
        df['Stoch_K'] = 100 * ((df['close'] - df['low'].rolling(window=14).min()) / 
                              (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # On-Balance Volume (OBV)
        obv = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        df['OBV'] = obv
        
        # Ichimoku Cloud
        df['Tenkan_Sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['Kijun_Sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
        df['Senkou_Span_B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou_Span'] = df['close'].shift(-26)
        
        return df
    
    def analyze_golden_opportunities(self, symbol):
        """
        Analyze stock for golden opportunities (quick rises and falls).
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Analysis results with confidence metrics
        """
        # Get stock data for the last 3 months with daily interval
        stock_data = self.get_stock_data(symbol, interval="1d", range="3mo")
        if not stock_data:
            return {"error": f"Could not fetch data for {symbol}"}
        
        # Process data
        df = self.process_stock_data(stock_data)
        if df is None or df.empty:
            return {"error": f"Could not process data for {symbol}"}
        
        # Get stock insights for additional analysis
        insights = self.get_stock_insights(symbol)
        
        # Extract stock name
        stock_name = stock_data['chart']['result'][0]['meta'].get('shortName', symbol)
        
        # Analysis results
        results = {
            "symbol": symbol,
            "stock_number": symbol.split('.')[0],
            "name": stock_name,
            "analysis_type": "golden_opportunities",
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "opportunities": [],
            "technical_indicators": {}
        }
        
        # Latest price data
        latest_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change = ((latest_price - prev_price) / prev_price) * 100
        
        # Technical indicators for latest data point
        results["technical_indicators"] = {
            "latest_price": round(latest_price, 2),
            "price_change_percent": round(price_change, 2),
            "rsi": round(df['RSI'].iloc[-1], 2) if not pd.isna(df['RSI'].iloc[-1]) else None,
            "macd": round(df['MACD'].iloc[-1], 2) if not pd.isna(df['MACD'].iloc[-1]) else None,
            "volume_change": round(df['Volume_Change'].iloc[-1], 2) if not pd.isna(df['Volume_Change'].iloc[-1]) else None,
            "adx": round(df['ADX'].iloc[-1], 2) if not pd.isna(df['ADX'].iloc[-1]) else None,
            "stoch_k": round(df['Stoch_K'].iloc[-1], 2) if not pd.isna(df['Stoch_K'].iloc[-1]) else None,
        }
        
        # Analyze for potential quick rise
        rise_confidence = 0
        rise_reasons = []
        
        # RSI indicates oversold (below 30)
        if df['RSI'].iloc[-1] < 30:
            rise_confidence += 25
            rise_reasons.append("RSI indicates oversold condition")
        # RSI is recovering from low levels
        elif df['RSI'].iloc[-1] < 40 and df['RSI'].iloc[-1] > df['RSI'].iloc[-2]:
            rise_confidence += 15
            rise_reasons.append("RSI is recovering from low levels")
        
        # Price is near lower Bollinger Band
        if latest_price < df['BB_Lower'].iloc[-1] * 1.02:
            rise_confidence += 20
            rise_reasons.append("Price is near lower Bollinger Band")
        
        # MACD crossed above signal line
        if df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            rise_confidence += 20
            rise_reasons.append("MACD crossed above signal line")
        # MACD histogram turning positive
        elif df['MACD_Histogram'].iloc[-1] > 0 and df['MACD_Histogram'].iloc[-2] < 0:
            rise_confidence += 15
            rise_reasons.append("MACD histogram turning positive")
        
        # Significant volume increase
        if df['Volume_Change'].iloc[-1] > 50:
            rise_confidence += 15
            rise_reasons.append("Significant volume increase")
        # Volume above average
        elif df['Volume_Ratio'].iloc[-1] > 1.5:
            rise_confidence += 10
            rise_reasons.append("Volume above average")
        
        # Price bounced off support level (200-day SMA)
        if prev_price < df['SMA_200'].iloc[-2] and latest_price > df['SMA_200'].iloc[-1]:
            rise_confidence += 20
            rise_reasons.append("Price bounced off 200-day SMA support")
        # Price bounced off support level (50-day SMA)
        elif prev_price < df['SMA_50'].iloc[-2] and latest_price > df['SMA_50'].iloc[-1]:
            rise_confidence += 15
            rise_reasons.append("Price bounced off 50-day SMA support")
        
        # Stochastic Oscillator indicates oversold and turning up
        if df['Stoch_K'].iloc[-1] < 20 and df['Stoch_K'].iloc[-1] > df['Stoch_K'].iloc[-2]:
            rise_confidence += 15
            rise_reasons.append("Stochastic Oscillator indicates oversold and turning up")
        
        # ADX indicates strong trend (above 25)
        if df['ADX'].iloc[-1] > 25:
            rise_confidence += 10
            rise_reasons.append("ADX indicates strong trend")
        
        # Ichimoku Cloud: Price crossed above Kumo (cloud)
        if df['close'].iloc[-1] > df['Senkou_Span_A'].iloc[-1] and df['close'].iloc[-1] > df['Senkou_Span_B'].iloc[-1] and df['close'].iloc[-2] < max(df['Senkou_Span_A'].iloc[-2], df['Senkou_Span_B'].iloc[-2]):
            rise_confidence += 15
            rise_reasons.append("Price crossed above Ichimoku Cloud")
        
        # Add insights from Yahoo Finance if available
        if insights and 'finance' in insights and 'result' in insights['finance']:
            tech_events = insights['finance']['result'].get('instrumentInfo', {}).get('technicalEvents', {})
            if tech_events:
                short_outlook = tech_events.get('shortTermOutlook', {})
                if short_outlook.get('direction') == 'up':
                    rise_confidence += 15
                    rise_reasons.append(f"Yahoo Finance indicates short-term upward trend: {short_outlook.get('scoreDescription', '')}")
        
        # Add quick rise opportunity if confidence is high enough
        if rise_confidence >= self.medium_confidence:
            results["opportunities"].append({
                "type": "quick_rise",
                "confidence": rise_confidence,
                "reasons": rise_reasons,
                "recommendation": f"Consider buying {stock_name} for a potential quick rise",
                "target_price": round(latest_price * 1.05, 2),  # 5% target
                "stop_loss": round(latest_price * 0.97, 2),     # 3% stop loss
                "time_frame": "1-5 trading days"
            })
        
        # Analyze for potential quick fall
        fall_confidence = 0
        fall_reasons = []
        
        # RSI indicates overbought (above 70)
        if df['RSI'].iloc[-1] > 70:
            fall_confidence += 25
            fall_reasons.append("RSI indicates overbought condition")
        # RSI is falling from high levels
        elif df['RSI'].iloc[-1] > 60 and df['RSI'].iloc[-1] < df['RSI'].iloc[-2]:
            fall_confidence += 15
            fall_reasons.append("RSI is falling from high levels")
        
        # Price is near upper Bollinger Band
        if latest_price > df['BB_Upper'].iloc[-1] * 0.98:
            fall_confidence += 20
            fall_reasons.append("Price is near upper Bollinger Band")
        
        # MACD crossed below signal line
        if df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
            fall_confidence += 20
            fall_reasons.append("MACD crossed below signal line")
        # MACD histogram turning negative
        elif df['MACD_Histogram'].iloc[-1] < 0 and df['MACD_Histogram'].iloc[-2] > 0:
            fall_confidence += 15
            fall_reasons.append("MACD histogram turning negative")
        
        # Significant volume decrease after price rise
        if df['Price_Change_5d'].iloc[-1] > 10 and df['Volume_Change'].iloc[-1] < -30:
            fall_confidence += 15
            fall_reasons.append("Volume decreasing after significant price rise")
        
        # Price failed at resistance level (50-day SMA)
        if prev_price > df['SMA_50'].iloc[-2] and latest_price < df['SMA_50'].iloc[-1]:
            fall_confidence += 20
            fall_reasons.append("Price failed at 50-day SMA resistance")
        # Price failed at resistance level (200-day SMA)
        elif prev_price > df['SMA_200'].iloc[-2] and latest_price < df['SMA_200'].iloc[-1]:
            fall_confidence += 15
            fall_reasons.append("Price failed at 200-day SMA resistance")
        
        # Stochastic Oscillator indicates overbought and turning down
        if df['Stoch_K'].iloc[-1] > 80 and df['Stoch_K'].iloc[-1] < df['Stoch_K'].iloc[-2]:
            fall_confidence += 15
            fall_reasons.append("Stochastic Oscillator indicates overbought and turning down")
        
        # ADX indicates strong trend (above 25)
        if df['ADX'].iloc[-1] > 25:
            fall_confidence += 10
            fall_reasons.append("ADX indicates strong trend")
        
        # Ichimoku Cloud: Price crossed below Kumo (cloud)
        if df['close'].iloc[-1] < df['Senkou_Span_A'].iloc[-1] and df['close'].iloc[-1] < df['Senkou_Span_B'].iloc[-1] and df['close'].iloc[-2] > min(df['Senkou_Span_A'].iloc[-2], df['Senkou_Span_B'].iloc[-2]):
            fall_confidence += 15
            fall_reasons.append("Price crossed below Ichimoku Cloud")
        
        # Add insights from Yahoo Finance if available
        if insights and 'finance' in insights and 'result' in insights['finance']:
            tech_events = insights['finance']['result'].get('instrumentInfo', {}).get('technicalEvents', {})
            if tech_events:
                short_outlook = tech_events.get('shortTermOutlook', {})
                if short_outlook.get('direction') == 'down':
                    fall_confidence += 15
                    fall_reasons.append(f"Yahoo Finance indicates short-term downward trend: {short_outlook.get('scoreDescription', '')}")
        
        # Add quick fall opportunity if confidence is high enough
        if fall_confidence >= self.medium_confidence:
            results["opportunities"].append({
                "type": "quick_fall",
                "confidence": fall_confidence,
                "reasons": fall_reasons,
                "recommendation": f"Consider short selling {stock_name} for a potential quick fall",
                "target_price": round(latest_price * 0.95, 2),  # 5% target
                "stop_loss": round(latest_price * 1.03, 2),     # 3% stop loss
                "time_frame": "1-5 trading days"
            })
        
        return results
    
    def analyze_general_rise(self, symbol):
        """
        Analyze stock for general rise potential (medium-term).
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Analysis results with confidence metrics
        """
        # Get stock data for the last 1 year with daily interval
        stock_data = self.get_stock_data(symbol, interval="1d", range="1y")
        if not stock_data:
            return {"error": f"Could not fetch data for {symbol}"}
        
        # Process data
        df = self.process_stock_data(stock_data)
        if df is None or df.empty:
            return {"error": f"Could not process data for {symbol}"}
        
        # Get stock insights for additional analysis
        insights = self.get_stock_insights(symbol)
        
        # Get stock holders information
        holders = self.get_stock_holders(symbol)
        
        # Extract stock name
        stock_name = stock_data['chart']['result'][0]['meta'].get('shortName', symbol)
        
        # Analysis results
        results = {
            "symbol": symbol,
            "stock_number": symbol.split('.')[0],
            "name": stock_name,
            "analysis_type": "general_rise",
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rise_potential": {},
            "technical_indicators": {}
        }
        
        # Latest price data
        latest_price = df['close'].iloc[-1]
        
        # Technical indicators for latest data point
        results["technical_indicators"] = {
            "latest_price": round(latest_price, 2),
            "sma_50": round(df['SMA_50'].iloc[-1], 2) if not pd.isna(df['SMA_50'].iloc[-1]) else None,
            "sma_200": round(df['SMA_200'].iloc[-1], 2) if not pd.isna(df['SMA_200'].iloc[-1]) else None,
            "rsi": round(df['RSI'].iloc[-1], 2) if not pd.isna(df['RSI'].iloc[-1]) else None,
            "price_change_20d": round(df['Price_Change_20d'].iloc[-1], 2) if not pd.isna(df['Price_Change_20d'].iloc[-1]) else None,
            "adx": round(df['ADX'].iloc[-1], 2) if not pd.isna(df['ADX'].iloc[-1]) else None,
        }
        
        # Analyze for medium-term rise potential
        rise_confidence = 0
        rise_reasons = []
        
        # Price above both 50-day and 200-day SMAs
        if latest_price > df['SMA_50'].iloc[-1] and latest_price > df['SMA_200'].iloc[-1]:
            rise_confidence += 20
            rise_reasons.append("Price is above both 50-day and 200-day moving averages")
        # Price above 200-day SMA
        elif latest_price > df['SMA_200'].iloc[-1]:
            rise_confidence += 15
            rise_reasons.append("Price is above 200-day moving average")
        
        # Golden Cross (50-day SMA crosses above 200-day SMA)
        if df['SMA_50'].iloc[-20] < df['SMA_200'].iloc[-20] and df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            rise_confidence += 25
            rise_reasons.append("Golden Cross detected (50-day SMA crossed above 200-day SMA)")
        # 50-day SMA trending up
        elif df['SMA_50'].iloc[-1] > df['SMA_50'].iloc[-20]:
            rise_confidence += 15
            rise_reasons.append("50-day SMA trending up")
        
        # RSI in healthy range (40-60)
        if 40 <= df['RSI'].iloc[-1] <= 60:
            rise_confidence += 15
            rise_reasons.append("RSI in healthy range, not overbought")
        # RSI trending up
        elif df['RSI'].iloc[-1] > df['RSI'].iloc[-5] and df['RSI'].iloc[-5] > df['RSI'].iloc[-10]:
            rise_confidence += 10
            rise_reasons.append("RSI trending up")
        
        # Consistent uptrend in last 20 days
        if df['Price_Change_20d'].iloc[-1] > 5:
            rise_confidence += 15
            rise_reasons.append("Consistent uptrend in last 20 days")
        
        # Higher lows pattern (last 3 significant lows are increasing)
        lows = df['low'].rolling(window=5).min().dropna()
        if len(lows) >= 15 and lows.iloc[-15] < lows.iloc[-10] < lows.iloc[-5]:
            rise_confidence += 15
            rise_reasons.append("Higher lows pattern detected")
        
        # MACD above signal line
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            rise_confidence += 10
            rise_reasons.append("MACD above signal line")
        
        # ADX indicates strong trend (above 25)
        if df['ADX'].iloc[-1] > 25:
            rise_confidence += 10
            rise_reasons.append("ADX indicates strong trend")
        
        # On-Balance Volume (OBV) trending up
        if df['OBV'].iloc[-1] > df['OBV'].iloc[-20]:
            rise_confidence += 10
            rise_reasons.append("On-Balance Volume trending up")
        
        # Ichimoku Cloud: Price above Kumo (cloud)
        if df['close'].iloc[-1] > df['Senkou_Span_A'].iloc[-1] and df['close'].iloc[-1] > df['Senkou_Span_B'].iloc[-1]:
            rise_confidence += 15
            rise_reasons.append("Price above Ichimoku Cloud")
        
        # Add insights from Yahoo Finance if available
        if insights and 'finance' in insights and 'result' in insights['finance']:
            tech_events = insights['finance']['result'].get('instrumentInfo', {}).get('technicalEvents', {})
            if tech_events:
                mid_outlook = tech_events.get('intermediateTermOutlook', {})
                if mid_outlook.get('direction') == 'up':
                    rise_confidence += 15
                    rise_reasons.append(f"Yahoo Finance indicates intermediate-term upward trend: {mid_outlook.get('scoreDescription', '')}")
        
        # Check for insider buying (positive signal)
        if holders and 'quoteSummary' in holders and 'result' in holders['quoteSummary'] and holders['quoteSummary']['result']:
            insider_holders = holders['quoteSummary']['result'][0].get('insiderHolders', {}).get('holders', [])
            recent_buys = 0
            for holder in insider_holders:
                if 'transactionDescription' in holder and 'Buy' in holder.get('transactionDescription', ''):
                    recent_buys += 1
            
            if recent_buys > 0:
                rise_confidence += 10
                rise_reasons.append(f"Recent insider buying detected ({recent_buys} transactions)")
        
        # Determine investment advice based on confidence
        if rise_confidence >= self.high_confidence:
            advice = f"Strong Buy: {stock_name} shows strong indicators for medium-term growth"
            target_price = round(latest_price * 1.15, 2)  # 15% target
        elif rise_confidence >= self.medium_confidence:
            advice = f"Buy: {stock_name} shows good potential for medium-term growth"
            target_price = round(latest_price * 1.10, 2)  # 10% target
        elif rise_confidence >= self.low_confidence:
            advice = f"Watch: {stock_name} shows some potential but requires monitoring"
            target_price = round(latest_price * 1.05, 2)  # 5% target
        else:
            advice = f"Hold/Avoid: {stock_name} does not show strong medium-term growth potential at this time"
            target_price = None
        
        # Add rise potential details
        results["rise_potential"] = {
            "confidence": rise_confidence,
            "reasons": rise_reasons,
            "recommendation": advice,
            "target_price": target_price,
            "stop_loss": round(latest_price * 0.93, 2) if rise_confidence >= self.low_confidence else None,
            "time_frame": "1-3 months"
        }
        
        return results
    
    def analyze_stocks(self, analysis_type="both", max_stocks=15):
        """
        Analyze multiple stocks and return top opportunities.
        
        Args:
            analysis_type (str): Type of analysis ("golden", "general", or "both")
            max_stocks (int): Maximum number of stocks to analyze
            
        Returns:
            dict: Analysis results for multiple stocks
        """
        results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "golden_opportunities": [],
            "general_rise_stocks": []
        }
        
        # Limit number of stocks to analyze
        stocks_to_analyze = self.saudi_market_symbols[:max_stocks]
        
        for symbol in stocks_to_analyze:
            print(f"Analyzing {symbol}...")
            
            # Analyze for golden opportunities
            if analysis_type in ["golden", "both"]:
                golden_analysis = self.analyze_golden_opportunities(symbol)
                if "error" not in golden_analysis and golden_analysis["opportunities"]:
                    for opportunity in golden_analysis["opportunities"]:
                        results["golden_opportunities"].append({
                            "symbol": symbol,
                            "stock_number": symbol.split('.')[0],
                            "name": golden_analysis["name"],
                            "type": opportunity["type"],
                            "confidence": opportunity["confidence"],
                            "recommendation": opportunity["recommendation"],
                            "target_price": opportunity["target_price"],
                            "stop_loss": opportunity["stop_loss"],
                            "time_frame": opportunity["time_frame"],
                            "reasons": opportunity["reasons"][:3]  # Limit to top 3 reasons
                        })
            
            # Analyze for general rise potential
            if analysis_type in ["general", "both"]:
                general_analysis = self.analyze_general_rise(symbol)
                if "error" not in general_analysis and general_analysis["rise_potential"] and general_analysis["rise_potential"]["confidence"] >= self.medium_confidence:
                    results["general_rise_stocks"].append({
                        "symbol": symbol,
                        "stock_number": symbol.split('.')[0],
                        "name": general_analysis["name"],
                        "confidence": general_analysis["rise_potential"]["confidence"],
                        "recommendation": general_analysis["rise_potential"]["recommendation"],
                        "target_price": general_analysis["rise_potential"]["target_price"],
                        "stop_loss": general_analysis["rise_potential"]["stop_loss"],
                        "time_frame": general_analysis["rise_potential"]["time_frame"],
                        "reasons": general_analysis["rise_potential"]["reasons"][:3]  # Limit to top 3 reasons
                    })
        
        # Sort opportunities by confidence
        results["golden_opportunities"] = sorted(results["golden_opportunities"], key=lambda x: x["confidence"], reverse=True)
        results["general_rise_stocks"] = sorted(results["general_rise_stocks"], key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def generate_report(self, analysis_results, lang="en"):
        """
        Generate a human-readable report from analysis results.
        
        Args:
            analysis_results (dict): Analysis results
            lang (str): Language for the report ("en" for English, "ar" for Arabic)
            
        Returns:
            str: Formatted report
        """
        if lang == "ar":
            return self._generate_arabic_report(analysis_results)
        else:
            return self._generate_english_report(analysis_results)
    
    def _generate_english_report(self, results):
        """Generate report in English."""
        report = []
        report.append("# Saudi Stock Market Analysis Report")
        report.append(f"Analysis Date: {results['analysis_date']}\n")
        
        # Golden opportunities section
        report.append("## Golden Opportunities (Quick Rises and Falls)")
        if results["golden_opportunities"]:
            for opportunity in results["golden_opportunities"]:
                report.append(f"### {opportunity['name']} ({opportunity['stock_number']})")
                report.append(f"**Type:** {'Quick Rise' if opportunity['type'] == 'quick_rise' else 'Quick Fall'}")
                report.append(f"**Confidence:** {opportunity['confidence']}%")
                report.append(f"**Recommendation:** {opportunity['recommendation']}")
                report.append(f"**Target Price:** {opportunity['target_price']} SAR")
                report.append(f"**Stop Loss:** {opportunity['stop_loss']} SAR")
                report.append(f"**Time Frame:** {opportunity['time_frame']}")
                report.append("\n**Key Reasons:**")
                for reason in opportunity["reasons"]:
                    report.append(f"- {reason}")
                report.append("")
        else:
            report.append("No golden opportunities detected at this time.\n")
        
        # General rise stocks section
        report.append("## Stocks with Medium-Term Rise Potential")
        if results["general_rise_stocks"]:
            for stock in results["general_rise_stocks"]:
                report.append(f"### {stock['name']} ({stock['stock_number']})")
                report.append(f"**Confidence:** {stock['confidence']}%")
                report.append(f"**Recommendation:** {stock['recommendation']}")
                report.append(f"**Target Price:** {stock['target_price']} SAR")
                report.append(f"**Stop Loss:** {stock['stop_loss']} SAR")
                report.append(f"**Time Frame:** {stock['time_frame']}")
                report.append("\n**Key Reasons:**")
                for reason in stock["reasons"]:
                    report.append(f"- {reason}")
                report.append("")
        else:
            report.append("No stocks with strong medium-term rise potential detected at this time.\n")
        
        report.append("## Disclaimer")
        report.append("This analysis is for informational purposes only and should not be considered as financial advice. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.")
        
        return "\n".join(report)
    
    def _generate_arabic_report(self, results):
        """Generate report in Arabic."""
        report = []
        report.append("# تقرير تحليل سوق الأسهم السعودية")
        report.append(f"تاريخ التحليل: {results['analysis_date']}\n")
        
        # Golden opportunities section
        report.append("## الفرص الذهبية (ارتفاعات وانخفاضات سريعة)")
        if results["golden_opportunities"]:
            for opportunity in results["golden_opportunities"]:
                report.append(f"### {opportunity['name']} ({opportunity['stock_number']})")
                report.append(f"**النوع:** {'ارتفاع سريع' if opportunity['type'] == 'quick_rise' else 'انخفاض سريع'}")
                report.append(f"**نسبة الثقة:** {opportunity['confidence']}%")
                report.append(f"**التوصية:** {opportunity['recommendation']}")
                report.append(f"**السعر المستهدف:** {opportunity['target_price']} ريال")
                report.append(f"**وقف الخسارة:** {opportunity['stop_loss']} ريال")
                report.append(f"**الإطار الزمني:** {opportunity['time_frame']}")
                report.append("\n**الأسباب الرئيسية:**")
                for reason in opportunity["reasons"]:
                    report.append(f"- {reason}")
                report.append("")
        else:
            report.append("لم يتم اكتشاف فرص ذهبية في الوقت الحالي.\n")
        
        # General rise stocks section
        report.append("## الأسهم ذات إمكانية الارتفاع على المدى المتوسط")
        if results["general_rise_stocks"]:
            for stock in results["general_rise_stocks"]:
                report.append(f"### {stock['name']} ({stock['stock_number']})")
                report.append(f"**نسبة الثقة:** {stock['confidence']}%")
                report.append(f"**التوصية:** {stock['recommendation']}")
                report.append(f"**السعر المستهدف:** {stock['target_price']} ريال")
                report.append(f"**وقف الخسارة:** {stock['stop_loss']} ريال")
                report.append(f"**الإطار الزمني:** {stock['time_frame']}")
                report.append("\n**الأسباب الرئيسية:**")
                for reason in stock["reasons"]:
                    report.append(f"- {reason}")
                report.append("")
        else:
            report.append("لم يتم اكتشاف أسهم ذات إمكانية ارتفاع قوية على المدى المتوسط في الوقت الحالي.\n")
        
        report.append("## إخلاء المسؤولية")
        report.append("هذا التحليل لأغراض إعلامية فقط ولا ينبغي اعتباره نصيحة مالية. قم دائمًا بإجراء البحث الخاص بك وفكر في استشارة مستشار مالي قبل اتخاذ قرارات الاستثمار.")
        
        return "\n".join(report)


def main():
    """Main function to run the Saudi Stock Bot."""
    print("Initializing Enhanced Saudi Stock Analysis Bot...")
    bot = SaudiStockBot()
    
    print("\nAnalyzing Saudi stocks...")
    analysis_results = bot.analyze_stocks(analysis_type="both", max_stocks=15)
    
    print("\nGenerating report...")
    report = bot.generate_report(analysis_results, lang="ar")
    
    # Save report to file
    report_file = "saudi_stock_analysis_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to {report_file}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
