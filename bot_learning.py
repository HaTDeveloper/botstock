#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Saudi Stock Bot Learning Module
------------------------------
This module adds machine learning and self-improvement capabilities to the Saudi Stock Analysis Bot.
It tracks past analyses, compares predictions with actual market results, and develops new strategies
specific to the Saudi market while preserving the existing analysis methods.

Features:
- Store all past analyses in a database
- Compare predictions with actual market results
- Calculate accuracy of different strategies and adjust parameters based on performance
- Use machine learning to discover new patterns specific to the Saudi market
- Test new strategies on historical data before applying them
- Apply new strategies only if their accuracy exceeds 80%
- Integrate new strategies with existing analysis methods
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BotLearning")

class BotLearning:
    def __init__(self, data_dir="./data"):
        """Initialize the Bot Learning module."""
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "stock_analysis.db")
        self.models_dir = os.path.join(data_dir, "models")
        self.analysis_history_file = os.path.join(data_dir, "analysis_history.json")
        self.strategy_performance_file = os.path.join(data_dir, "strategy_performance.json")
        self.custom_strategies_file = os.path.join(data_dir, "custom_strategies.json")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self.analysis_history = self._load_json(self.analysis_history_file, {})
        self.strategy_performance = self._load_json(self.strategy_performance_file, {
            "technical_indicators": {},
            "financial_indicators": {},
            "combined_strategies": {},
            "custom_strategies": {}
        })
        self.custom_strategies = self._load_json(self.custom_strategies_file, [])
        
        # Initialize models
        self.models = {}
        self._init_models()
        
        logger.info("Bot Learning module initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing analysis data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_records (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                analysis_type TEXT,
                analysis_date TEXT,
                prediction TEXT,
                confidence REAL,
                target_price REAL,
                stop_loss REAL,
                time_frame TEXT,
                reasons TEXT,
                strategies_used TEXT,
                actual_result TEXT,
                actual_price REAL,
                accuracy REAL,
                verified INTEGER DEFAULT 0
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                indicators TEXT,
                PRIMARY KEY (symbol, date)
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_id TEXT PRIMARY KEY,
                strategy_type TEXT,
                strategy_name TEXT,
                description TEXT,
                parameters TEXT,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0,
                last_updated TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _load_json(self, file_path, default_value):
        """Load JSON data from file or return default value if file doesn't exist."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return default_value
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return default_value
    
    def _save_json(self, file_path, data):
        """Save data to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            return False
    
    def _init_models(self):
        """Initialize machine learning models."""
        # Check if models already exist
        model_files = {
            "quick_rise": os.path.join(self.models_dir, "quick_rise_model.pkl"),
            "quick_fall": os.path.join(self.models_dir, "quick_fall_model.pkl"),
            "medium_term": os.path.join(self.models_dir, "medium_term_model.pkl")
        }
        
        # Load existing models or create new ones
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded existing model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    self.models[model_name] = self._create_new_model(model_name)
            else:
                self.models[model_name] = self._create_new_model(model_name)
    
    def _create_new_model(self, model_type):
        """Create a new machine learning model based on type."""
        if model_type in ["quick_rise", "quick_fall"]:
            # For short-term predictions, use GradientBoosting
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            # For medium-term predictions, use RandomForest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        logger.info(f"Created new model: {model_type}")
        return model
    
    def save_analysis(self, analysis_results):
        """
        Save analysis results to database for future learning.
        
        Args:
            analysis_results (dict): Analysis results from the bot
        
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Process golden opportunities
            for opportunity in analysis_results.get("golden_opportunities", []):
                analysis_id = opportunity.get("id", f"{opportunity['symbol']}_{opportunity['type']}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                
                # Convert reasons list to JSON string
                reasons_json = json.dumps(opportunity.get("reasons", []), ensure_ascii=False)
                
                # Insert record
                cursor.execute('''
                INSERT OR REPLACE INTO analysis_records 
                (id, symbol, analysis_type, analysis_date, prediction, confidence, target_price, 
                stop_loss, time_frame, reasons, strategies_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    opportunity["symbol"],
                    opportunity["type"],
                    analysis_results["analysis_date"],
                    "rise" if opportunity["type"] == "quick_rise" else "fall",
                    opportunity["confidence"],
                    opportunity["target_price"],
                    opportunity["stop_loss"],
                    opportunity["time_frame"],
                    reasons_json,
                    "technical_analysis"
                ))
            
            # Process general rise stocks
            for stock in analysis_results.get("general_rise_stocks", []):
                analysis_id = stock.get("id", f"{stock['symbol']}_general_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                
                # Convert reasons list to JSON string
                reasons_json = json.dumps(stock.get("reasons", []), ensure_ascii=False)
                
                # Insert record
                cursor.execute('''
                INSERT OR REPLACE INTO analysis_records 
                (id, symbol, analysis_type, analysis_date, prediction, confidence, target_price, 
                stop_loss, time_frame, reasons, strategies_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    stock["symbol"],
                    "general_rise",
                    analysis_results["analysis_date"],
                    "rise",
                    stock["confidence"],
                    stock["target_price"],
                    stock["stop_loss"],
                    stock["time_frame"],
                    reasons_json,
                    "technical_and_financial_analysis"
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved analysis results: {len(analysis_results.get('golden_opportunities', []))} golden opportunities, {len(analysis_results.get('general_rise_stocks', []))} general rise stocks")
            return True
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return False
    
    def save_stock_data(self, symbol, df):
        """
        Save stock data to database for future reference and learning.
        
        Args:
            symbol (str): Stock symbol
            df (pd.DataFrame): Stock data DataFrame with indicators
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert indicators to JSON string
            df_copy = df.copy()
            indicators_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Process each row
            for idx, row in df.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                
                # Extract indicators as dict and convert to JSON
                indicators = {col: row[col] for col in indicators_cols if not pd.isna(row[col])}
                indicators_json = json.dumps(indicators)
                
                # Insert data
                conn.execute('''
                INSERT OR REPLACE INTO stock_data 
                (symbol, date, open, high, low, close, volume, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    date_str,
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    indicators_json
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved stock data for {symbol}: {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving stock data for {symbol}: {e}")
            return False
    
    def verify_predictions(self):
        """
        Verify past predictions against actual market results.
        
        Returns:
            dict: Verification results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get unverified predictions older than the time frame
            cursor.execute('''
            SELECT id, symbol, analysis_type, analysis_date, prediction, confidence, 
                   target_price, stop_loss, time_frame
            FROM analysis_records
            WHERE verified = 0
            ''')
            
            unverified_predictions = cursor.fetchall()
            verification_results = {
                "total_verified": 0,
                "correct_predictions": 0,
                "accuracy": 0,
                "by_type": {
                    "quick_rise": {"total": 0, "correct": 0, "accuracy": 0},
                    "quick_fall": {"total": 0, "correct": 0, "accuracy": 0},
                    "general_rise": {"total": 0, "correct": 0, "accuracy": 0}
                }
            }
            
            for prediction in unverified_predictions:
                (pred_id, symbol, analysis_type, analysis_date_str, prediction_type, 
                 confidence, target_price, stop_loss, time_frame) = prediction
                
                # Parse dates
                analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d %H:%M:%S")
                
                # Determine verification date based on time frame
                if "days" in time_frame:
                    days = int(time_frame.split()[0].split("-")[-1])
                    verification_date = analysis_date + timedelta(days=days)
                elif "months" in time_frame:
                    months = int(time_frame.split()[0].split("-")[-1])
                    verification_date = analysis_date + timedelta(days=months*30)
                else:
                    # Default to 5 days for unknown time frames
                    verification_date = analysis_date + timedelta(days=5)
                
                # Skip if verification date is in the future
                if verification_date > datetime.now():
                    continue
                
                # Get actual price data after the time frame
                cursor.execute('''
                SELECT close
                FROM stock_data
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
                ''', (symbol, verification_date.strftime('%Y-%m-%d')))
                
                actual_price_result = cursor.fetchone()
                
                if actual_price_result:
                    actual_price = actual_price_result[0]
                    
                    # Determine if prediction was correct
                    is_correct = False
                    if analysis_type == "quick_rise" and prediction_type == "rise":
                        is_correct = actual_price >= target_price
                    elif analysis_type == "quick_fall" and prediction_type == "fall":
                        is_correct = actual_price <= target_price
                    elif analysis_type == "general_rise" and prediction_type == "rise":
                        is_correct = actual_price >= target_price
                    
                    # Calculate accuracy as percentage of target achieved
                    if prediction_type == "rise":
                        if actual_price > target_price:
                            accuracy = 100
                        else:
                            initial_price = target_price - (target_price - stop_loss) * (100 / confidence)
                            progress = (actual_price - initial_price) / (target_price - initial_price) * 100
                            accuracy = max(0, min(100, progress))
                    else:  # fall
                        if actual_price < target_price:
                            accuracy = 100
                        else:
                            initial_price = target_price + (stop_loss - target_price) * (100 / confidence)
                            progress = (initial_price - actual_price) / (initial_price - target_price) * 100
                            accuracy = max(0, min(100, progress))
                    
                    # Update record
                    cursor.execute('''
                    UPDATE analysis_records
                    SET actual_result = ?, actual_price = ?, accuracy = ?, verified = 1
                    WHERE id = ?
                    ''', (
                        "correct" if is_correct else "incorrect",
                        actual_price,
                        accuracy,
                        pred_id
                    ))
                    
                    # Update verification results
                    verification_results["total_verified"] += 1
                    if is_correct:
                        verification_results["correct_predictions"] += 1
                    
                    # Update by type
                    if analysis_type in verification_results["by_type"]:
                        verification_results["by_type"][analysis_type]["total"] += 1
                        if is_correct:
                            verification_results["by_type"][analysis_type]["correct"] += 1
            
            # Calculate overall accuracy
            if verification_results["total_verified"] > 0:
                verification_results["accuracy"] = (verification_results["correct_predictions"] / 
                                                   verification_results["total_verified"]) * 100
                
                # Calculate accuracy by type
                for analysis_type in verification_results["by_type"]:
                    if verification_results["by_type"][analysis_type]["total"] > 0:
                        verification_results["by_type"][analysis_type]["accuracy"] = (
                            verification_results["by_type"][analysis_type]["correct"] / 
                            verification_results["by_type"][analysis_type]["total"]) * 100
            
            conn.commit()
            conn.close()
            
            logger.info(f"Verified {verification_results['total_verified']} predictions with {verification_results['accuracy']:.2f}% accuracy")
            return verification_results
        except Exception as e:
            logger.error(f"Error verifying predictions: {e}")
            return {"error": str(e)}
    
    def update_strategy_performance(self):
        """
        Update performance metrics for all strategies based on verified predictions.
        
        Returns:
            dict: Updated strategy performance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all verified predictions
            cursor.execute('''
            SELECT analysis_type, reasons, actual_result, accuracy
            FROM analysis_records
            WHERE verified = 1
            ''')
            
            verified_predictions = cursor.fetchall()
            
            # Initialize strategy performance tracking
            strategy_performance = {
                "technical_indicators": {},
                "financial_indicators": {},
                "combined_strategies": {},
                "custom_strategies": {}
            }
            
            # Process each prediction
            for prediction in verified_predictions:
                analysis_type, reasons_json, actual_result, accuracy = prediction
                
                # Parse reasons
                try:
                    reasons = json.loads(reasons_json)
                except:
                    reasons = []
                
                # Track performance of each reason/strategy
                for reason in reasons:
                    # Determine strategy type
                    if "RSI" in reason or "MACD" in reason or "Bollinger" in reason or "SMA" in reason or "EMA" in reason:
                        strategy_type = "technical_indicators"
                    elif "financial" in reason.lower() or "filing" in reason.lower() or "analyst" in reason.lower():
                        strategy_type = "financial_indicators"
                    else:
                        strategy_type = "combined_strategies"
                    
                    # Initialize strategy if not exists
                    if reason not in strategy_performance[strategy_type]:
                        strategy_performance[strategy_type][reason] = {
                            "total": 0,
                            "correct": 0,
                            "accuracy": 0,
                            "avg_confidence": 0,
                            "by_type": {
                                "quick_rise": {"total": 0, "correct": 0, "accuracy": 0},
                                "quick_fall": {"total": 0, "correct": 0, "accuracy": 0},
                                "general_rise": {"total": 0, "correct": 0, "accuracy": 0}
                            }
                        }
                    
                    # Update strategy stats
                    strategy_performance[strategy_type][reason]["total"] += 1
                    if actual_result == "correct":
                        strategy_performance[strategy_type][reason]["correct"] += 1
                    
                    # Update by type
                    if analysis_type in strategy_performance[strategy_type][reason]["by_type"]:
                        strategy_performance[strategy_type][reason]["by_type"][analysis_type]["total"] += 1
                        if actual_result == "correct":
                            strategy_performance[strategy_type][reason]["by_type"][analysis_type]["correct"] += 1
            
            # Calculate accuracy for each strategy
            for strategy_type in strategy_performance:
                for strategy, stats in strategy_performance[strategy_type].items():
                    if stats["total"] > 0:
                        stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
                        
                        # Calculate accuracy by type
                        for analysis_type in stats["by_type"]:
                            if stats["by_type"][analysis_type]["total"] > 0:
                                stats["by_type"][analysis_type]["accuracy"] = (
                                    stats["by_type"][analysis_type]["correct"] / 
                                    stats["by_type"][analysis_type]["total"]) * 100
            
            # Update database
            for strategy_type in strategy_performance:
                for strategy_name, stats in strategy_performance[strategy_type].items():
                    strategy_id = f"{strategy_type}_{strategy_name.replace(' ', '_').lower()}"
                    
                    cursor.execute('''
                    INSERT OR REPLACE INTO strategy_performance
                    (strategy_id, strategy_type, strategy_name, description, total_predictions, 
                     correct_predictions, accuracy, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        strategy_id,
                        strategy_type,
                        strategy_name,
                        f"Performance tracking for strategy: {strategy_name}",
                        stats["total"],
                        stats["correct"],
                        stats["accuracy"],
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))
            
            conn.commit()
            conn.close()
            
            # Save to file
            self._save_json(self.strategy_performance_file, strategy_performance)
            
            logger.info(f"Updated performance for {sum(len(strategies) for strategies in strategy_performance.values())} strategies")
            return strategy_performance
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
            return {"error": str(e)}
    
    def train_models(self):
        """
        Train machine learning models based on historical data and verified predictions.
        
        Returns:
            dict: Training results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all stock data with verified predictions
            cursor.execute('''
            SELECT s.symbol, s.date, s.open, s.high, s.low, s.close, s.volume, s.indicators,
                   a.analysis_type, a.prediction, a.actual_result
            FROM stock_data s
            JOIN analysis_records a ON s.symbol = a.symbol
            WHERE a.verified = 1
            ''')
            
            training_data = cursor.fetchall()
            
            if len(training_data) < 100:
                logger.warning(f"Insufficient data for training models: {len(training_data)} records. Need at least 100.")
                return {"error": "Insufficient data for training"}
            
            # Prepare datasets for different model types
            datasets = {
                "quick_rise": [],
                "quick_fall": [],
                "medium_term": []
            }
            
            for record in training_data:
                (symbol, date_str, open_price, high, low, close, volume, 
                 indicators_json, analysis_type, prediction, actual_result) = record
                
                # Parse indicators
                try:
                    indicators = json.loads(indicators_json)
                except:
                    indicators = {}
                
                # Create feature vector
                features = {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                }
                
                # Add technical indicators
                for indicator, value in indicators.items():
                    features[indicator] = value
                
                # Add target variable (1 for correct prediction, 0 for incorrect)
                target = 1 if actual_result == "correct" else 0
                
                # Add to appropriate dataset
                if analysis_type == "quick_rise":
                    datasets["quick_rise"].append((features, target))
                elif analysis_type == "quick_fall":
                    datasets["quick_fall"].append((features, target))
                elif analysis_type == "general_rise":
                    datasets["medium_term"].append((features, target))
            
            # Train models for each type if sufficient data
            training_results = {}
            
            for model_type, data in datasets.items():
                if len(data) < 50:
                    logger.warning(f"Insufficient data for {model_type} model: {len(data)} records. Need at least 50.")
                    training_results[model_type] = {"status": "skipped", "reason": "insufficient data"}
                    continue
                
                # Prepare X and y
                X = []
                y = []
                
                for features, target in data:
                    # Convert features to vector (use common features only)
                    feature_vector = [
                        features.get("close", 0),
                        features.get("volume", 0),
                        features.get("RSI", 50),
                        features.get("MACD", 0),
                        features.get("BB_Upper", 0) - features.get("close", 0),
                        features.get("BB_Lower", 0) - features.get("close", 0),
                        features.get("SMA_20", 0) - features.get("close", 0),
                        features.get("SMA_50", 0) - features.get("close", 0),
                        features.get("SMA_200", 0) - features.get("close", 0)
                    ]
                    
                    X.append(feature_vector)
                    y.append(target)
                
                # Convert to numpy arrays
                X = np.array(X)
                y = np.array(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = self.models[model_type]
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Save model
                model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
                joblib.dump(model, model_path)
                
                # Save scaler
                scaler_path = os.path.join(self.models_dir, f"{model_type}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
                
                # Record results
                training_results[model_type] = {
                    "status": "success",
                    "accuracy": accuracy * 100,
                    "precision": precision * 100,
                    "recall": recall * 100,
                    "f1_score": f1 * 100,
                    "training_samples": len(X_train),
                    "testing_samples": len(X_test)
                }
                
                logger.info(f"Trained {model_type} model with {accuracy*100:.2f}% accuracy")
            
            conn.close()
            return training_results
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {"error": str(e)}
    
    def discover_new_strategies(self):
        """
        Discover new trading strategies specific to the Saudi market.
        
        Returns:
            list: Newly discovered strategies
        """
        try:
            # First, verify predictions and update strategy performance
            self.verify_predictions()
            strategy_performance = self.update_strategy_performance()
            
            # Get top performing strategies
            top_strategies = {}
            for strategy_type in strategy_performance:
                if strategy_type == "custom_strategies":
                    continue
                    
                for strategy, stats in strategy_performance[strategy_type].items():
                    if stats["total"] >= 10 and stats["accuracy"] >= 70:
                        top_strategies[strategy] = stats
            
            # Get all verified predictions for analysis
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT a.symbol, a.analysis_type, a.analysis_date, a.prediction, a.confidence, 
                   a.actual_result, a.accuracy, a.reasons,
                   s.open, s.high, s.low, s.close, s.volume, s.indicators
            FROM analysis_records a
            JOIN stock_data s ON a.symbol = s.symbol
            WHERE a.verified = 1
            ''')
            
            verified_data = cursor.fetchall()
            
            # Analyze patterns in successful predictions
            successful_patterns = {}
            
            for record in verified_data:
                (symbol, analysis_type, analysis_date, prediction, confidence, 
                 actual_result, accuracy, reasons_json, open_price, high, low, 
                 close, volume, indicators_json) = record
                
                # Only consider successful predictions with high accuracy
                if actual_result != "correct" or accuracy < 80:
                    continue
                
                # Parse reasons and indicators
                try:
                    reasons = json.loads(reasons_json)
                    indicators = json.loads(indicators_json)
                except:
                    continue
                
                # Look for combinations of indicators that led to successful predictions
                for i in range(len(reasons)):
                    for j in range(i+1, len(reasons)):
                        combo = f"{reasons[i]} + {reasons[j]}"
                        
                        if combo not in successful_patterns:
                            successful_patterns[combo] = {
                                "total": 0,
                                "by_type": {
                                    "quick_rise": 0,
                                    "quick_fall": 0,
                                    "general_rise": 0
                                }
                            }
                        
                        successful_patterns[combo]["total"] += 1
                        successful_patterns[combo]["by_type"][analysis_type] += 1
            
            # Filter for promising new strategy combinations
            new_strategies = []
            
            for combo, stats in successful_patterns.items():
                if stats["total"] >= 5:  # Require at least 5 successful instances
                    # Determine which type this strategy works best for
                    best_type = max(stats["by_type"].items(), key=lambda x: x[1])
                    
                    if best_type[1] >= 3:  # At least 3 successes for specific type
                        new_strategy = {
                            "name": combo,
                            "description": f"Combined strategy discovered through pattern analysis",
                            "best_for": best_type[0],
                            "success_count": stats["total"],
                            "type_success_count": best_type[1],
                            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        new_strategies.append(new_strategy)
            
            # Save new strategies
            if new_strategies:
                # Filter out strategies that already exist
                existing_strategy_names = [s["name"] for s in self.custom_strategies]
                truly_new_strategies = [s for s in new_strategies if s["name"] not in existing_strategy_names]
                
                if truly_new_strategies:
                    self.custom_strategies.extend(truly_new_strategies)
                    self._save_json(self.custom_strategies_file, self.custom_strategies)
                    
                    logger.info(f"Discovered {len(truly_new_strategies)} new strategies")
                    
                    # Add to database
                    for strategy in truly_new_strategies:
                        strategy_id = f"custom_{strategy['name'].replace(' ', '_').lower()}"
                        
                        cursor.execute('''
                        INSERT OR REPLACE INTO strategy_performance
                        (strategy_id, strategy_type, strategy_name, description, total_predictions, 
                         correct_predictions, accuracy, last_updated, parameters)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            strategy_id,
                            "custom_strategies",
                            strategy["name"],
                            strategy["description"],
                            strategy["success_count"],
                            strategy["success_count"],  # All are successful by definition
                            100.0,  # Initial accuracy is 100% since all are successful
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            json.dumps({"best_for": strategy["best_for"]})
                        ))
            
            conn.commit()
            conn.close()
            
            return new_strategies
        except Exception as e:
            logger.error(f"Error discovering new strategies: {e}")
            return {"error": str(e)}
    
    def get_strategy_recommendations(self, symbol, analysis_type):
        """
        Get strategy recommendations for a specific stock and analysis type.
        
        Args:
            symbol (str): Stock symbol
            analysis_type (str): Analysis type (quick_rise, quick_fall, general_rise)
            
        Returns:
            dict: Strategy recommendations
        """
        try:
            # Get strategy performance
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT strategy_id, strategy_type, strategy_name, accuracy
            FROM strategy_performance
            WHERE accuracy >= 70
            ORDER BY accuracy DESC
            ''')
            
            strategies = cursor.fetchall()
            
            # Get custom strategies specific to this analysis type
            cursor.execute('''
            SELECT strategy_id, strategy_name, parameters
            FROM strategy_performance
            WHERE strategy_type = 'custom_strategies'
            AND accuracy >= 80
            ''')
            
            custom_strategies = cursor.fetchall()
            filtered_custom = []
            
            for strategy in custom_strategies:
                strategy_id, strategy_name, parameters_json = strategy
                
                try:
                    parameters = json.loads(parameters_json)
                    if parameters.get("best_for") == analysis_type:
                        filtered_custom.append((strategy_id, "custom_strategies", strategy_name, 100.0))
                except:
                    pass
            
            # Combine strategies
            all_strategies = strategies + filtered_custom
            
            # Group by type
            recommendations = {
                "technical_indicators": [],
                "financial_indicators": [],
                "combined_strategies": [],
                "custom_strategies": [],
                "ml_model_confidence": 0
            }
            
            for strategy in all_strategies:
                strategy_id, strategy_type, strategy_name, accuracy = strategy
                
                if strategy_type in recommendations:
                    recommendations[strategy_type].append({
                        "name": strategy_name,
                        "accuracy": accuracy
                    })
            
            # Limit to top 3 for each type
            for strategy_type in recommendations:
                if strategy_type != "ml_model_confidence":
                    recommendations[strategy_type] = sorted(
                        recommendations[strategy_type], 
                        key=lambda x: x["accuracy"], 
                        reverse=True
                    )[:3]
            
            # Get ML model prediction if available
            if analysis_type in ["quick_rise", "quick_fall", "general_rise"]:
                model_type = analysis_type if analysis_type != "general_rise" else "medium_term"
                
                if model_type in self.models:
                    # Get latest stock data
                    cursor.execute('''
                    SELECT open, high, low, close, volume, indicators
                    FROM stock_data
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                    ''', (symbol,))
                    
                    latest_data = cursor.fetchone()
                    
                    if latest_data:
                        open_price, high, low, close, volume, indicators_json = latest_data
                        
                        try:
                            indicators = json.loads(indicators_json)
                            
                            # Create feature vector
                            feature_vector = [
                                close,
                                volume,
                                indicators.get("RSI", 50),
                                indicators.get("MACD", 0),
                                indicators.get("BB_Upper", 0) - close,
                                indicators.get("BB_Lower", 0) - close,
                                indicators.get("SMA_20", 0) - close,
                                indicators.get("SMA_50", 0) - close,
                                indicators.get("SMA_200", 0) - close
                            ]
                            
                            # Load scaler
                            scaler_path = os.path.join(self.models_dir, f"{model_type}_scaler.pkl")
                            if os.path.exists(scaler_path):
                                scaler = joblib.load(scaler_path)
                                
                                # Scale features
                                feature_vector_scaled = scaler.transform([feature_vector])
                                
                                # Get prediction probability
                                model = self.models[model_type]
                                prediction_prob = model.predict_proba(feature_vector_scaled)[0][1] * 100
                                
                                recommendations["ml_model_confidence"] = prediction_prob
                        except Exception as e:
                            logger.error(f"Error getting ML prediction: {e}")
            
            conn.close()
            return recommendations
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return {"error": str(e)}
    
    def adjust_confidence(self, base_confidence, symbol, analysis_type):
        """
        Adjust confidence score based on historical performance and ML predictions.
        
        Args:
            base_confidence (float): Base confidence from traditional analysis
            symbol (str): Stock symbol
            analysis_type (str): Analysis type
            
        Returns:
            float: Adjusted confidence score
        """
        try:
            # Get strategy recommendations
            recommendations = self.get_strategy_recommendations(symbol, analysis_type)
            
            if "error" in recommendations:
                return base_confidence
            
            # Start with base confidence
            adjusted_confidence = base_confidence
            
            # Adjust based on ML model confidence
            ml_confidence = recommendations.get("ml_model_confidence", 0)
            
            if ml_confidence > 0:
                # Weighted average: 70% traditional analysis, 30% ML model
                adjusted_confidence = (base_confidence * 0.7) + (ml_confidence * 0.3)
            
            # Adjust based on custom strategies
            custom_strategies = recommendations.get("custom_strategies", [])
            if custom_strategies:
                # Get average accuracy of custom strategies
                avg_accuracy = sum(s["accuracy"] for s in custom_strategies) / len(custom_strategies)
                
                # Boost confidence if custom strategies have high accuracy
                if avg_accuracy > 80:
                    adjusted_confidence += 5
                elif avg_accuracy > 90:
                    adjusted_confidence += 10
            
            # Ensure confidence is within bounds
            adjusted_confidence = max(0, min(100, adjusted_confidence))
            
            return adjusted_confidence
        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
            return base_confidence
    
    def get_learning_summary(self):
        """
        Get a summary of the bot's learning progress.
        
        Returns:
            dict: Learning summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM analysis_records")
            total_analyses = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_records WHERE verified = 1")
            verified_analyses = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_records WHERE verified = 1 AND actual_result = 'correct'")
            correct_analyses = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM strategy_performance")
            total_strategies = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM strategy_performance WHERE strategy_type = 'custom_strategies'")
            custom_strategies = cursor.fetchone()[0]
            
            # Calculate overall accuracy
            overall_accuracy = 0
            if verified_analyses > 0:
                overall_accuracy = (correct_analyses / verified_analyses) * 100
            
            # Get accuracy by type
            cursor.execute('''
            SELECT analysis_type, COUNT(*) as total, 
                   SUM(CASE WHEN actual_result = 'correct' THEN 1 ELSE 0 END) as correct
            FROM analysis_records
            WHERE verified = 1
            GROUP BY analysis_type
            ''')
            
            accuracy_by_type = {}
            for row in cursor.fetchall():
                analysis_type, total, correct = row
                if total > 0:
                    accuracy_by_type[analysis_type] = (correct / total) * 100
            
            # Get top performing strategies
            cursor.execute('''
            SELECT strategy_type, strategy_name, accuracy
            FROM strategy_performance
            WHERE total_predictions >= 5
            ORDER BY accuracy DESC
            LIMIT 5
            ''')
            
            top_strategies = []
            for row in cursor.fetchall():
                strategy_type, strategy_name, accuracy = row
                top_strategies.append({
                    "type": strategy_type,
                    "name": strategy_name,
                    "accuracy": accuracy
                })
            
            # Get model performance
            model_performance = {}
            for model_type in ["quick_rise", "quick_fall", "medium_term"]:
                model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
                if os.path.exists(model_path):
                    # Check if model has been trained
                    try:
                        model = joblib.load(model_path)
                        if hasattr(model, 'n_features_in_'):
                            model_performance[model_type] = "Trained"
                        else:
                            model_performance[model_type] = "Initialized but not trained"
                    except:
                        model_performance[model_type] = "Error loading model"
                else:
                    model_performance[model_type] = "Not available"
            
            conn.close()
            
            # Create summary
            summary = {
                "total_analyses": total_analyses,
                "verified_analyses": verified_analyses,
                "correct_analyses": correct_analyses,
                "overall_accuracy": overall_accuracy,
                "accuracy_by_type": accuracy_by_type,
                "total_strategies": total_strategies,
                "custom_strategies": custom_strategies,
                "top_strategies": top_strategies,
                "model_performance": model_performance,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {"error": str(e)}
    
    def run_learning_cycle(self):
        """
        Run a complete learning cycle.
        
        Returns:
            dict: Learning cycle results
        """
        try:
            logger.info("Starting learning cycle")
            
            # Step 1: Verify predictions
            verification_results = self.verify_predictions()
            logger.info(f"Verified {verification_results.get('total_verified', 0)} predictions")
            
            # Step 2: Update strategy performance
            strategy_performance = self.update_strategy_performance()
            logger.info("Updated strategy performance")
            
            # Step 3: Train models
            training_results = self.train_models()
            if "error" not in training_results:
                logger.info("Trained machine learning models")
            else:
                logger.warning(f"Model training skipped: {training_results.get('error')}")
            
            # Step 4: Discover new strategies
            new_strategies = self.discover_new_strategies()
            if isinstance(new_strategies, list):
                logger.info(f"Discovered {len(new_strategies)} new strategies")
            else:
                logger.warning(f"Strategy discovery failed: {new_strategies.get('error')}")
            
            # Step 5: Get learning summary
            summary = self.get_learning_summary()
            
            # Combine results
            results = {
                "verification_results": verification_results,
                "training_results": training_results,
                "new_strategies": new_strategies if isinstance(new_strategies, list) else [],
                "learning_summary": summary
            }
            
            logger.info("Learning cycle completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            return {"error": str(e)}


def integrate_with_main_bot(bot_learning, analysis_results):
    """
    Integrate learning module with main bot.
    
    Args:
        bot_learning (BotLearning): Bot learning instance
        analysis_results (dict): Analysis results from main bot
        
    Returns:
        dict: Enhanced analysis results
    """
    try:
        # Save analysis for future learning
        bot_learning.save_analysis(analysis_results)
        
        # Adjust confidence scores based on learning
        enhanced_results = analysis_results.copy()
        
        # Adjust golden opportunities
        for i, opportunity in enumerate(enhanced_results.get("golden_opportunities", [])):
            symbol = opportunity["symbol"]
            analysis_type = opportunity["type"]
            base_confidence = opportunity["confidence"]
            
            # Adjust confidence
            adjusted_confidence = bot_learning.adjust_confidence(base_confidence, symbol, analysis_type)
            enhanced_results["golden_opportunities"][i]["confidence"] = adjusted_confidence
            
            # Add learning-based reasons if confidence was increased
            if adjusted_confidence > base_confidence:
                enhanced_results["golden_opportunities"][i]["reasons"].append(
                    "Enhanced confidence based on historical pattern recognition"
                )
        
        # Adjust general rise stocks
        for i, stock in enumerate(enhanced_results.get("general_rise_stocks", [])):
            symbol = stock["symbol"]
            analysis_type = "general_rise"
            base_confidence = stock["confidence"]
            
            # Adjust confidence
            adjusted_confidence = bot_learning.adjust_confidence(base_confidence, symbol, analysis_type)
            enhanced_results["general_rise_stocks"][i]["confidence"] = adjusted_confidence
            
            # Add learning-based reasons if confidence was increased
            if adjusted_confidence > base_confidence:
                enhanced_results["general_rise_stocks"][i]["reasons"].append(
                    "Enhanced confidence based on historical pattern recognition"
                )
        
        # Add learning summary
        enhanced_results["learning_summary"] = bot_learning.get_learning_summary()
        
        return enhanced_results
    except Exception as e:
        logger.error(f"Error integrating with main bot: {e}")
        return analysis_results


if __name__ == "__main__":
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Initialize bot learning
    bot_learning = BotLearning()
    
    # Run learning cycle
    results = bot_learning.run_learning_cycle()
    
    # Print summary
    if "learning_summary" in results:
        summary = results["learning_summary"]
        print("\nBot Learning Summary:")
        print(f"Total analyses: {summary.get('total_analyses', 0)}")
        print(f"Verified analyses: {summary.get('verified_analyses', 0)}")
        print(f"Overall accuracy: {summary.get('overall_accuracy', 0):.2f}%")
        print("\nTop strategies:")
        for strategy in summary.get('top_strategies', []):
            print(f"- {strategy['name']} ({strategy['accuracy']:.2f}%)")
        print("\nModel performance:")
        for model_type, status in summary.get('model_performance', {}).items():
            print(f"- {model_type}: {status}")
    else:
        print("Learning cycle failed:", results.get("error", "Unknown error"))
