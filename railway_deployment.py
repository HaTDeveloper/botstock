#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Saudi Stock Bot - Railway Deployment Integration
-----------------------------------------------
This module integrates the Saudi Stock Analysis Bot with the bot_learning module
and prepares it for deployment on Railway platform.
"""

import os
import sys
import logging
from datetime import datetime
import threading
import time

# Import bot modules
from enhanced_saudi_stock_bot import SaudiStockBot
from bot_learning import BotLearning, integrate_with_main_bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("railway_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RailwayDeployment")

# Get Discord webhook URL from environment variable
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')

def initialize_data_directories():
    """Initialize necessary data directories."""
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    logger.info("Data directories initialized")

def run_bot_with_learning():
    """Run the Saudi Stock Bot with learning capabilities."""
    try:
        logger.info("Starting Saudi Stock Analysis Bot with learning capabilities")
        
        # Initialize bot
        bot = SaudiStockBot()
        if DISCORD_WEBHOOK_URL:
            bot.discord_webhook_url = DISCORD_WEBHOOK_URL
            logger.info("Using Discord webhook URL from environment variable")
        
        # Initialize learning module
        bot_learning = BotLearning()
        logger.info("Bot learning module initialized")
        
        # Initial analysis and report
        logger.info("Performing initial analysis...")
        analysis_results = bot.analyze_stocks(analysis_type="both", max_stocks=15)
        
        # Enhance results with learning
        enhanced_results = integrate_with_main_bot(bot_learning, analysis_results)
        
        # Generate report
        logger.info("Generating initial report...")
        report = bot.generate_report(enhanced_results, lang="ar")
        
        # Save report to file
        report_file = "saudi_stock_analysis_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Initial report saved to {report_file}")
        
        # Send to Discord webhook
        logger.info("Sending initial analysis to Discord webhook...")
        bot.send_to_discord_webhook(enhanced_results, report_type="daily")
        
        # Run learning cycle in a separate thread
        def run_learning_cycle_periodically():
            while True:
                try:
                    logger.info("Running learning cycle...")
                    bot_learning.run_learning_cycle()
                    logger.info("Learning cycle completed, sleeping for 24 hours")
                    # Sleep for 24 hours
                    time.sleep(86400)
                except Exception as e:
                    logger.error(f"Error in learning cycle: {e}")
                    # Sleep for 1 hour before retrying
                    time.sleep(3600)
        
        learning_thread = threading.Thread(target=run_learning_cycle_periodically)
        learning_thread.daemon = True
        learning_thread.start()
        logger.info("Learning cycle thread started")
        
        # Start scheduler in a separate thread
        logger.info("Starting scheduler for continuous operation...")
        scheduler_thread = threading.Thread(target=bot.run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("Bot is now running continuously on Railway")
        
        # Keep the main thread alive
        while True:
            time.sleep(60)
            
    except Exception as e:
        logger.error(f"Error in main bot execution: {e}")
        # Sleep for a while before restarting
        time.sleep(300)
        # Restart the bot
        run_bot_with_learning()

if __name__ == "__main__":
    # Initialize data directories
    initialize_data_directories()
    
    # Run the bot
    run_bot_with_learning()
