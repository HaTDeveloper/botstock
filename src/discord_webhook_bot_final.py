import os
import json
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('saudi_stock_bot')

# Load environment variables
load_dotenv()

class SaudiStockBot:
    def __init__(self):
        """Initialize the Saudi Stock Bot with webhook functionality."""
        self.discord_webhook_url = os.getenv('https://discord.com/api/webhooks/1346789304181194814/Ggi0o1j8pPq-74U8uyuiXNSWs_bN42d9MkpoCGtm2iRit4OZ25myPRMx_YI9kZkOfdlB')
        self.bot_name = "Saudi Stock Analysis Bot"
        self.bot_avatar_url = "https://example.com/bot-avatar.png"  # Replace with actual avatar URL
        logger.info("Saudi Stock Bot initialized")
    
    def send_to_discord_webhook(self, data, report_type="analysis"):
        """Send data to Discord webhook.
        
        Args:
            data (dict): The data to send to Discord
            report_type (str): Type of report (daily, weekly, analysis, alert)
        """
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not set. Skipping webhook notification.")
            return False
        
        try:
            # Format the message based on report type
            if report_type == "daily":
                title = f"📊 Daily Saudi Stock Market Report - {datetime.now().strftime('%Y-%m-%d')}"
                color = 3447003  # Blue
            elif report_type == "weekly":
                title = f"📈 Weekly Saudi Stock Market Analysis - Week {datetime.now().isocalendar()[1]}"
                color = 10181046  # Purple
            elif report_type == "alert":
                title = f"⚠️ Stock Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                color = 15158332  # Red
            else:
                title = f"🔍 Saudi Stock Market Analysis - {datetime.now().strftime('%Y-%m-%d')}"
                color = 3066993  # Green
            
            # Create embed for Discord webhook
            embed = {
                "title": title,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "Saudi Stock Analysis Bot",
                    "icon_url": self.bot_avatar_url
                },
                "fields": []
            }
            
            # Add data fields to embed
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "top_performers" and isinstance(value, list):
                        field_value = "\n".join([f"• {item['symbol']} ({item['name']}): {item['change']}%" for item in value[:5]])
                        embed["fields"].append({
                            "name": "🔝 Top Performers",
                            "value": field_value or "No data available",
                            "inline": False
                        })
                    elif key == "market_summary" and isinstance(value, dict):
                        field_value = "\n".join([f"• {k}: {v}" for k, v in value.items()])
                        embed["fields"].append({
                            "name": "📑 Market Summary",
                            "value": field_value or "No data available",
                            "inline": False
                        })
                    elif key == "recommendations" and isinstance(value, list):
                        field_value = "\n".join([f"• {item}" for item in value[:5]])
                        embed["fields"].append({
                            "name": "💡 Recommendations",
                            "value": field_value or "No recommendations available",
                            "inline": False
                        })
                    elif key not in ["top_performers", "market_summary", "recommendations"]:
                        embed["fields"].append({
                            "name": key.replace("_", " ").title(),
                            "value": str(value) or "No data available",
                            "inline": True
                        })
            else:
                # If data is not a dict, convert it to string and add as a single field
                embed["fields"].append({
                    "name": "Analysis Results",
                    "value": str(data) or "No data available",
                    "inline": False
                })
            
            # Prepare webhook payload
            webhook_data = {
                "username": self.bot_name,
                "avatar_url": self.bot_avatar_url,
                "embeds": [embed]
            }
            
            # Send to Discord webhook
            response = requests.post(
                self.discord_webhook_url,
                data=json.dumps(webhook_data),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 204:
                logger.info(f"Successfully sent {report_type} report to Discord webhook")
                return True
            else:
                logger.error(f"Failed to send to Discord webhook. Status code: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending to Discord webhook: {str(e)}")
            return False
    
    def send_alert(self, symbol, price, change_percent, message):
        """Send a stock alert to Discord webhook.
        
        Args:
            symbol (str): Stock symbol
            price (float): Current price
            change_percent (float): Percentage change
            message (str): Alert message
        """
        alert_data = {
            "symbol": symbol,
            "price": price,
            "change_percent": f"{change_percent}%",
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.send_to_discord_webhook(alert_data, report_type="alert")

# Main function for testing
if __name__ == "__main__":
    # For testing webhook functionality
    bot = SaudiStockBot()
    
    # Test data
    test_data = {
        "market_summary": {
            "TASI Index": "12,345.67 (+0.5%)",
            "Trading Volume": "1.2B shares",
            "Market Cap": "2.5T SAR"
        },
        "top_performers": [
            {"symbol": "2222.SR", "name": "Aramco", "change": 2.5},
            {"symbol": "1111.SR", "name": "Al Rajhi Bank", "change": 1.8},
            {"symbol": "3333.SR", "name": "SABIC", "change": 1.5}
        ],
        "recommendations": [
            "Consider adding Aramco (2222.SR) to your portfolio",
            "Monitor banking sector for potential opportunities",
            "Diversify investments across multiple sectors"
        ]
    }
    
    # Send test data to webhook
    bot.send_to_discord_webhook(test_data, report_type="daily")
