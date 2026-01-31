import os
from alpaca.data.live import CryptoDataStream
import asyncio
import nest_asyncio
from dotenv import load_dotenv # Import for loading .env file

# Load environment variables from .env file
load_dotenv()

# Apply nest_asyncio (often needed for running asyncio in VS Code's interactive terminal or when debugging)
nest_asyncio.apply()

# --- Configuration (now securely loaded from environment variables) ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY not found in environment variables.")
    print("Please set them in your .env file or as system environment variables.")
    exit()

# Initialize the CryptoDataStream client
crypto_stream = CryptoDataStream(
    API_KEY,
    SECRET_KEY,
)

# --- Event Handlers ---
from alpaca.data.models import Bar, Quote, Trade, Orderbook

async def on_trade_update(trade: Trade):
    print(f"Trade: {trade.symbol} - Price: {trade.price} - Size: {trade.size} - Timestamp: {trade.timestamp}")

async def on_quote_update(quote: Quote):
    print(f"Quote: {quote.symbol} - Bid: {quote.bid_price} ({quote.bid_size}) - Ask: {quote.ask_price} ({quote.ask_size}) - Timestamp: {quote.timestamp}")

async def on_bar_update(bar: Bar):
    print(f"Bar: {bar.symbol} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume} - Timestamp: {bar.timestamp}")

async def on_orderbook_update(orderbook: Orderbook):
    print(f"Orderbook: {orderbook.symbol} - Bids: {orderbook.bids[:2]} - Asks: {orderbook.asks[:2]} - Timestamp: {orderbook.timestamp}") # Print top 2 levels

# --- Main Subscription Logic ---
async def main():
    # Subscribe to trades for BTC/USD and ETH/USD
    crypto_stream.subscribe_trades(on_trade_update, "BTC/USD", "ETH/USD")

    # Subscribe to quotes for BTC/USD
    crypto_stream.subscribe_quotes(on_quote_update, "BTC/USD")

    # Subscribe to 1-minute bars for SOL/USD
    crypto_stream.subscribe_bars(on_bar_update, "SOL/USD")

    # Subscribe to orderbooks for LTC/USD (Level 2 data)
    crypto_stream.subscribe_orderbooks(on_orderbook_update, "LTC/USD")

    print("Subscribed to data streams. Press Ctrl+C to stop.")

    # Run the WebSocket connection indefinitely
    await crypto_stream.run()

if __name__ == "__main__":
    asyncio.run(main())
