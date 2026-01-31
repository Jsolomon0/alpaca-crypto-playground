import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from alpaca.data.live import CryptoDataStream
from alpaca.data.models import Bar, Quote, Trade, Orderbook

# --- 1. Load Environment Variables ---
# Make sure you have a .env file in the same directory as this script
# with your API_KEY and SECRET_KEY:
# ALPACA_API_KEY_ID=YOUR_API_KEY_ID
# ALPACA_API_SECRET_KEY=YOUR_SECRET_KEY
load_dotenv()

# Apply nest_asyncio for compatibility in environments like VS Code's interactive terminal
# This helps prevent RuntimeError: asyncio.run() cannot be called from a running event loop
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY not found in environment variables.")
    print("Please set them in your .env file or as system environment variables.")
    exit(1) # Exit the script if keys are missing

# Initialize the CryptoDataStream client
# feed=CryptoFeed.US is the default, but explicitly showing it here.
crypto_stream = CryptoDataStream(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    raw_data=False # We want parsed Python objects, not raw dictionaries
)

# --- 3. Define Async Event Handlers (Callbacks) ---

async def on_trade_update(trade: Trade):
    """Callback for trade data."""
    print(f"TRADE: {trade.symbol} - Price: {trade.price} - Size: {trade.size} - Timestamp: {trade.timestamp}")

async def on_quote_update(quote: Quote):
    """Callback for quote data (bid/ask)."""
    print(f"QUOTE: {quote.symbol} - Bid: {quote.bid_price} ({quote.bid_size}) - Ask: {quote.ask_price} ({quote.ask_size}) - Timestamp: {quote.timestamp}")

async def on_minute_bar_update(bar: Bar):
    """Callback for minute bar data (OHLCV)."""
    print(f"MIN_BAR: {bar.symbol} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume} - Timestamp: {bar.timestamp}")

async def on_daily_bar_update(bar: Bar):
    """Callback for daily bar data (OHLCV)."""
    print(f"DAILY_BAR: {bar.symbol} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume} - Timestamp: {bar.timestamp}")

async def on_updated_bar_update(bar: Bar):
    """Callback for updated minute bar data (revisions to closed bars)."""
    print(f"UPDATED_BAR: {bar.symbol} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume} - Timestamp: {bar.timestamp}")

async def on_orderbook_update(orderbook: Orderbook):
    """Callback for order book data (Level 2)."""
    # Orderbook data can be very verbose, only printing top few levels for brevity
    print(f"ORDERBOOK: {orderbook.symbol} - Bids: {[(b.price, b.size) for b in orderbook.bids[:2]]} - Asks: {[(a.price, a.size) for a in orderbook.asks[:2]]} - Timestamp: {orderbook.timestamp}")


# --- 4. Main Function to Subscribe and Run ---

async def main():
    """Main asynchronous function to set up subscriptions and run the stream."""
    print("--- Subscribing to Crypto Data Streams ---")

    # Subscribe to different data types for various symbols
    # Note: Use valid crypto symbols supported by Alpaca (e.g., BTC/USD, ETH/USD, SOL/USD, LTC/USD)
    crypto_stream.subscribe_trades(on_trade_update, "BTC/USD", "ETH/USD")
    print("Subscribed to Trades: BTC/USD, ETH/USD")

    crypto_stream.subscribe_quotes(on_quote_update, "BTC/USD")
    print("Subscribed to Quotes: BTC/USD")

    crypto_stream.subscribe_bars(on_minute_bar_update, "SOL/USD")
    print("Subscribed to Minute Bars: SOL/USD")

    # You can subscribe to ALL symbols for a data type using "*"
    # crypto_stream.subscribe_quotes(on_quote_update, "*")
    # print("Subscribed to Quotes for ALL symbols")

    # Example of other subscriptions based on the provided documentation
    # crypto_stream.subscribe_daily_bars(on_daily_bar_update, "LTC/USD")
    # print("Subscribed to Daily Bars: LTC/USD")

    # crypto_stream.subscribe_updated_bars(on_updated_bar_update, "ETH/USD")
    # print("Subscribed to Updated Bars: ETH/USD")

    # crypto_stream.subscribe_orderbooks(on_orderbook_update, "DOGE/USD")
    # print("Subscribed to Orderbooks: DOGE/USD")


    print("\n--- Starting Crypto Data Stream ---")
    print("Press Ctrl+C to stop the stream gracefully.")

    try:
        # Run the WebSocket connection indefinitely.
        # This will block until the stream is explicitly stopped or an error occurs.
        await crypto_stream.run()
    except KeyboardInterrupt:
        print("\n--- KeyboardInterrupt detected. Stopping stream... ---")
    finally:
        # Ensure the connection is closed cleanly on exit
        await crypto_stream.close()
        print("--- Crypto Data Stream stopped and closed. ---")

# --- 5. Run the Main Function ---
if __name__ == "__main__":
    asyncio.run(main())