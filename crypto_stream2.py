import os
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.live import CryptoDataStream
from alpaca.data.models import Bar, Quote, Trade, Orderbook

# --- 1. Load Environment Variables ---
load_dotenv()

# Apply nest_asyncio
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY not found in environment variables.")
    exit(1)

# Initialize the CryptoDataStream client
crypto_stream = CryptoDataStream(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    raw_data=False
)

# --- 3. Global DataFrame to Store Data ---
data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
data.set_index('timestamp', inplace=True) #set index for time series analysis

# --- 4. Data Processing Functions ---

def handle_missing_values(df, columns, method='ffill'): #Forward fill is most appropriate for time series data.
    """Handles missing values in specified columns."""
    for col in columns:
        if df[col].isnull().any():
            if method == 'ffill':
                df[col] = df[col].ffill()
            elif method == 'bfill':
                df[col] = df[col].bfill() #back fill in case first data point is missing.
            elif method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'drop':
                df = df.dropna(subset=[col])
            else:
                raise ValueError(f"Invalid imputation method: {method}")
            print(f"Handled missing values in column '{col}' using {method}.")
        else:
            print(f"No missing values in column '{col}'.")
    return df

def handle_outliers_iqr(df, columns, multiplier=1.5):
    """Handles outliers using the IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
        print(f"Handled outliers in column '{col}' using IQR method.")
    return df

def calculate_sma(data, period):
    """Calculates Simple Moving Average (SMA)."""
    return data.rolling(window=period).mean()

def calculate_ema(data, series, period):
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.rolling(period).mean()
    roll_down1 = np.abs(down.rolling(period).mean())
    RS = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    ema_fast = calculate_ema(data, data['close'], fast_period)
    ema_slow = calculate_ema(data, data['close'], slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(data, macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculates Bollinger Bands."""
    middle_band = calculate_sma(data['close'], period)
    std_dev = data['close'].rolling(window=period).std()
    upper_band = middle_band + num_std * std_dev
    lower_band = middle_band - num_std * std_dev
    return upper_band, middle_band, lower_band

# --- 5. Event Handlers (Modified to Process Data) ---

async def on_minute_bar_update(bar: Bar):
    """Callback for minute bar data (OHLCV)."""
    global data

    new_data = pd.DataFrame([{
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume,
        'timestamp': bar.timestamp
    }])

    new_data.set_index('timestamp', inplace=True)
    data = pd.concat([data, new_data]) #Append to the DataFrame
    data = data[~data.index.duplicated(keep='last')] #ensure only one entry per timestamp.

    # Preprocessing (Apply to last n rows to handle streaming data)
    data_processed = data.copy()
    data_processed = handle_missing_values(data_processed, ['open', 'high', 'low', 'close', 'volume'])
    data_processed = handle_outliers_iqr(data_processed, ['open', 'high', 'low', 'close', 'volume'])

    # Feature Engineering (Apply to last n rows)
    data_processed['SMA_20'] = calculate_sma(data_processed['close'], period=20)
    data_processed['RSI_14'] = calculate_rsi(data_processed['close'], period=14)
    macd_line, signal_line, histogram = calculate_macd(data_processed)
    data_processed['MACD'] = macd_line
    data_processed['Signal'] = signal_line
    data_processed['Histogram'] = histogram
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data_processed)
    data_processed['Upper_Band'] = upper_band
    data_processed['Middle_Band'] = middle_band
    data_processed['Lower_Band'] = lower_band

    # Print the last row with calculated features for testing.
    print("\n---Last Data Point with Features---")
    print(data_processed.tail(1)) #Tail(1) prints the last row.

async def on_trade_update(trade: Trade):
    """Callback for trade data."""
    print(f"TRADE: {trade.symbol} - Price: {trade.price} - Size: {trade.size} - Timestamp: {trade.timestamp}")

async def on_quote_update(quote: Quote):
    """Callback for quote data (bid/ask)."""
    print(f"QUOTE: {quote.symbol} - Bid: {quote.bid_price} ({quote.bid_size}) - Ask: {quote.ask_price} ({quote.ask_size}) - Timestamp: {quote.timestamp}")

async def on_daily_bar_update(bar: Bar):
    """Callback for daily bar data (OHLCV)."""
    print(f"DAILY_BAR: {bar.symbol} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume} - Timestamp: {bar.timestamp}")

async def on_updated_bar_update(bar: Bar):
    """Callback for updated minute bar data (revisions to closed bars)."""
    print(f"UPDATED_BAR: {bar.symbol} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume} - Timestamp: {bar.timestamp}")

async def on_orderbook_update(orderbook: Orderbook):
    """Callback for order book data (Level 2)."""
    print(f"ORDERBOOK: {orderbook.symbol} - Bids: {[(b.price, b.size) for b in orderbook.bids[:2]]} - Asks: {[(a.price, a.size) for a in orderbook.asks[:2]]} - Timestamp: {orderbook.timestamp}")

# --- 6. Main Function to Subscribe and Run ---

async def main():
    """Main function to set up subscriptions and run the stream."""
    print("--- Subscribing to Crypto Data Streams ---")

    # Subscribe to different data types for various symbols
    crypto_stream.subscribe_bars(on_minute_bar_update, "BTC/USD")
    print("Subscribed to Minute Bars: BTC/USD")

    #crypto_stream.subscribe_trades(on_trade_update, "BTC/USD", "ETH/USD")
    #print("Subscribed to Trades: BTC/USD, ETH/USD")

    #crypto_stream.subscribe_quotes(on_quote_update, "BTC/USD")
    #print("Subscribed to Quotes: BTC/USD")

    #crypto_stream.subscribe_daily_bars(on_daily_bar_update, "LTC/USD")
    #print("Subscribed to Daily Bars: LTC/USD")

    #crypto_stream.subscribe_updated_bars(on_updated_bar_update, "ETH/USD")
    #print("Subscribed to Updated Bars: ETH/USD")

    #crypto_stream.subscribe_orderbooks(on_orderbook_update, "DOGE/USD")
    #print("Subscribed to Orderbooks: DOGE/USD")

    print("\n--- Starting Crypto Data Stream ---")
    print("Press Ctrl+C to stop the stream gracefully.")

    try:
        await crypto_stream.run()
    except KeyboardInterrupt:
        print("\n--- KeyboardInterrupt detected. Stopping stream... ---")
    finally:
        await crypto_stream.close()
        print("--- Crypto Data Stream stopped and closed. ---")

# --- 7. Run the Main Function ---
if __name__ == "__main__":
    asyncio.run(main())