
import os
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.live import CryptoDataStream
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.data.models import Bar, Quote, Trade, Orderbook

# --- 1. Load Environment Variables ---
load_dotenv()

# Apply nest_asyncio
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"  # Define the crypto symbol
HISTORICAL_DAYS = 7 # Define the lookback period
if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY not found in environment variables.")
    exit(1)

# Initialize the CryptoDataStream and HistoricalData Client
crypto_stream = CryptoDataStream(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    raw_data=False
)

crypto_client = CryptoHistoricalDataClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY
)
# --- 3. Global DataFrame to Store Data ---
data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
data.index.name = 'timestamp' # Explicitly name the index 'timestamp'

# --- 4. Data Processing Functions ---

def handle_missing_values(df, columns, method='ffill'):
    """Handles missing values in specified columns."""
    for col in columns:
        if df[col].isnull().any():
            if method == 'ffill':
                df[col] = df[col].ffill()
            elif method == 'bfill':
                df[col] = df[col].bfill()
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

# --- 5. Fetch Historical Data Function ---

async def fetch_historical_data(symbol, days):
    """Fetches historical minute bar data for the given symbol and number of days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )

    bars = crypto_client.get_crypto_bars(request)

    historical_data = bars.df
    historical_data = historical_data.droplevel(0) #remove multi-index

    print(f"Fetched {len(historical_data)} historical bars for {symbol} from {start_date} to {end_date}")
    print(historical_data)

    return historical_data

# --- 6. Event Handlers (Modified to Process Data) ---

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
    data = pd.concat([data, new_data])
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
    print(data_processed.tail(1))

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

# --- 7. Main Function to Subscribe and Run ---

async def main():
    """Main function to set up subscriptions and run the stream."""
    global data

    print("--- Fetching Historical Data ---")
    historical_data = await fetch_historical_data(SYMBOL, HISTORICAL_DAYS) # Fetch historical data
    data = pd.concat([data, historical_data]) # Concatenate historical data to the DataFrame
    data = data[~data.index.duplicated(keep='last')] # De-duplicate again after combining

    print("--- Preprocessing Historical Data ---")
    data = handle_missing_values(data, ['open', 'high', 'low', 'close', 'volume']) # Preprocess historical data
    data = handle_outliers_iqr(data, ['open', 'high', 'low', 'close', 'volume'])

    print("--- Subscribing to Crypto Data Streams ---")

    # Subscribe to different data types for various symbols
    crypto_stream.subscribe_bars(on_minute_bar_update, SYMBOL)
    print(f"Subscribed to Minute Bars: {SYMBOL}")

    # Optionally subscribe to other streams as needed
    #crypto_stream.subscribe_trades(on_trade_update, "BTC/USD", "ETH/USD")
    #print("Subscribed to Trades: BTC/USD, ETH/USD")

    print("\n--- Starting Crypto Data Stream ---")
    print("Press Ctrl+C to stop the stream gracefully.")

    try:
        await crypto_stream.run()
    except KeyboardInterrupt:
        print("\n--- KeyboardInterrupt detected. Stopping stream... ---")
    finally:
        await crypto_stream.close()
        print("--- Crypto Data Stream stopped and closed. ---")

# --- 8. Run the Main Function ---
if __name__ == "__main__":
    asyncio.run(main())