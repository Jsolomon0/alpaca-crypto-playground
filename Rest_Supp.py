import os
import asyncio
import nest_asyncio
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import matplotlib.pyplot as plt

# --- 1. Load Environment Variables ---
load_dotenv()

# Apply nest_asyncio
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"  # Define the crypto symbol
HISTORICAL_DAYS = 365  # Define the lookback period

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY not found in environment variables.")
    exit(1)

# Initialize the CryptoHistoricalData Client
crypto_client = CryptoHistoricalDataClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY
)


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
    if historical_data.empty:
        print(f"No historical data fetched for {symbol} from {start_date} to {end_date}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    historical_data = historical_data.droplevel(0)  # remove multi-index

    print(f"Fetched {len(historical_data)} historical bars for {symbol} from {start_date} to {end_date}")

    return historical_data

# --- 6. Strategy Class (Simplified) ---

class SupportResistance:
    def __init__(self, data: pd.DataFrame, sr_period: int = 1440):
        """
        Initializes the SupportResistance strategy.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
        if len(data) < sr_period:
            raise ValueError("Insufficient data for the specified period. Provide more data points.")

        self.data = data.copy()  # Work on a copy
        self.sr_period = sr_period
        self._calculate_levels()

    def _calculate_levels(self):
        """Calculates support and resistance levels."""
        self.data['Resistance'] = self.data['high'].rolling(window=self.sr_period).max()
        self.data['Support'] = self.data['low'].rolling(window=self.sr_period).min()
        self.data.dropna(inplace=True)

    def plot_levels(self):
        """Plots the closing prices, support, and resistance levels."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['close'], label='Closing Price', alpha=0.7)
        plt.plot(self.data['Resistance'], label='Resistance', color='red', linestyle='--')
        plt.plot(self.data['Support'], label='Support', color='green', linestyle='--')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title('Support and Resistance Levels')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# --- 9. Main Execution Logic ---

async def main():

    # Fetch historical data
    initial_data = await fetch_historical_data(SYMBOL, HISTORICAL_DAYS)

    if initial_data.empty:
        print("No historical data available. Exiting.")
        return

    # Create and plot support/resistance levels
    sr_strategy = SupportResistance(initial_data)
    sr_strategy.plot_levels()


if __name__ == '__main__':
    # Run the main asynchronous function
    asyncio.run(main())