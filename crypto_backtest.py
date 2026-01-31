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
import math

# --- 1. Load Environment Variables ---
load_dotenv()

# Apply nest_asyncio
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"  # Define the crypto symbol
HISTORICAL_DAYS = 7  # Define the lookback period
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
data.index.name = 'timestamp'  # Explicitly name the index 'timestamp'

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
            # print(f"Handled missing values in column '{col}' using {method}.") # Commented out for cleaner output
        # else:
            # print(f"No missing values in column '{col}'.") # Commented out for cleaner output
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
        # print(f"Handled outliers in column '{col}' using IQR method.") # Commented out for cleaner output
    return df

def calculate_sma(data, period):
    """Calculates Simple Moving Average (SMA)."""
    return data.rolling(window=period).mean()

def calculate_ema(data, series, period):
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(close, period):
    """Calculates Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    # Handle division by zero for rs
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN

def calculate_atr(data, period=14):
    """Calculates Average True Range (ATR)."""
    high = data['high']
    low = data['low']
    close = data['close']

    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(com=period - 1, adjust=False).mean()
    return atr

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
    if historical_data.empty:
        print(f"No historical data fetched for {symbol} from {start_date} to {end_date}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    historical_data = historical_data.droplevel(0)  # remove multi-index

    print(f"Fetched {len(historical_data)} historical bars for {symbol} from {start_date} to {end_date}")

    return historical_data

# --- 6. Strategy Classes ---

class ScalpingStrategy:
    """
    A trading strategy that uses RSI, Stochastic Oscillator, and Moving Average
    to generate buy/sell signals.
    """

    def __init__(self, data: pd.DataFrame, rsi_period: int = 14, stochastic_period: int = 14, ma_period: int = 10):
        """
        Initializes the ScalpingStrategy with market data and indicator parameters.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'open', 'high', 'low', and 'close' columns.")

        self.data = data.copy()  # Ensure working on a copy
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.ma_period = ma_period

        # Calculate Indicators
        self.data['RSI'] = calculate_rsi(self.data['close'], self.rsi_period)
        self.data['Stochastic_K'], self.data['Stochastic_D'] = self._calculate_stochastic(self.stochastic_period)  # Calls stochastic calc
        self.data['SMA'] = calculate_sma(self.data['close'], self.ma_period)
        self.data['ATR'] = calculate_atr(self.data)
        self.data.dropna(inplace=True)  # Drop NaNs introduced by indicator calculations

    def _calculate_stochastic(self, period: int) -> tuple[pd.Series, pd.Series]:
        """Calculates Stochastic Oscillator (%K and %D)."""
        lowest_low = self.data['low'].rolling(window=period).min()
        highest_high = self.data['high'].rolling(window=period).max()

        # Handle division by zero for %K
        range_high_low = highest_high - lowest_low
        k_percent = ((self.data['close'] - lowest_low) / range_high_low) * 100
        k_percent = k_percent.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN

        # %D is typically a 3-period simple moving average of %K
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent

    def generate_signal(self, current_index):
        """Generates trading signal (Buy, Sell, Hold)."""
        if current_index >= len(self.data):
            return "HOLD"  # Index out of bounds

        # Ensure indicators are available
        required_indicators = ['RSI', 'Stochastic_K', 'SMA']
        for indicator in required_indicators:
            if indicator not in self.data.columns or pd.isna(self.data[indicator].iloc[current_index]):
                return "HOLD"

        # Implement your scalping strategy logic here using self.data and calculated indicators
        # Example:
        if self.data['RSI'].iloc[current_index] > 70 and self.data['Stochastic_K'].iloc[current_index] > 80:
            return "SELL"
        elif self.data['RSI'].iloc[current_index] < 30 and self.data['Stochastic_K'].iloc[current_index] < 20:
            return "BUY"
        else:
            return "HOLD"

class MomentumTradingStrategy:
    def __init__(self, data, long_ma_period=50, short_ma_period=20, macd_fast_period=12, macd_slow_period=26, macd_signal_period=9, roc_period=12):
        self.data = data.copy() # Ensure working on a copy
        self.long_ma_period = long_ma_period
        self.short_ma_period = short_ma_period
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.roc_period = roc_period

        # Calculate Indicators
        self.data['Long_SMA'] = calculate_sma(self.data['close'], self.long_ma_period)
        self.data['Short_SMA'] = calculate_sma(self.data['close'], self.short_ma_period)
        macd_line, signal_line, histogram = calculate_macd(self.data)
        self.data['MACD'] = macd_line
        self.data['Signal'] = signal_line
        self.data['ROC'] = self.calculate_roc(self.roc_period)
        self.data.dropna(inplace=True) # Drop NaNs introduced by indicator calculations

    def calculate_roc(self, period):
        """Calculates Rate of Change (ROC)."""
        delta = self.data['close'].diff(period)
        roc = delta / self.data['close'].shift(period) * 100
        return roc

    def generate_signal(self, current_index):
        """Generates trading signal."""
        if current_index >= len(self.data):
            return "HOLD" # Index out of bounds

        # Ensure indicators are available
        required_indicators = ['MACD', 'Signal', 'ROC']
        for indicator in required_indicators:
            if indicator not in self.data.columns or pd.isna(self.data[indicator].iloc[current_index]):
                return "HOLD"

        # Implement your momentum trading strategy logic here
        if self.data['MACD'].iloc[current_index] > self.data['Signal'].iloc[current_index] and self.data['ROC'].iloc[current_index] > 0:
            return "BUY"
        elif self.data['MACD'].iloc[current_index] < self.data['Signal'].iloc[current_index] and self.data['ROC'].iloc[current_index] < 0:
            return "SELL"
        else:
            return "HOLD"

class ReversalTradingStrategy:
    """
    A trading strategy that identifies potential price reversals using a combination of
    RSI, Stochastic Oscillator, and Bollinger Bands.
    """

    def __init__(self, data: pd.DataFrame,
                 rsi_period: int = 14,
                 stochastic_period: int = 14,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 stochastic_oversold: int = 20,
                 stochastic_overbought: int = 80):
        """
        Initializes the ReversalTradingStrategy with market data and indicator parameters.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'open', 'high', 'low', and 'close' columns.")

        self.data = data.copy()  # Work on a copy to avoid modifying original DataFrame directly
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stochastic_oversold = stochastic_oversold
        self.stochastic_overbought = stochastic_overbought

        self._calculate_all_indicators()
        self.data.dropna(inplace=True) # Drop NaNs introduced by indicator calculations


    def _calculate_all_indicators(self):
        """Calculates all necessary technical indicators and adds them to the DataFrame."""
        # print("Calculating technical indicators...") # Commented out for cleaner output
        self.data['RSI'] = calculate_rsi(self.data['close'], self.rsi_period)
        self.data['Stochastic_K'], self.data['Stochastic_D'] = self._calculate_stochastic(self.stochastic_period)

        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            self.data, period=self.bb_period, num_std=self.bb_std
        )
        self.data['Upper_BB'] = upper_band
        self.data['Middle_BB'] = middle_band
        self.data['Lower_BB'] = lower_band
        # print("Indicator calculation complete.") # Commented out for cleaner output

    def _calculate_stochastic(self, period: int) -> tuple[pd.Series, pd.Series]:
        """Calculates Stochastic Oscillator (%K and %D)."""
        lowest_low = self.data['low'].rolling(window=period).min()
        highest_high = self.data['high'].rolling(window=period).max()

        # Handle division by zero for %K
        range_high_low = highest_high - lowest_low
        k_percent = ((self.data['close'] - lowest_low) / range_high_low) * 100
        k_percent = k_percent.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN

        # %D is typically a 3-period simple moving average of %K
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent

    def generate_signal(self, current_index: int) -> str:
        """
        Generates a trading signal (BUY, SELL, or HOLD) based on the reversal strategy logic.
        """
        if current_index >= len(self.data):
            return "HOLD" # Index out of bounds

        required_indicators = ['RSI', 'Upper_BB', 'Lower_BB', 'Stochastic_K']

        # Ensure all required indicator values are available at the current_index
        for indicator in required_indicators:
            if indicator not in self.data.columns or pd.isna(self.data.iloc[current_index][indicator]): #Use iloc for integer-based indexing
                # print(f"Insufficient data at index {current_index} for indicator '{indicator}'. Returning HOLD.") # Commented out for cleaner output
                return "HOLD"

        current_rsi = self.data['RSI'].iloc[current_index] #Use iloc for integer-based indexing
        current_close = self.data['close'].iloc[current_index] #Use iloc for integer-based indexing
        current_upper_bb = self.data['Upper_BB'].iloc[current_index] #Use iloc for integer-based indexing
        current_lower_bb = self.data['Lower_BB'].iloc[current_index] #Use iloc for integer-based indexing
        current_stochastic_k = self.data['Stochastic_K'].iloc[current_index] #Use iloc for integer-based indexing

        # Bearish Reversal (Sell Signal)
        if (current_rsi > self.rsi_overbought and
            current_close > current_upper_bb and
            current_stochastic_k > self.stochastic_overbought):
            # print(f"SELL signal at index {current_index}: RSI={current_rsi:.2f}, Close={current_close:.2f}, UpperBB={current_upper_bb:.2f}, StochasticK={current_stochastic_k:.2f}") # Commented out for cleaner output
            return "SELL"
        # Bullish Reversal (Buy Signal)
        elif (current_rsi < self.rsi_oversold and
              current_close < current_lower_bb and
              current_stochastic_k < self.stochastic_oversold):
            # print(f"BUY signal at index {current_index}: RSI={current_rsi:.2f}, Close={current_close:.2f}, LowerBB={current_lower_bb:.2f}, StochasticK={current_stochastic_k:.2f}") # Commented out for cleaner output
            return "BUY"
        else:
            # print(f"HOLD signal at index {current_index}.") # Commented out for cleaner output
            return "HOLD"

    def generate_signals_for_dataframe(self) -> pd.Series:
        """
        Generates trading signals for the entire DataFrame.
        """
        signals = pd.Series("HOLD", index=self.data.index, dtype=str)
        for i in range(len(self.data)):
            signals.iloc[i] = self.generate_signal(i)
        return signals

class BreakoutTradingStrategy:
    def __init__(self, data: pd.DataFrame, sr_period: int = 20, atr_period: int = 14,
                 atr_multiplier_sl: float = 1.5, atr_multiplier_tp: float = 3.0,
                 confirmation_bars: int = 1):
        """
        Initializes the BreakoutTradingStrategy with historical data and parameters.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
        if len(data) < max(sr_period, atr_period):
            raise ValueError("Insufficient data for the specified periods. Provide more data points.")

        self.data = data.copy()  # Work on a copy to avoid modifying original data
        self.sr_period = sr_period
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl  # Multiplier for Stop Loss
        self.atr_multiplier_tp = atr_multiplier_tp  # Multiplier for Take Profit
        self.confirmation_bars = confirmation_bars  # Number of bars to confirm a breakout

        self._calculate_indicators()
        self.data.dropna(inplace=True) # Drop NaNs introduced by indicator calculations

        # Define attributes for the Backtester class
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp


    def _calculate_indicators(self):
        """Calculates all necessary indicators for the strategy."""
        # Calculate Resistance (Highest high over the period)
        self.data['Resistance'] = self.data['high'].rolling(window=self.sr_period).max()
        # Calculate Support (Lowest low over the period)
        self.data['Support'] = self.data['low'].rolling(window=self.sr_period).min()

        # Calculate ATR
        self.data['TR'] = self._calculate_true_range(self.data)
        self.data['ATR'] = self.data['TR'].rolling(window=self.atr_period).mean()


    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the True Range for ATR."""
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    def generate_signal(self, current_index: int) -> str:
        """
        Generates trading signal (BUY, SELL, or HOLD).
        Prints entry, stop-loss, and take-profit levels for informational purposes.
        """
        if current_index < max(self.sr_period, self.atr_period, self.confirmation_bars) or current_index >= len(self.data):
            return "HOLD"

        current_close = self.data['close'].iloc[current_index]
        current_resistance = self.data['Resistance'].iloc[current_index]
        current_support = self.data['Support'].iloc[current_index]
        current_atr = self.data['ATR'].iloc[current_index]

        signal = "HOLD"
        entry_price = None
        stop_loss = None
        take_profit = None

        # Check for Buy Signal (Breakout above Resistance)
        if current_close > current_resistance:
            confirmed_breakout = True
            for i in range(1, self.confirmation_bars + 1):
                if current_index - i < 0 or self.data['close'].iloc[current_index - i] <= self.data['Resistance'].iloc[current_index - i]:
                    confirmed_breakout = False
                    break

            if confirmed_breakout:
                signal = "BUY"
                entry_price = current_close
                stop_loss = current_close - (current_atr * self.atr_multiplier_sl)
                take_profit = current_close + (current_atr * self.atr_multiplier_tp)
                # print(f"BUY signal at index {current_index}: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}") # Commented out for cleaner output

        # Check for Sell Signal (Breakout below Support)
        elif current_close < current_support:
            confirmed_breakout = True
            for i in range(1, self.confirmation_bars + 1):
                if current_index - i < 0 or self.data['close'].iloc[current_index - i] >= self.data['Support'].iloc[current_index - i]:
                    confirmed_breakout = False
                    break

            if confirmed_breakout:
                signal = "SELL"
                entry_price = current_close
                stop_loss = current_close + (current_atr * self.atr_multiplier_sl)
                take_profit = current_close - (current_atr * self.atr_multiplier_tp)
                # print(f"SELL signal at index {current_index}: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}") # Commented out for cleaner output

        return signal

class RangeTradingStrategy:
    def __init__(self, data: pd.DataFrame, support_resistance_period: int = 20,
                 breakout_threshold_multiplier: float = 0.005,
                 min_range_width_multiplier: float = 0.01):
        """
        Initializes the RangeTradingStrategy with historical data and parameters.
        """
        if not isinstance(data, pd.DataFrame) or not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Data must be a pandas DataFrame with 'high', 'low', and 'close' columns.")
        if support_resistance_period <= 0:
            raise ValueError("support_resistance_period must be a positive integer.")
        if not (0 <= breakout_threshold_multiplier < 1):
            raise ValueError("breakout_threshold_multiplier must be between 0 and 1.")
        if not (0 <= min_range_width_multiplier < 1):
            raise ValueError("min_range_width_multiplier must be between 0 and 1.")

        self.data = data.copy()  # Work on a copy to avoid modifying original DataFrame
        self.support_resistance_period = support_resistance_period
        self.breakout_threshold_multiplier = breakout_threshold_multiplier
        self.min_range_width_multiplier = min_range_width_multiplier

        self._calculate_indicators()
        self.data.dropna(inplace=True) # Drop NaNs introduced by indicator calculations


    def _calculate_indicators(self):
        """
        Calculates Support, Resistance, and other necessary indicators.
        Uses a private method convention as it's intended for internal use during initialization.
        """
        # Using .copy() to avoid SettingWithCopyWarning if original data frame is a slice
        self.data['Resistance'] = self.data['high'].rolling(window=self.support_resistance_period).max()
        self.data['Support'] = self.data['low'].rolling(window=self.support_resistance_period).min()

        # Calculate Range Width
        self.data['Range_Width'] = self.data['Resistance'] - self.data['Support']

        # Calculate Breakout Threshold
        # This threshold helps filter out false breakouts.
        self.data['Breakout_Threshold_Buy'] = self.data['Support'] * (1 - self.breakout_threshold_multiplier)
        self.data['Breakout_Threshold_Sell'] = self.data['Resistance'] * (1 + self.breakout_threshold_multiplier)

    def generate_signal(self, current_index: int) -> str:
        """
        Generates a trading signal (BUY, SELL, or HOLD) for a given index.
        """
        if current_index < self.support_resistance_period - 1 or current_index >= len(self.data):
            return "HOLD"  # Not enough data to calculate S/R or index out of bounds

        current_close = self.data['close'].iloc[current_index]
        current_support = self.data['Support'].iloc[current_index]
        current_resistance = self.data['Resistance'].iloc[current_index]
        current_range_width = self.data['Range_Width'].iloc[current_index]
        # current_breakout_threshold_buy = self.data['Breakout_Threshold_Buy'].iloc[current_index] # Not used in current logic
        # current_breakout_threshold_sell = self.data['Breakout_Threshold_Sell'].iloc[current_index] # Not used in current logic

        # Handle NaN values that can occur at the beginning of the data due to rolling window
        if pd.isna(current_support) or pd.isna(current_resistance):
            return "HOLD"

        # Avoid trading in very narrow ranges (e.g., during low volatility)
        if current_range_width < self.data['close'].iloc[current_index] * self.min_range_width_multiplier:
            return "HOLD"

        # Range Trading Logic with Breakout Thresholds
        if current_close <= current_support:  # Price is at or below support
            return "BUY"
        elif current_close >= current_resistance:  # Price is at or above resistance
            return "SELL"
        else:
            return "HOLD"

class GapTradingStrategy:
    """
    A trading strategy that generates buy/sell/hold signals based on the size
    of the price gap between the previous day's close and the current day's open.
    """

    def __init__(self, data: pd.DataFrame, positive_gap_threshold: float = 0.02, negative_gap_threshold: float = -0.02):
        """
        Initializes the GapTradingStrategy with historical price data and gap thresholds.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if 'open' not in data.columns or 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'open' and 'close' columns.")
        if not all(isinstance(val,