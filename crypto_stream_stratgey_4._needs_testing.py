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

def calculate_rsi(close, period):
    """Calculates Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    historical_data = historical_data.droplevel(0)  # remove multi-index

    print(f"Fetched {len(historical_data)} historical bars for {symbol} from {start_date} to {end_date}")

    return historical_data

# --- 6. Strategy Classes ---

class ScalpingStrategy:
    def __init__(self, data, rsi_period=14, stochastic_period=14, ma_period=10):
        self.data = data
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.ma_period = ma_period

        # Calculate Indicators
        self.data['RSI'] = calculate_rsi(self.data['close'], self.rsi_period)
        # self.data['Stochastic'] = self.calculate_stochastic(self.stochastic_period) #Needs full Implementation
        self.data['SMA'] = calculate_sma(self.data['close'], self.ma_period)
        self.data['ATR'] = calculate_atr(self.data)

    def calculate_stochastic(self, period):
        """Calculates Stochastic Oscillator (Placeholder - needs full implementation)."""
        # Implement stochastic calculation here
        # Requires calculating %K and %D
        return pd.Series(index=self.data.index)  # Placeholder

    def generate_signal(self, current_index):
        """Generates trading signal (Buy, Sell, Hold)."""
        # Implement your scalping strategy logic here using self.data and calculated indicators
        # Example:
        if self.data['RSI'][current_index] > 70:
            return "SELL"
        elif self.data['RSI'][current_index] < 30:
            return "BUY"
        else:
            return "HOLD"

class MomentumTradingStrategy:
    def __init__(self, data, long_ma_period=50, short_ma_period=20, macd_fast_period=12, macd_slow_period=26, macd_signal_period=9, roc_period=12):
        self.data = data
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

    def calculate_roc(self, period):
        """Calculates Rate of Change (ROC)."""
        delta = self.data['close'].diff(period)
        roc = delta / self.data['close'].shift(period) * 100
        return roc

    def generate_signal(self, current_index):
        """Generates trading signal."""
        # Implement your momentum trading strategy logic here
        if self.data['MACD'][current_index] > self.data['Signal'][current_index] and self.data['ROC'][current_index] > 0:
            return "BUY"
        elif self.data['MACD'][current_index] < self.data['Signal'][current_index] and self.data['ROC'][current_index] < 0:
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

    def _calculate_all_indicators(self):
        """Calculates all necessary technical indicators and adds them to the DataFrame."""
        print("Calculating technical indicators...")
        self.data['RSI'] = calculate_rsi(self.data['close'], self.rsi_period)
        self.data['Stochastic_K'], self.data['Stochastic_D'] = self._calculate_stochastic(self.stochastic_period)

        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(
            self.data['close'], period=self.bb_period, num_std=self.bb_std
        )
        self.data['Upper_BB'] = upper_band
        self.data['Middle_BB'] = middle_band
        self.data['Lower_BB'] = lower_band
        print("Indicator calculation complete.")

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
        required_indicators = ['RSI', 'Upper_BB', 'Lower_BB', 'Stochastic_K']

        # Ensure all required indicator values are available at the current_index
        for indicator in required_indicators:
            if indicator not in self.data.columns or pd.isna(self.data.iloc[current_index][indicator]): #Use iloc for integer-based indexing
                print(f"Insufficient data at index {current_index} for indicator '{indicator}'. Returning HOLD.")
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
            print(f"SELL signal at index {current_index}: RSI={current_rsi:.2f}, Close={current_close:.2f}, UpperBB={current_upper_bb:.2f}, StochasticK={current_stochastic_k:.2f}")
            return "SELL"
        # Bullish Reversal (Buy Signal)
        elif (current_rsi < self.rsi_oversold and
              current_close < current_lower_bb and
              current_stochastic_k < self.stochastic_oversold):
            print(f"BUY signal at index {current_index}: RSI={current_rsi:.2f}, Close={current_close:.2f}, LowerBB={current_lower_bb:.2f}, StochasticK={current_stochastic_k:.2f}")
            return "BUY"
        else:
            print(f"HOLD signal at index {current_index}.")
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

    def _calculate_indicators(self):
        """Calculates all necessary indicators for the strategy."""
        # Calculate Resistance (Highest high over the period)
        self.data['Resistance'] = self.data['high'].rolling(window=self.sr_period).max()
        # Calculate Support (Lowest low over the period)
        self.data['Support'] = self.data['low'].rolling(window=self.sr_period).min()

        # Calculate ATR
        self.data['TR'] = self._calculate_true_range(self.data)
        self.data['ATR'] = self.data['TR'].rolling(window=self.atr_period).mean()

        # Drop NaN values introduced by rolling calculations
        self.data.dropna(inplace=True)

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the True Range for ATR."""
        high_low = df['high'] - df['low']
        high_prev_close = abs(df['high'] - df['close'].shift(1))
        low_prev_close = abs(df['low'] - df['close'].shift(1))
        return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    def generate_signal(self, current_index: int) -> dict:
        """
        Generates trading signal with entry, stop-loss, and take-profit levels.
        Returns a dictionary with 'signal', 'entry_price', 'stop_loss', and 'take_profit'.
        """
        if current_index < max(self.sr_period, self.atr_period, self.confirmation_bars):
            return {"signal": "HOLD", "entry_price": None, "stop_loss": None, "take_profit": None}

        current_close = self.data['close'].iloc[current_index] #Use iloc for integer-based indexing
        current_high = self.data['high'].iloc[current_index] #Use iloc for integer-based indexing
        current_low = self.data['low'].iloc[current_index] #Use iloc for integer-based indexing
        current_resistance = self.data['Resistance'].iloc[current_index] #Use iloc for integer-based indexing
        current_support = self.data['Support'].iloc[current_index] #Use iloc for integer-based indexing
        current_atr = self.data['ATR'].iloc[current_index] #Use iloc for integer-based indexing

        signal = "HOLD"
        entry_price = None
        stop_loss = None
        take_profit = None

        # Check for Buy Signal (Breakout above Resistance)
        if current_close > current_resistance:
            # Check for confirmation over the last 'confirmation_bars'
            confirmed_breakout = True
            for i in range(1, self.confirmation_bars + 1):
                if self.data['close'].iloc[current_index - i] <= self.data['Resistance'].iloc[current_index - i]: #Use iloc for integer-based indexing
                    confirmed_breakout = False
                    break

            if confirmed_breakout:
                signal = "BUY"
                entry_price = current_close
                stop_loss = current_close - (current_atr * self.atr_multiplier_sl)
                take_profit = current_close + (current_atr * self.atr_multiplier_tp)

        # Check for Sell Signal (Breakout below Support)
        elif current_close < current_support:
            # Check for confirmation over the last 'confirmation_bars'
            confirmed_breakout = True
            for i in range(1, self.confirmation_bars + 1):
                if self.data['close'].iloc[current_index - i] >= self.data['Support'].iloc[current_index - i]: #Use iloc for integer-based indexing
                    confirmed_breakout = False
                    break

            if confirmed_breakout:
                signal = "SELL"
                entry_price = current_close
                stop_loss = current_close + (current_atr * self.atr_multiplier_sl)
                take_profit = current_close - (current_atr * self.atr_multiplier_tp)

        # Ensure stop_loss and take_profit are sensible
        if signal == "BUY" and stop_loss is not None and stop_loss >= entry_price:
            stop_loss = entry_price - (current_atr * 0.5)  # Fallback for extremely low ATR
        if signal == "SELL" and stop_loss is not None and stop_loss <= entry_price:
            stop_loss = entry_price + (current_atr * 0.5)  # Fallback for extremely low ATR

        return {
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

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
        if current_index < self.support_resistance_period - 1:
            return "HOLD"  # Not enough data to calculate S/R

        current_close = self.data['close'].iloc[current_index] #Use iloc for integer-based indexing
        current_support = self.data['Support'].iloc[current_index] #Use iloc for integer-based indexing
        current_resistance = self.data['Resistance'].iloc[current_index] #Use iloc for integer-based indexing
        current_range_width = self.data['Range_Width'].iloc[current_index] #Use iloc for integer-based indexing
        current_breakout_threshold_buy = self.data['Breakout_Threshold_Buy'].iloc[current_index] #Use iloc for integer-based indexing
        current_breakout_threshold_sell = self.data['Breakout_Threshold_Sell'].iloc[current_index] #Use iloc for integer-based indexing

        # Handle NaN values that can occur at the beginning of the data due to rolling window
        if pd.isna(current_support) or pd.isna(current_resistance):
            return "HOLD"

        # Avoid trading in very narrow ranges (e.g., during low volatility)
        if current_range_width < self.data['close'].iloc[current_index] * self.min_range_width_multiplier: #Use iloc for integer-based indexing
            return "HOLD"

        # Range Trading Logic with Breakout Thresholds
        if current_close <= current_support:  # Price is at or below support
            return "BUY"
        elif current_close >= current_resistance:  # Price is at or above resistance
            return "SELL"
        # Optional: Add logic for breakout confirmation if desired, for example:
        # if current_close < current_breakout_threshold_buy:
        #     return "STRONG_BUY"
        # elif current_close > current_breakout_threshold_sell:
        #     return "STRONG_SELL"
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
        if not all(isinstance(val, (int, float)) for col in ['open', 'close'] for val in data[col]):
            raise ValueError("Price columns 'open' and 'close' must contain numeric values.")
        if not (isinstance(positive_gap_threshold, (int, float)) and positive_gap_threshold > 0):
            raise ValueError("Positive gap threshold must be a positive numeric value.")
        if not (isinstance(negative_gap_threshold, (int, float)) and negative_gap_threshold < 0):
            raise ValueError("Negative gap threshold must be a negative numeric value.")
        if positive_gap_threshold <= abs(negative_gap_threshold):
            print("Warning: Positive gap threshold is not strictly greater than the absolute negative gap threshold. This might lead to overlapping or unexpected signal logic.")

        self.data = data
        self.positive_gap_threshold = positive_gap_threshold
        self.negative_gap_threshold = negative_gap_threshold

    def generate_signal(self, current_index: int) -> str:
        """
        Generates a trading signal ("BUY", "SELL", or "HOLD") based on the gap
        between the previous day's close and the current day's open.
        """
        # Ensure the index is valid
        if not isinstance(current_index, int):
            raise TypeError("current_index must be an integer.")
        if current_index < 0 or current_index >= len(self.data):
            raise IndexError(f"current_index {current_index} is out of bounds for data of length {len(self.data)}.")

        # Not enough data to calculate the gap for the first data point
        if current_index == 0:
            return "HOLD"

        try:
            previous_close = self.data['close'].iloc[current_index - 1] #Use iloc for integer-based indexing
            current_open = self.data['open'].iloc[current_index] #Use iloc for integer-based indexing
        except KeyError as e:
            # This should ideally be caught in __init__, but as a safeguard
            raise ValueError(f"Missing 'open' or 'close' column in data: {e}")
        except IndexError as e:
            # This should ideally be caught by the current_index check, but as a safeguard
            raise IndexError(f"Data access error at index {current_index}: {e}")

        # Handle potential division by zero if previous_close is zero or very close to it
        if previous_close == 0:
            # If previous close is zero, gap calculation is problematic.
            # Depending on desired behavior, could return HOLD or raise error.
            # For robustness, returning HOLD is safer.
            return "HOLD"
        elif abs(previous_close) < 1e-9:  # A very small number close to zero
            print(f"Warning: Previous close price ({previous_close}) is very close to zero at index {current_index}. Gap calculation might be unstable.")
            return "HOLD"

        gap = (current_open - previous_close) / previous_close

        if gap > self.positive_gap_threshold:
            return "BUY"
        elif gap < self.negative_gap_threshold:
            return "SELL"
        else:
            return "HOLD"

class BuyAndHoldStrategy:
    """A simple strategy that buys at the beginning and holds."""
    def __init__(self, data: pd.DataFrame):
        """Initializes the BuyAndHoldStrategy."""
        self.data = data
        self.bought = False

    def generate_signal(self, current_index: int) -> str:
        """Buys on the first available data point and then holds."""
        if not self.bought:
            self.bought = True
            return "BUY"
        else:
            return "HOLD"

# --- 7. Event Handlers (Modified to Process Data) ---

async def on_minute_bar_update(bar: Bar):
    """Callback for minute bar data (OHLCV)."""
    global data, scalping_strategy, momentum_strategy, reversal_strategy, breakout_strategy, range_strategy, gap_strategy, buy_and_hold_strategy

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
    data = data[~data.index.duplicated(keep='last')]  # ensure only one entry per timestamp.

    # Preprocessing (Apply to last n rows to handle streaming data)
    data_processed = data.copy()
    data_processed = handle_missing_values(data_processed, ['open', 'high', 'low', 'close', 'volume'])
    data_processed = handle_outliers_iqr(data_processed, ['open', 'high', 'low', 'close', 'volume'])

    # At least a minimum data length is required to properly run the strategies. If not, the data is skipped until this is met.
    min_data_length = 50
    if len(data_processed) < min_data_length:
        print(f"Data length is less than the minimum required length of {min_data_length}. Skipping signal generation.")
        return

    # Generate Trading Signals
    scalping_signal = scalping_strategy.generate_signal(len(data_processed) - 1)
    momentum_signal = momentum_strategy.generate_signal(len(data_processed) - 1)
    reversal_signal = reversal_strategy.generate_signal(len(data_processed) - 1)
    breakout_signal = breakout_strategy.generate_signal(len(data_processed) - 1)
    range_signal = range_strategy.generate_signal(len(data_processed) - 1)
    gap_signal = gap_strategy.generate_signal(len(data_processed) - 1)
    buy_and_hold_signal = buy_and_hold_strategy.generate_signal(len(data_processed) - 1)

    print(f"Scalping Signal: {scalping_signal}")
    print(f"Momentum Signal: {momentum_signal}")
    print(f"Reversal Signal: {reversal_signal}")
    print(f"Breakout Signal: {breakout_signal}")
    print(f"Range Signal: {range_signal}")
    print(f"Gap Signal: {gap_signal}")
    print(f"Buy and Hold Signal:{buy_and_hold_signal}")

async def main():
    global data # Ensure we are using the global data DataFrame

    # Fetch initial historical data
    initial_data = await fetch_historical_data(SYMBOL, HISTORICAL_DAYS)

    if initial_data.empty:
        print("No historical data available to run backtests. Exiting.")
        return

    # Preprocess initial data for strategies
    data_for_strategies = initial_data.copy()
    data_for_strategies = handle_missing_values(data_for_strategies, ['open', 'high', 'low', 'close', 'volume'])
    data_for_strategies = handle_outliers_iqr(data_for_strategies, ['open', 'high', 'low', 'close', 'volume'])

    # Ensure enough data remains after preprocessing for strategies to initialize
    min_strategy_data_length = max(
        14, # RSI, ATR periods
        50, # Momentum MA periods
        20, # Bollinger Bands, SR periods
    )
    if len(data_for_strategies) < min_strategy_data_length:
        print(f"Not enough preprocessed data ({len(data_for_strategies)} bars) for strategies to initialize. Minimum required: {min_strategy_data_length}. Exiting.")
        return

    strategies = {
        "Scalping Strategy": ScalpingStrategy(data_for_strategies.copy()),
        "Momentum Trading Strategy": MomentumTradingStrategy(data_for_strategies.copy()),
        "Reversal Trading Strategy": ReversalTradingStrategy(data_for_strategies.copy()),
        "Breakout Trading Strategy": BreakoutTradingStrategy(data_for_strategies.copy()),
        "Range Trading Strategy": RangeTradingStrategy(data_for_strategies.copy()),
        "Gap Trading Strategy": GapTradingStrategy(data_for_strategies.copy()),
        "Buy and Hold Strategy": BuyAndHoldStrategy(data_for_strategies.copy())
    }

    print("\n--- Running Backtests for Each Strategy ---")
    for name, strategy_instance in strategies.items():
        print(f"\n===== Backtesting: {name} =====")
        try:
            backtester = Backtester(strategy_instance, initial_capital=100000)
            metrics = backtester.run_backtest()

            if "Error" in metrics:
                print(f"Error during backtest for {name}: {metrics['Error']}")
                continue

            print("\nI. Absolute Performance & Profitability")
            print(f"  Final Balance: ${metrics['Final Balance']:.2f}")
            print(f"  Total Returns: {metrics['Total Returns (%)']:.2f}%")
            print(f"  Profit Factor: {metrics['Profit Factor']:.2f}")
            print(f"  Win Rate: {metrics['Win Rate (%)']:.2f}%")
            print(f"  Loss Rate: {metrics['Loss Rate (%)']:.2f}%")
            print(f"  Average Win: ${metrics['Average Win']:.2f}")
            print(f"  Average Loss: ${metrics['Average Loss']:.2f}")
            print(f"  Expectancy: ${metrics['Expectancy']:.2f}")

            print("\nII. Risk (Downside & Volatility)")
            print(f"  Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
            print(f"  Max Drawdown Duration: {metrics['Max Drawdown Duration (minutes)']:.2f} minutes")
            print(f"  VaR (95%): {metrics['VaR (95%)']:.4f}")
            print(f"  CVaR (95%): {metrics['CVaR (95%)']:.4f}")
            print(f"  Ulcer Index: {metrics['Ulcer Index']:.4f}")

            print("\nIII. Risk-Adjusted Returns (Absolute & Downside Focused)")
            print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            print(f"  Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
            print(f"  Omega Ratio: {metrics['Omega Ratio']}") # May be N/A
            print(f"  Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
            print(f"  Recovery Factor: {metrics['Recovery Factor']:.2f}")

            print("\nIV. Relative Performance & Systematic Risk")
            print(f"  Beta: {metrics['Beta']:.2f}")
            print(f"  R-squared: {metrics['R-squared']:.2f}")
            print(f"  Jensen's Alpha: {metrics['Jensen\'s Alpha']:.4f}")
            print(f"  Treynor Ratio: {metrics['Treynor Ratio']:.2f}")
            print(f"  Tracking Error: {metrics['Tracking Error']:.4f}")
            print(f"  Information Ratio: {metrics['Information Ratio']:.2f}")

        except Exception as e:
            print(f"An error occurred during backtesting for {name}: {e}")

    # The live streaming part is commented out to focus on backtesting
    # crypto_stream.subscribe_bars(on_minute_bar_update, SYMBOL)
    # print(f"\nSubscribed to minute bar updates for {SYMBOL}. Starting stream...")
    # await crypto_stream.run()

if __name__ == '__main__':
    # Run the main asynchronous function
    asyncio.run(main())