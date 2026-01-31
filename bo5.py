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
import matplotlib.pyplot as plt  # Import matplotlib

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
# --- 7. Backtester Class ---

class Backtester:
    def __init__(self, strategy_instance, initial_capital=100000, commission_rate=0.001, slippage_rate=0.0001, risk_free_rate=0.02):
        """
        Initializes the Backtester with a strategy and simulation parameters.
        :param strategy_instance: An instance of a trading strategy class.
        :param initial_capital: Starting capital for the backtest.
        :param commission_rate: Transaction cost as a percentage of trade value (e.g., 0.001 for 0.1%).
        :param slippage_rate: Price deviation as a percentage of trade price (e.g., 0.0001 for 0.01%).
        :param risk_free_rate: Annual risk-free rate for risk-adjusted metrics (e.g., 0.02 for 2%).
        """
        self.strategy = strategy_instance
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate

        self.equity_curve = pd.Series(dtype=float)
        self.trades = [] # List of dictionaries: {'type': 'BUY/SELL', 'entry_price', 'exit_price', 'profit_loss', 'duration', 'entry_time', 'exit_time', 'shares'}
        self.current_position_shares = 0 # Number of shares/units currently held
        self.entry_price = 0
        self.entry_time = None
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.data = strategy_instance.data # Use the pre-processed data from the strategy
        self.metrics = {}

        # Ensure data has enough rows for calculations
        if len(self.data) == 0:
            raise ValueError("Strategy data is empty. Cannot run backtest.")

    def run_backtest(self):
        """
        Runs the backtest simulation over the historical data.
        """
        # Initialize equity curve with initial capital
        self.equity_curve = pd.Series(index=self.data.index, dtype=float)
        self.equity_curve.iloc[0] = self.initial_capital

        for i in range(len(self.data)):
            current_timestamp = self.data.index[i]
            current_close = self.data['close'].iloc[i]

            # Update portfolio value at each step (even if no trade)
            if i > 0:
                # Previous portfolio value
                prev_portfolio_value = self.equity_curve.iloc[i-1]
                # Calculate value of current position
                position_value = self.current_position_shares * current_close
                # Total portfolio value = cash + value of position
                self.portfolio_value = self.cash + position_value
                self.equity_curve.iloc[i] = self.portfolio_value
            else:
                self.equity_curve.iloc[i] = self.initial_capital


            signal = self.strategy.generate_signal(i)

            if signal == "BUY":
                if self.current_position_shares == 0: # Only buy if not already in a position
                    # Calculate actual entry price with slippage
                    actual_entry_price = current_close * (1 + self.slippage_rate)
                    # Determine how many shares to buy (use all available cash)
                    shares_to_buy = (self.cash / actual_entry_price) * (1 - self.commission_rate) # Account for commission on purchase
                    self.current_position_shares = shares_to_buy
                    self.entry_price = actual_entry_price
                    self.entry_time = current_timestamp
                    self.cash -= (shares_to_buy * actual_entry_price) / (1 - self.commission_rate) # Deduct actual cost including commission
                    # print(f"BUY at {current_timestamp}: {shares_to_buy:.4f} shares at {actual_entry_price:.2f}") # Commented out for cleaner output

            elif signal == "SELL":
                if self.current_position_shares > 0: # Only sell if in a long position
                    # Calculate actual exit price with slippage
                    actual_exit_price = current_close * (1 - self.slippage_rate)
                    # Calculate profit/loss for this trade
                    profit_loss = (actual_exit_price - self.entry_price) * self.current_position_shares
                    # Add profit/loss to cash, minus commission on sale
                    self.cash += (self.current_position_shares * actual_exit_price) * (1 - self.commission_rate)

                    self.trades.append({
                        'type': 'BUY_SELL',
                        'entry_price': self.entry_price,
                        'exit_price': actual_exit_price,
                        'profit_loss': profit_loss,
                        'duration': (current_timestamp - self.entry_time).total_seconds() / 60 if self.entry_time else 0, # in minutes
                        'entry_time': self.entry_time,
                        'exit_time': current_timestamp,
                        'shares': self.current_position_shares
                    })
                    # Reset position
                    self.current_position_shares = 0
                    self.entry_price = 0
                    self.entry_time = None
                    # print(f"SELL at {current_timestamp}: P/L = {profit_loss:.2f}") # Commented out for cleaner output

        # If still in a position at the end, close it out
        if self.current_position_shares > 0:
            final_close_price = self.data['close'].iloc[-1]
            actual_exit_price = final_close_price * (1 - self.slippage_rate)
            profit_loss = (actual_exit_price - self.entry_price) * self.current_position_shares
            self.cash += (self.current_position_shares * actual_exit_price) * (1 - self.commission_rate)
            self.trades.append({
                'type': 'FINAL_SELL',
                'entry_price': self.entry_price,
                'exit_price': actual_exit_price,
                'profit_loss': profit_loss,
                'duration': (self.data.index[-1] - self.entry_time).total_seconds() / 60 if self.entry_time else 0,
                'entry_time': self.entry_time,
                'exit_time': self.data.index[-1],
                'shares': self.current_position_shares
            })
            self.current_position_shares = 0

        # Final portfolio value
        self.portfolio_value = self.cash + (self.current_position_shares * self.data['close'].iloc[-1])
        self.equity_curve.iloc[-1] = self.portfolio_value

        self._calculate_metrics()
        self.plot_strategy()  # Call the plotting function after backtest
        return self.metrics

    def _calculate_metrics(self):
        """Calculates all performance metrics."""
        metrics = {}

        # Ensure equity curve is not empty
        if self.equity_curve.empty:
            self.metrics = {"Error": "Equity curve is empty, cannot calculate metrics."}
            return

        # Calculate daily returns for metrics requiring it
        # Assuming minute data, convert to daily returns, then annualize.  More stable than using minute returns directly.
        daily_equity = self.equity_curve.resample('D').last() #Resample to daily, keeping the last value of the day
        daily_strategy_returns = daily_equity.pct_change().dropna()
        MINUTES_PER_YEAR = 365  # Since we have daily returns.

        num_periods = len(self.equity_curve)
        if num_periods < 2:
            self.metrics = {"Error": "Not enough data points in equity curve to calculate returns."}
            return

        # Calculate period returns
        strategy_returns = self.equity_curve.pct_change().dropna()

        # Calculate benchmark returns (using close price of the asset itself)
        benchmark_returns = self.data['close'].pct_change().dropna()
        # Align indices for benchmark and strategy returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        if strategy_returns.empty:
            self.metrics = {"Error": "No valid returns to calculate metrics."}
            return

        # Annualized returns
        annualized_strategy_return = (1 + daily_strategy_returns).prod()**(MINUTES_PER_YEAR / len(daily_strategy_returns)) - 1 if len(daily_strategy_returns) > 0 else 0
        annualized_benchmark_return = (1 + benchmark_returns).prod()**(MINUTES_PER_YEAR / len(benchmark_returns)) - 1 if len(benchmark_returns) > 0 else 0

        # Annualized volatility (standard deviation)
        annualized_strategy_volatility = daily_strategy_returns.std() * math.sqrt(MINUTES_PER_YEAR) if len(daily_strategy_returns) > 1 else 0
        annualized_benchmark_volatility = benchmark_returns.std() * math.sqrt(MINUTES_PER_YEAR) if len(benchmark_returns) > 1 else 0


        # I. Absolute Performance & Profitability
        metrics['Final Balance'] = self.portfolio_value
        metrics['Total Returns (%)'] = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100

        gross_profits = sum(t['profit_loss'] for t in self.trades if t['profit_loss'] > 0)
        gross_losses = sum(t['profit_loss'] for t in self.trades if t['profit_loss'] < 0)
        metrics['Profit Factor'] = gross_profits / abs(gross_losses) if gross_losses != 0 else np.inf

        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        losing_trades = [t for t in self.trades if t['profit_loss'] < 0]
        total_trades = len(self.trades)

        metrics['Win Rate (%)'] = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        metrics['Loss Rate (%)'] = (len(losing_trades) / total_trades) * 100 if total_trades > 0 else 0
        metrics['Average Win'] = sum(t['profit_loss'] for t in winning_trades) / len(winning_trades) if len(winning_trades) > 0 else 0
        metrics['Average Loss'] = sum(t['profit_loss'] for t in losing_trades) / len(losing_trades) if len(losing_trades) > 0 else 0
        metrics['Expectancy'] = (metrics['Win Rate (%)'] / 100 * metrics['Average Win']) + (metrics['Loss Rate (%)'] / 100 * metrics['Average Loss'])


        # II. Risk (Downside & Volatility)
        # Drawdown calculation
        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = (self.equity_curve - peak) / peak
        metrics['Max Drawdown (%)'] = drawdown.min() * 100 if not drawdown.empty else 0

        # Max Drawdown Duration
        if not drawdown.empty and drawdown.min() < 0:
            # Find the start and end of the max drawdown period
            end_of_mdd = drawdown.idxmin()
            start_of_mdd = self.equity_curve.loc[:end_of_mdd].idxmax()
            recovery_point = self.equity_curve.loc[end_of_mdd:].index[
                (self.equity_curve.loc[end_of_mdd:] >= self.equity_curve.loc[start_of_mdd]).values
            ]
            if not recovery_point.empty:
                metrics['Max Drawdown Duration (minutes)'] = (recovery_point[0] - start_of_mdd).total_seconds() / 60
            else:
                metrics['Max Drawdown Duration (minutes)'] = (self.equity_curve.index[-1] - start_of_mdd).total_seconds() / 60
        else:
            metrics['Max Drawdown Duration (minutes)'] = 0

        # VaR (Value-at-Risk) - 95% confidence level
        # Assuming normal distribution for simplicity, can use historical simulation
        # For historical simulation, we sort returns and pick the percentile
        if not strategy_returns.empty:
            metrics['VaR (95%)'] = np.percentile(strategy_returns, 5) # 5th percentile of returns
        else:
            metrics['VaR (95%)'] = 0

        # CVaR (Conditional Value-at-Risk) or Expected Shortfall (ES)
        if not strategy_returns.empty:
            var_threshold = metrics['VaR (95%)']
            cvar_returns = strategy_returns[strategy_returns <= var_threshold]
            metrics['CVaR (95%)'] = cvar_returns.mean() if not cvar_returns.empty else 0
        else:
            metrics['CVaR (95%)'] = 0

        # Ulcer Index - requires a more complex calculation
        # Simplified Ulcer Index: sqrt(sum((drawdown_i)^2) / N)
        if not drawdown.empty:
            metrics['Ulcer Index'] = np.sqrt((drawdown**2).sum() / len(drawdown)) if len(drawdown) > 0 else 0
        else:
            metrics['Ulcer Index'] = 0


        # III. Risk-Adjusted Returns (Absolute & Downside Focused)
        # Sharpe Ratio
        excess_returns = daily_strategy_returns - (self.risk_free_rate / MINUTES_PER_YEAR)
        metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * math.sqrt(MINUTES_PER_YEAR) if excess_returns.std() != 0 else np.nan

        # Sortino Ratio
        downside_returns = daily_strategy_returns[daily_strategy_returns < (self.risk_free_rate / MINUTES_PER_YEAR)]
        downside_deviation = downside_returns.std()
        metrics['Sortino Ratio'] = excess_returns.mean() / downside_deviation * math.sqrt(MINUTES_PER_YEAR) if downside_deviation != 0 else np.nan

        # Omega Ratio - more complex, requires integral, simplified approximation
        # For simplicity, not implementing full Omega Ratio here due to complexity for minute data.
        # It typically involves a threshold return and integrating the distribution.
        metrics['Omega Ratio'] = "N/A (Complex for minute data)"

        # Calmar Ratio
        metrics['Calmar Ratio'] = annualized_strategy_return / abs(metrics['Max Drawdown (%)'] / 100) if metrics['Max Drawdown (%)'] != 0 else np.nan

        # Recovery Factor
        net_profit = self.portfolio_value - self.initial_capital
        metrics['Recovery Factor'] = net_profit / abs(self.initial_capital * metrics['Max Drawdown (%)'] / 100) if metrics['Max Drawdown (%)'] != 0 else np.nan


        # IV. Relative Performance & Systematic Risk
        if not benchmark_returns.empty and len(benchmark_returns) > 1:
            # Beta
            covariance = strategy_returns.cov(benchmark_returns)
            variance_benchmark = benchmark_returns.var()
            metrics['Beta'] = covariance / variance_benchmark if variance_benchmark != 0 else np.nan

            # R-squared
            correlation = strategy_returns.corr(benchmark_returns)
            metrics['R-squared'] = correlation**2 if not pd.isna(correlation) else np.nan

            # Alpha / Jensen's Alpha
            # Jensen's Alpha = Rp - [Rf + Beta * (Rm - Rf)]
            # Rp = strategy return, Rf = risk-free rate, Rm = benchmark return
            if not pd.isna(metrics['Beta']):
                metrics['Jensens Alpha'] = annualized_strategy_return - (self.risk_free_rate + metrics['Beta'] * (annualized_benchmark_return - self.risk_free_rate))
            else:
                metrics['Jensens Alpha'] = np.nan

            # Treynor Ratio
            metrics['Treynor Ratio'] = (annualized_strategy_return - self.risk_free_rate) / metrics['Beta'] if metrics['Beta'] != 0 else np.nan

            # Tracking Error
            active_returns = strategy_returns - benchmark_returns
            metrics['Tracking Error'] = active_returns.std() * math.sqrt(MINUTES_PER_YEAR) if len(active_returns) > 1 else np.nan

            # Information Ratio (IR)
            metrics['Information Ratio'] = (active_returns.mean() * MINUTES_PER_YEAR) / metrics['Tracking Error'] if metrics['Tracking Error'] != 0 else np.nan
    
        else:
            metrics['Beta'] = np.nan
            metrics['R-squared'] = np.nan
            metrics['Jensens Alpha'] = np.nan
            metrics['Treynor Ratio'] = np.nan
            metrics['Tracking Error'] = np.nan
            metrics['Information Ratio'] = np.nan


        self.metrics = metrics
        return

    def plot_strategy(self):
        """Plots the closing prices, support, and resistance levels."""
        strategy_name = type(self.strategy).__name__
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['close'], label='Closing Price', alpha=0.7)
        plt.plot(self.data['Resistance'], label='Resistance', color='red', linestyle='--')
        plt.plot(self.data['Support'], label='Support', color='green', linestyle='--')

        # Plot buy/sell signals (you might need to adjust the logic depending on how exactly your strategy generates signals)
        buy_signals = [trade['entry_time'] for trade in self.trades if trade['type'] == 'BUY_SELL']
        sell_signals = [trade['exit_time'] for trade in self.trades if trade['type'] == 'BUY_SELL']
        final_sell_signal = [trade['exit_time'] for trade in self.trades if trade['type'] == 'FINAL_SELL']

        # Find corresponding indices to extract the closing prices
        buy_indices = []
        for ts in buy_signals:
            try:
                buy_indices.append(self.data.index.get_loc(ts))
            except KeyError:
                print(f"Warning: Buy signal timestamp {ts} not found in data index. Skipping.")

        sell_indices = []
        for ts in sell_signals:
            try:
                sell_indices.append(self.data.index.get_loc(ts))
            except KeyError:
                print(f"Warning: Sell signal timestamp {ts} not found in data index. Skipping.")

        final_sell_indices = []
        for ts in final_sell_signal:
            try:
                final_sell_indices.append(self.data.index.get_loc(ts))
            except KeyError:
                print(f"Warning: Final sell signal timestamp {ts} not found in data index. Skipping.")

        # Get the corresponding close prices
        buy_prices = self.data['close'].iloc[buy_indices] if buy_indices else []
        sell_prices = self.data['close'].iloc[sell_indices] if sell_indices else []
        final_sell_prices = self.data['close'].iloc[final_sell_indices] if final_sell_indices else []

        # Plot buy and sell signals
        plt.scatter(buy_prices.index, buy_prices, marker='^', s=100, color='green', label='Buy Signal')
        plt.scatter(sell_prices.index, sell_prices, marker='v', s=100, color='red', label='Sell Signal')
        plt.scatter(final_sell_prices.index, final_sell_prices, marker='v', s=100, color='purple', label='Final Sell Signal')

        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title(f'{strategy_name} Backtest with Support & Resistance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# --- 9. Main Execution Logic ---

async def main():
    global data # Ensure we are using the global data DataFrame
    TEST_HISTORIC_DAYS = 365

    # Fetch initial historical data
    initial_data = await fetch_historical_data(SYMBOL, TEST_HISTORIC_DAYS)

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
        "Breakout Trading Strategy": BreakoutTradingStrategy(data_for_strategies.copy()),
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
            print(f"  Jensen's Alpha: {metrics['Jensens Alpha']:.4f}")
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