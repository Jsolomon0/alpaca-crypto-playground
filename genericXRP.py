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

class GenericStrategy:
    def __init__(self, data, rsi_period=14, stochastic_period=14, ma_sm_period=15, ma_lg_period=60):
        self.data = data.copy() # Ensure working on a copy
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.ma_sm_period = ma_sm_period
        self.ma_lg_period = ma_lg_period

        # Calculate Indicators
        self.data['RSI'] = calculate_rsi(self.data['close'], self.rsi_period)
        # self.data['Stochastic'] = self.calculate_stochastic(self.stochastic_period) #Needs full Implementation
        self.data['SMA'] = calculate_sma(self.data['close'], self.ma_period)
        self.data['ATR'] = calculate_atr(self.data)
        self.data.dropna(inplace=True) # Drop NaNs introduced by indicator calculations

    def calculate_stochastic(self, period):
        """Calculates Stochastic Oscillator (Placeholder - needs full implementation)."""
        # Implement stochastic calculation here
        # Requires calculating %K and %D
        return pd.Series(index=self.data.index)  # Placeholder

    def generate_signal(self, current_index):
        """Generates trading signal (Buy, Sell, Hold)."""
        if current_index >= len(self.data):
            return "HOLD" # Index out of bounds

        # Ensure indicators are available
        if 'RSI' not in self.data.columns or pd.isna(self.data['RSI'].iloc[current_index]):
            return "HOLD"

        # Implement your scalping strategy logic here using self.data and calculated indicators
        # Example:
        if self.data['RSI'].iloc[current_index] > 70:
            return "SELL"
        elif self.data['RSI'].iloc[current_index] < 30:
            return "BUY"
        else:
            return "HOLD"

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

        return self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculates all performance metrics."""
        metrics = {}

        # Ensure equity curve is not empty
        if self.equity_curve.empty:
            return {"Error": "Equity curve is empty, cannot calculate metrics."}

        # Calculate daily returns for metrics requiring it
        # Assuming minute data, convert to daily for more stable annualization
        # Or, use minute returns and a larger annualization factor
        # For simplicity, let's use minute returns and annualize based on total minutes in a year
        # Total minutes in a year = 365 days * 24 hours/day * 60 minutes/hour
        MINUTES_PER_YEAR = 365 * 24 * 60
        num_periods = len(self.equity_curve)
        if num_periods < 2:
            return {"Error": "Not enough data points in equity curve to calculate returns."}

        # Calculate period returns
        strategy_returns = self.equity_curve.pct_change().dropna()

        # Calculate benchmark returns (using close price of the asset itself)
        benchmark_returns = self.data['close'].pct_change().dropna()
        # Align indices for benchmark and strategy returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        if strategy_returns.empty:
            return {"Error": "No valid returns to calculate metrics."}

        # Annualized returns
        annualized_strategy_return = (1 + strategy_returns).prod()**(MINUTES_PER_YEAR / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0
        annualized_benchmark_return = (1 + benchmark_returns).prod()**(MINUTES_PER_YEAR / len(benchmark_returns)) - 1 if len(benchmark_returns) > 0 else 0

        # Annualized volatility (standard deviation)
        annualized_strategy_volatility = strategy_returns.std() * math.sqrt(MINUTES_PER_YEAR) if len(strategy_returns) > 1 else 0
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
        excess_returns = strategy_returns - (self.risk_free_rate / MINUTES_PER_YEAR)
        metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * math.sqrt(MINUTES_PER_YEAR) if excess_returns.std() != 0 else np.nan

        # Sortino Ratio
        downside_returns = strategy_returns[strategy_returns < (self.risk_free_rate / MINUTES_PER_YEAR)]
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


        return metrics

# --- 8. Event Handlers (Modified to Process Data) ---

# This function is for live streaming, not directly used in backtesting
async def on_minute_bar_update( bar: Bar):
    """Callback for minute bar data (OHLCV)."""
    global data, generic_strategy

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

    # Generate Trading Signals (These would be for live trading, not backtesting)
    
    generic_signal = generic_strategy.generate_signal(len(data_processed) - 1)
    print(f"Generic Signal: {generic_signal}")
    


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
        "Generic Strategy": GenericStrategy(data_for_strategies.copy())
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