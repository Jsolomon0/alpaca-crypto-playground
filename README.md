# Trader9

## Overview
Trader9 is a collection of Python scripts focused on crypto market data streaming, indicator calculation, and strategy backtesting using Alpaca's data APIs. The scripts are mostly standalone and experiment with different strategies (breakout, scalping, momentum, reversal, range, gap, buy-and-hold) against historical minute bars, with some real-time streaming examples.

## Requirements
- Python 3.x
- Suggested packages (varies by script):
  - alpaca-py
  - pandas
  - numpy
  - python-dotenv
  - nest_asyncio
  - matplotlib (for scripts that plot)

## Setup
1) Create a `.env` file in this folder with your Alpaca credentials:

```
ALPACA_API_KEY_ID=your_key
ALPACA_API_SECRET_KEY=your_secret
```

2) Install dependencies (example):

```
pip install alpaca-py pandas numpy python-dotenv nest_asyncio matplotlib
```

## Running
Most scripts can be run directly:

```
python breakout.py
```

Many files use `asyncio` and call `asyncio.run(main())` when executed as a script.

## Script Guide
- `crypto_stream_app0.py`: Simple example streaming trades, quotes, bars, and order books.
- `crypto_data_stream_app1.py`: Similar streaming example with clearer comments and callbacks.
- `crypto_stream2.py`: Streaming with a rolling DataFrame and basic preprocessing helpers.
- `crypto_stream3.py`: Streaming plus historical data fetch for a single symbol.
- `crypto_streamNback_5.py`: Multi-strategy backtest runner plus optional streaming hooks.
- `crypto_stream_stratgey_4._needs_testing.py`: Experimental multi-strategy backtest; marked as needing testing.

- `breakout.py`, `breakout2.py`, `breakout3.py`, `breakout4.py`, `bo5.py`: Breakout strategy backtests with historical data and indicator calculations.
- `crypto_backtest.py`: Multi-strategy backtesting (scalping, momentum, reversal, breakout, range, gap).
- `generic2.py`, `genericXRP.py`, `genericXRP.1.py`: Generic strategy backtests (despite names, default symbol is `BTC/USD`).
- `gap3.py`: Gap strategy backtest with additional debugging notes at top.
- `Rest_Supp.py`: Historical data fetch and plotting support code.

- `tester.py`: Small file write example (not part of trading logic).
- `Gap.py`, `historic_downloader.py`: Empty placeholders.

## Configuration Notes
- Most scripts default to `SYMBOL = "BTC/USD"` and set a lookback window (e.g., `HISTORICAL_DAYS`).
- Streaming subscriptions are sometimes commented out; review the `main()` function in each script.
- These scripts use Alpaca data APIs only; no live trading logic is included.

## Safety
This repository is for research and experimentation. Use caution with real money, and validate any strategy thoroughly before considering live trading.
