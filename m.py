# savage_tuner.py

import ccxt
import pandas as pd
import time
from hyperopt import fmin, tpe, hp, Trials
from rich.console import Console
import json

console = Console()

# === Load Historical Data ===
def load_data(symbol, timeframe='5m', limit=1000):
    exchange = ccxt.gateio()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# === Backtest Logic ===
def backtest(df, params):
    balance = 100  # Start with $100
    position = 0
    entry_price = 0

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]

        if position == 0:
            if df['close'].iloc[i] > df['open'].iloc[i] * (1 + params['entry_threshold']):
                position = balance / price
                entry_price = price
        else:
            if price >= entry_price * (1 + params['tp']) or price <= entry_price * (1 - params['sl']):
                balance = position * price
                position = 0

    if position > 0:
        balance = position * df['close'].iloc[-1]

    return balance

# === Hyperopt Objective ===
def objective(params):
    df = load_data(symbol="BTC/USDT")
    final_balance = backtest(df, params)
    loss = -final_balance  # We want to maximize final balance => minimize negative balance
    return loss

# === Define Search Space ===
space = {
    'entry_threshold': hp.uniform('entry_threshold', 0.001, 0.01),
    'tp': hp.uniform('tp', 0.002, 0.02),
    'sl': hp.uniform('sl', 0.002, 0.02),
}

# === Hyperparameter Tuning ===
def hyperparam_tune():
    console.print("[bold green]🚀 Launching FULL SAVAGE Hyperopt...[/bold green]")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    console.print(f"[bold yellow]🏆 Best Params Found: {json.dumps(best, indent=2)}[/bold yellow]")
    return best

# === Apply Best Config Live ===
def live_trading(best_params):
    console.print("[bold green]💥 Starting LIVE mode with tuned parameters![/bold green]")
    # Here you'd plug best_params into your real-time system (your faster_beast or maximum_beast)

if __name__ == "__main__":
    best_params = hyperparam_tune()
    live_trading(best_params)
