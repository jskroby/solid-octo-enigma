# savage_hyperlogger.py

import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
import uuid

console = Console()

# === Load Historical Data ===
def load_data(symbol, timeframe='5m', limit=500):
    exchange = ccxt.gateio()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# === Hyper Logging Backtester ===
class Backtester:
    def __init__(self, df, entry_threshold=0.003, tp=0.01, sl=0.005):
        self.df = df
        self.entry_threshold = entry_threshold
        self.tp = tp
        self.sl = sl
        self.balance = 100  # Start balance
        self.equity = []
        self.trades = []

    def run(self):
        position = 0
        entry_price = 0

        for i in range(1, len(self.df)):
            price = self.df['close'].iloc[i]
            prev_price = self.df['close'].iloc[i-1]

            if position == 0:
                if price > self.df['open'].iloc[i] * (1 + self.entry_threshold):
                    position = self.balance / price
                    entry_price = price
                    trade = {
                        'id': str(uuid.uuid4()),
                        'direction': 'long',
                        'entry': entry_price,
                        'exit': None,
                        'pnl': None,
                        'timestamp_entry': self.df['timestamp'].iloc[i],
                        'timestamp_exit': None
                    }
                    self.trades.append(trade)
            else:
                if price >= entry_price * (1 + self.tp) or price <= entry_price * (1 - self.sl):
                    self.balance = position * price
                    position = 0
                    self.trades[-1]['exit'] = price
                    self.trades[-1]['timestamp_exit'] = self.df['timestamp'].iloc[i]
                    raw_pnl = (price - self.trades[-1]['entry'])
                    self.trades[-1]['pnl'] = raw_pnl * (self.trades[-1]['entry'] != 0)

            self.equity.append(self.balance)

    def show_logs(self):
        table = Table(title="🐷 Trade Log")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("Entry Time", justify="right")
        table.add_column("Exit Time", justify="right")

        for trade in self.trades:
            table.add_row(
                trade['id'][:6],
                f"{trade['entry']:.2f}",
                f"{trade['exit']:.2f}" if trade['exit'] else "-",
                f"{trade['pnl']:.4f}" if trade['pnl'] else "-",
                str(trade['timestamp_entry']),
                str(trade['timestamp_exit']) if trade['timestamp_exit'] else "-"
            )
        console.print(table)

    def plot_equity(self):
        fig, ax = plt.subplots()
        ax.set_title("🔥 Live Equity Curve")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Balance ($)")

        def animate(i):
            ax.clear()
            ax.plot(self.equity[:i])
            ax.set_title(f"🔥 Live Equity Curve - Step {i}")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Balance ($)")
            plt.tight_layout()

        ani = animation.FuncAnimation(fig, animate, frames=len(self.equity), interval=50)
        plt.show()

if __name__ == "__main__":
    df = load_data('BTC/USDT')
    bt = Backtester(df, entry_threshold=0.003, tp=0.008, sl=0.004)

    console.print("[bold yellow]⏳ Starting Savage HyperLogger Backtest...[/bold yellow]")
    bt.run()
    bt.show_logs()
    bt.plot_equity()
