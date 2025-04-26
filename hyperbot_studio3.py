import ccxt
import time
import threading
import random
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'

TARGET_USD_PER_TRADE = 5
LEVERAGE = 20
PRICE_OFFSET = 0.0002
HEDGE_THRESHOLD = 0.004
GROWTH_FACTOR = 1.01

exchange = ccxt.gateio({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

active_trades = []
start_time = time.time()

class Trade:
    def __init__(self, symbol, side, size, entry_price):
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = time.time()
        self.exit_price = None
        self.pnl = 0

def fetch_best_symbol():
    symbols = [s for s in exchange.load_markets().keys() if '/USDT:USDT' in s]
    best = None
    best_rate = -999
    for symbol in random.sample(symbols, min(30, len(symbols))):
        try:
            funding = exchange.fetch_funding_rate(symbol)
            if funding['fundingRate'] > best_rate:
                best = symbol
                best_rate = funding['fundingRate']
        except:
            continue
    return best, best_rate

def place_post_only(symbol, side, usd_amount):
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    price = price * (1 - PRICE_OFFSET) if side == "buy" else price * (1 + PRICE_OFFSET)
    size = usd_amount / price
    size = round(size, 5)
    try:
        exchange.create_limit_order(
            symbol.replace("_", "/"),
            side,
            size,
            price,
            params={"postOnly": True, "marginMode": "cross"}
        )
        console.print(f"[green]Placed {side.upper()} {symbol} size {size} @ {price:.6f}[/green]")
        active_trades.append(Trade(symbol, side, size, price))
    except Exception as e:
        console.print(f"[red]Failed to place order: {e}[/red]")

def monitor_trades():
    while True:
        for trade in active_trades[:]:
            ticker = exchange.fetch_ticker(trade.symbol)
            current_price = ticker['last']
            delta = (current_price - trade.entry_price) / trade.entry_price
            if trade.side == "sell":
                delta = -delta
            if abs(delta) >= HEDGE_THRESHOLD:
                opposite = "buy" if trade.side == "sell" else "sell"
                try:
                    exchange.create_market_order(
                        trade.symbol.replace("_", "/"),
                        opposite,
                        trade.size,
                        params={"reduceOnly": True, "marginMode": "cross"}
                    )
                    trade.exit_price = current_price
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - trade.exit_price) * trade.size
                    active_trades.remove(trade)
                    console.print(f"[bold red]Auto-hedged {trade.symbol} for PnL {trade.pnl:.4f}[/bold red]")
                except Exception as e:
                    console.print(f"[red]Failed to hedge: {e}[/red]")
        time.sleep(5)

def blood_swarm():
    usd_size = TARGET_USD_PER_TRADE
    while True:
        best_symbol, funding = fetch_best_symbol()
        if best_symbol:
            side = 'sell' if funding > 0 else 'buy'
            place_post_only(best_symbol, side, usd_size)
            if funding > 0.0005:
                usd_size *= GROWTH_FACTOR
        time.sleep(300)

def dashboard():
    with Live(refresh_per_second=1) as live:
        while True:
            table = Table(title="💀 BLOOD GOD MATRIX")
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Entry Price")
            table.add_column("Current Price")
            table.add_column("PNL ($)")
            elapsed = max(1, time.time() - start_time)
            cps = sum([t.pnl for t in active_trades]) * 100 / elapsed
            for trade in active_trades:
                try:
                    ticker = exchange.fetch_ticker(trade.symbol)
                    curr_price = ticker['last']
                    pnl = (curr_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - curr_price) * trade.size
                    table.add_row(
                        trade.symbol,
                        trade.side.upper(),
                        f"{trade.entry_price:.6f}",
                        f"{curr_price:.6f}",
                        f"{pnl:.4f}"
                    )
                except:
                    continue
            console.print(f"[bold magenta]Cents per second (CPS): {cps:.5f}[/bold magenta]")
            live.update(table)
            time.sleep(2)

if __name__ == "__main__":
    console.print("[bold red]🚀 BLOOD GOD MODE: DEPLOYING FULL SWARM 🚀[/bold red]")
    threading.Thread(target=blood_swarm, daemon=True).start()
    threading.Thread(target=monitor_trades, daemon=True).start()
    dashboard()
