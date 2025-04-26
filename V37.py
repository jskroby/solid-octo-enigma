import ccxt
import random
import time
import threading
import numpy as np
import os
import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live

# === CONFIG ===
API_KEY = os.getenv("GATEIO_API_KEY")
API_SECRET = os.getenv("GATEIO_API_SECRET")
OLLAMA_URL = "http://localhost:11434/api/generate"  # Update if needed
OLLAMA_MODEL = "llama3"

TARGET_USD_PER_TRADE = 5
LEVERAGE = 25
GROWTH_FACTOR = 1.05
HEDGE_THRESHOLD = 0.0025
VOLATILITY_THRESHOLD = 0.005

console = Console()
start_time = time.time()
lock = threading.Lock()
active_trades = []
total_pnl = 0

# === INIT EXCHANGE ===
exchange = ccxt.gateio({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

class Trade:
    def __init__(self, symbol, side, size, entry_price):
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = time.time()
        self.exit_price = None
        self.pnl = 0

def fetch_symbols():
    return [s for s in exchange.load_markets() if '/USDT:USDT' in s and exchange.markets[s]['active']]

def volatility_score(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=12)
        if len(ohlcv) < 5:
            return 0
        closes = np.array([c[4] for c in ohlcv])
        returns = np.diff(np.log(closes))
        return np.std(returns)
    except:
        return 0

def predict_volatility(symbol, vol_now):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Symbol {symbol} volatility {vol_now:.5f}. Should we trade? Reply YES or NO.",
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=5)
        if "yes" in response.json()["response"].lower():
            return 1.0
        else:
            return 0.0
    except Exception as e:
        console.print(f"[red]Ollama error: {e}[/red]")
        return 0.0

def place_order(symbol, side, usd_amount):
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        if not price:
            return
        price = price * (1 - 0.0001) if side == "buy" else price * (1 + 0.0001)
        size = round(usd_amount / price, 4)
        exchange.create_limit_order(
            symbol.replace("_", "/"), side, size, price,
            params={"postOnly": True, "marginMode": "cross"}
        )
        console.print(f"[green]Placed {side.upper()} {symbol} {size} @ {price:.5f}[/green]")
        with lock:
            active_trades.append(Trade(symbol, side, size, price))
    except Exception as e:
        console.print(f"[red]Order error: {e}[/red]")

def blood_swarm():
    global TARGET_USD_PER_TRADE
    while True:
        try:
            symbols = fetch_symbols()
            random.shuffle(symbols)
            for symbol in symbols[:40]:
                vol_now = volatility_score(symbol)
                if np.isnan(vol_now) or vol_now == 0:
                    continue
                prediction = predict_volatility(symbol, vol_now)
                if prediction > 0.4 and vol_now > VOLATILITY_THRESHOLD:
                    side = random.choice(["buy", "sell"])
                    place_order(symbol, side, TARGET_USD_PER_TRADE)
                elif vol_now > VOLATILITY_THRESHOLD and random.random() < 0.5:
                    side = random.choice(["buy", "sell"])
                    console.print(f"[yellow]⚡ FORCED TRADE {side.upper()} {symbol}[/yellow]")
                    place_order(symbol, side, TARGET_USD_PER_TRADE)
        except Exception as e:
            console.print(f"[red]Swarm error: {e}[/red]")
        time.sleep(0.5)

def monitor_trades():
    global total_pnl, TARGET_USD_PER_TRADE
    while True:
        with lock:
            for trade in active_trades[:]:
                try:
                    ticker = exchange.fetch_ticker(trade.symbol)
                    price = ticker['last']
                    delta = (price - trade.entry_price) / trade.entry_price
                    if trade.side == "sell":
                        delta = -delta
                    if abs(delta) >= HEDGE_THRESHOLD:
                        opposite = "buy" if trade.side == "sell" else "sell"
                        exchange.create_market_order(
                            trade.symbol.replace("_", "/"), opposite, trade.size,
                            params={"reduceOnly": True, "marginMode": "cross"}
                        )
                        trade.exit_price = price
                        trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - trade.exit_price) * trade.size
                        total_pnl += trade.pnl
                        TARGET_USD_PER_TRADE *= GROWTH_FACTOR
                        active_trades.remove(trade)
                        console.print(f"[bold red]CLOSED {trade.symbol} for +${trade.pnl:.5f} profit[/bold red]")
                except Exception as e:
                    console.print(f"[red]Monitor error: {e}[/red]")
        time.sleep(0.5)

def dashboard():
    with Live(refresh_per_second=1) as live:
        while True:
            table = Table(title="💀 BLOODFIRE V38 - SINGLE KEY OLLAMA ⚡")
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Entry")
            table.add_column("Current")
            table.add_column("PNL ($)")
            with lock:
                elapsed = max(1, time.time() - start_time)
                cps = (total_pnl * 100) / elapsed
                for trade in active_trades:
                    try:
                        ticker = exchange.fetch_ticker(trade.symbol)
                        curr_price = ticker['last']
                        pnl = (curr_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - curr_price) * trade.size
                        table.add_row(
                            trade.symbol,
                            trade.side.upper(),
                            f"{trade.entry_price:.5f}",
                            f"{curr_price:.5f}",
                            f"{pnl:.5f}"
                        )
                    except:
                        continue
            console.print(f"[bold magenta]⚡ CPS: {cps:.5f} ¢/s ⚡ Total PnL: ${total_pnl:.5f}[/bold magenta]")
            live.update(table)
            time.sleep(2)

if __name__ == "__main__":
    console.print("[bold red]🚀 BLOODFIRE V38: FINAL CLEAN LAUNCH 🚀[/bold red]")
    threading.Thread(target=blood_swarm, daemon=True).start()
    threading.Thread(target=monitor_trades, daemon=True).start()
    dashboard()
