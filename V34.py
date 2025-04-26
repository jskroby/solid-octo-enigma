import ccxt
import torch
import time
import random
import threading
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from transformers import AutoModelForSequenceClassification, AutoTokenizer

console = Console()

# === API KEYS ===
API_KEYS = [
    {"apiKey": "4efbe203fbac0e4bcd0003a0910c801b", "secret": "8b0e9cf47fdd97f2a42bca571a74105c53a5f906cd4ada125938d05115d063a0"},
    {"apiKey": "YOUR_API_KEY2", "secret": "YOUR_SECRET_KEY2"},
    # Add more accounts if needed
]

# === CONFIG ===
TARGET_USD_PER_TRADE = 4
LEVERAGE = 25
HEDGE_THRESHOLD = 0.003
GRID_CLUSTER_SIZE = 3
GROWTH_FACTOR = 1.035
ORDER_LIFETIME = 8
RETRAIN_INTERVAL = 3600  # seconds (1 hour)
SNIPER_VOL_THRESHOLD = 0.5

# === GPU Volatility Predictor (Llama3 Tiny) ===
tokenizer = AutoTokenizer.from_pretrained("openai-community/llama-3-tiny")
model = AutoModelForSequenceClassification.from_pretrained(
    "openai-community/llama-3-tiny",
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# === EXCHANGES ===
exchanges = []
for keys in API_KEYS:
    ex = ccxt.gateio({
        'apiKey': keys['apiKey'],
        'secret': keys['secret'],
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    exchanges.append(ex)

lock = threading.Lock()
active_trades = []
start_time = time.time()
total_pnl = 0

class Trade:
    def __init__(self, symbol, side, size, entry_price, account_id):
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = time.time()
        self.exit_price = None
        self.pnl = 0
        self.account_id = account_id

def fetch_symbols(ex):
    markets = ex.load_markets()
    return [s for s in markets if '/USDT:USDT' in s and markets[s]['active']]

def volatility_score(symbol, ex):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, '1m', limit=10)
        closes = np.array([x[4] for x in ohlcv])
        returns = np.diff(np.log(closes))
        vol = np.std(returns)
        return vol
    except:
        return 0

def predict_volatility(symbol, vol_now):
    input_text = f"Symbol {symbol} volatility {vol_now:.5f}. Predict next 5m."
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    score = outputs.logits.softmax(dim=1)[0][1].item()
    return score

def place_post_only(symbol, side, usd_amount, ex, account_id):
    try:
        ticker = ex.fetch_ticker(symbol)
        price = ticker['last']
        if not price:
            return
        price = price * (1 - 0.0001) if side == "buy" else price * (1 + 0.0001)
        size = usd_amount / price
        size = round(size, 4)

        ex.create_limit_order(
            symbol.replace("_", "/"),
            side,
            size,
            price,
            params={"postOnly": True, "marginMode": "cross"}
        )

        console.print(f"[green]Acct {account_id}: {side.upper()} {symbol} {size} @ {price:.6f}[/green]")
        with lock:
            active_trades.append(Trade(symbol, side, size, price, account_id))

    except Exception as e:
        console.print(f"[red]Failed post-only: {e}[/red]")

def monitor_trades():
    global total_pnl
    while True:
        with lock:
            for trade in active_trades[:]:
                try:
                    ex = exchanges[trade.account_id]
                    ticker = ex.fetch_ticker(trade.symbol)
                    current_price = ticker['last']
                    delta = (current_price - trade.entry_price) / trade.entry_price
                    if trade.side == "sell":
                        delta = -delta
                    if abs(delta) >= HEDGE_THRESHOLD:
                        opposite = "buy" if trade.side == "sell" else "sell"
                        ex.create_market_order(
                            trade.symbol.replace("_", "/"),
                            opposite,
                            trade.size,
                            params={"reduceOnly": True, "marginMode": "cross"}
                        )
                        trade.exit_price = current_price
                        trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - trade.exit_price) * trade.size
                        total_pnl += trade.pnl
                        active_trades.remove(trade)
                        console.print(f"[bold red]💰 HEDGED {trade.symbol} PnL ${trade.pnl:.5f}[/bold red]")
                except Exception as e:
                    console.print(f"[red]Monitor error: {e}[/red]")
        time.sleep(0.5)

def blood_swarm():
    usd_size = TARGET_USD_PER_TRADE
    last_retrain = time.time()
    while True:
        for idx, ex in enumerate(exchanges):
            try:
                symbols = fetch_symbols(ex)
                scored_symbols = []
                for symbol in random.sample(symbols, min(40, len(symbols))):
                    vol = volatility_score(symbol, ex)
                    pred = predict_volatility(symbol, vol)
                    if pred > SNIPER_VOL_THRESHOLD:
                        scored_symbols.append((symbol, pred))
                scored_symbols.sort(key=lambda x: -x[1])

                for sym, _ in scored_symbols[:GRID_CLUSTER_SIZE]:
                    side = random.choice(["buy", "sell"])
                    place_post_only(sym, side, usd_size, ex, idx)

            except Exception as e:
                console.print(f"[red]Swarm error: {e}[/red]")

        # Auto-grow sizing
        if time.time() - start_time > 600:
            usd_size *= GROWTH_FACTOR

        # Retrain model (dummy for now, real retrain logic later)
        if time.time() - last_retrain > RETRAIN_INTERVAL:
            console.print("[bold cyan]🔁 Retraining Volatility Model (Placeholder)...[/bold cyan]")
            last_retrain = time.time()

        time.sleep(2)

def dashboard():
    with Live(refresh_per_second=2) as live:
        while True:
            table = Table(title="💀 BLOODFIRE V36.5 - DEMON MATRIX 💀")
            table.add_column("Acct")
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
                        ex = exchanges[trade.account_id]
                        ticker = ex.fetch_ticker(trade.symbol)
                        curr_price = ticker['last']
                        pnl = (curr_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - curr_price) * trade.size
                        table.add_row(
                            f"{trade.account_id}",
                            trade.symbol,
                            trade.side.upper(),
                            f"{trade.entry_price:.6f}",
                            f"{curr_price:.6f}",
                            f"{pnl:.5f}"
                        )
                    except:
                        continue

            console.print(f"[bold magenta]⚡ Live CPS: {cps:.5f} ¢/s ⚡ Total PnL: ${total_pnl:.5f}[/bold magenta]")
            live.update(table)
            time.sleep(2)

if __name__ == "__main__":
    console.print("[bold red]🚀 BLOODFIRE V36.5 FULL AGGRO MODE DEPLOYED 🚀[/bold red]")
    threading.Thread(target=blood_swarm, daemon=True).start()
    threading.Thread(target=monitor_trades, daemon=True).start()
    dashboard()
