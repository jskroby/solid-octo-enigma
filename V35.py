import ccxt
import torch
import random
import time
import threading
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.live import Live

# === CONFIG ===
API_KEYS = [{"apiKey": "4efbe203fbac0e4bcd0003a0910c801b", "secret": "8b0e9cf47fdd97f2a42bca571a74105c53a5f906cd4ada125938d05115d063a0"}]  # 🔥 Add more accounts for mega-blood mode
TARGET_USD_PER_TRADE = 5
LEVERAGE = 25
GROWTH_FACTOR = 1.05
HEDGE_THRESHOLD = 0.0025
VOLATILITY_THRESHOLD = 0.015
MODEL_ID = "openai-community/llama-3-tiny"

console = Console()
start_time = time.time()
lock = threading.Lock()
active_trades = []
total_pnl = 0

# === INIT MODELS ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
)

# === INIT EXCHANGES ===
exchanges = []
for keys in API_KEYS:
    ex = ccxt.gateio({
        'apiKey': keys['apiKey'],
        'secret': keys['secret'],
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    exchanges.append(ex)

# === TRADE CLASS ===
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

# === UTILS ===
def fetch_symbols(ex):
    return [s for s in ex.load_markets() if '/USDT:USDT' in s and ex.markets[s]['active']]

def volatility_score(symbol, ex):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, '1m', limit=12)
        closes = np.array([c[4] for c in ohlcv])
        returns = np.diff(np.log(closes))
        return np.std(returns)
    except:
        return 0

def predict_volatility(symbol, vol_now):
    input_text = f"Symbol {symbol} volatility {vol_now:.5f}. Predict next."
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    return outputs.logits.softmax(dim=1)[0][1].item()

def place_order(symbol, side, usd_amount, ex, account_id):
    try:
        ticker = ex.fetch_ticker(symbol)
        price = ticker['last']
        if not price:
            return
        price = price * (1 - 0.00015) if side == "buy" else price * (1 + 0.00015)
        size = round(usd_amount / price, 4)
        ex.create_limit_order(
            symbol.replace("_", "/"), side, size, price,
            params={"postOnly": True, "marginMode": "cross"}
        )
        console.print(f"[green]Placed {side.upper()} {symbol} {size} @ {price:.5f} (Acct {account_id})[/green]")
        with lock:
            active_trades.append(Trade(symbol, side, size, price, account_id))
    except Exception as e:
        console.print(f"[red]Order error: {e}[/red]")

# === BLOOD SWARM ===
def blood_swarm():
    global TARGET_USD_PER_TRADE
    while True:
        for idx, ex in enumerate(exchanges):
            try:
                symbols = fetch_symbols(ex)
                random.shuffle(symbols)
                for symbol in symbols[:10]:
                    vol_now = volatility_score(symbol, ex)
                    prediction = predict_volatility(symbol, vol_now)
                    if vol_now > VOLATILITY_THRESHOLD and prediction > 0.5:
                        side = random.choice(["buy", "sell"])
                        place_order(symbol, side, TARGET_USD_PER_TRADE, ex, idx)
            except Exception as e:
                console.print(f"[red]Swarm error: {e}[/red]")
        time.sleep(2)

# === TRADE MONITOR ===
def monitor_trades():
    global total_pnl, TARGET_USD_PER_TRADE
    while True:
        with lock:
            for trade in active_trades[:]:
                try:
                    ex = exchanges[trade.account_id]
                    ticker = ex.fetch_ticker(trade.symbol)
                    price = ticker['last']
                    delta = (price - trade.entry_price) / trade.entry_price
                    if trade.side == "sell":
                        delta = -delta
                    if abs(delta) >= HEDGE_THRESHOLD:
                        opposite = "buy" if trade.side == "sell" else "sell"
                        ex.create_market_order(
                            trade.symbol.replace("_", "/"), opposite, trade.size,
                            params={"reduceOnly": True, "marginMode": "cross"}
                        )
                        trade.exit_price = price
                        trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - trade.exit_price) * trade.size
                        total_pnl += trade.pnl
                        TARGET_USD_PER_TRADE *= GROWTH_FACTOR  # Grow after every win
                        active_trades.remove(trade)
                        console.print(f"[bold red]HEDGED {trade.symbol} for +${trade.pnl:.5f} profit[/bold red]")
                except Exception as e:
                    console.print(f"[red]Monitor error: {e}[/red]")
        time.sleep(0.4)

# === DASHBOARD ===
def dashboard():
    with Live(refresh_per_second=1) as live:
        while True:
            table = Table(title="💀 BLOODFIRE V36.5 - AGGRO LIVE 💀")
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
                            str(trade.account_id),
                            trade.symbol,
                            trade.side.upper(),
                            f"{trade.entry_price:.5f}",
                            f"{curr_price:.5f}",
                            f"{pnl:.5f}"
                        )
                    except:
                        continue
            console.print(f"[bold magenta]⚡ Live CPS: {cps:.5f} ¢/s ⚡ Total PnL: ${total_pnl:.5f}[/bold magenta]")
            live.update(table)
            time.sleep(2)

# === RUN BLOODFIRE ===
if __name__ == "__main__":
    console.print("[bold red]🚀 BLOODFIRE V36.5 DEPLOYED 🚀[/bold red]")
    threading.Thread(target=blood_swarm, daemon=True).start()
    threading.Thread(target=monitor_trades, daemon=True).start()
    dashboard()
