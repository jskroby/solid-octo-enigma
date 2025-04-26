import os, time, threading, json, sqlite3
import ccxt
import numpy as np
from sklearn.neural_network import MLPRegressor
from dotenv import load_dotenv
from rich.table import Table
from rich.live import Live
from rich.console import Console

# === INIT ===
load_dotenv()
console = Console()

ACCOUNTS = [
    {"apiKey": os.getenv("GATEIO_API_KEY"), "secret": os.getenv("GATE_API_SECRET")},
    {"apiKey": os.getenv("GATEIO_API_KEY"), "secret": os.getenv("GATE_API_SECRET")},
]

DB_FILE = "blood_trades_v3.db"
PAIR_CANDIDATES = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT"]
LEVERAGE = 5
MIN_TRADE_USD = 1.0
GROWTH_RATE = 1.01
PAIR_REFRESH = 300  # seconds
ORDER_SPACING = 0.0015  # tighter sniper grid

# === DB ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id TEXT, account INTEGER, symbol TEXT, side TEXT,
        entry_price REAL, exit_price REAL, size REAL,
        entry_time REAL, exit_time REAL, pnl REAL
    )""")
    conn.commit()
    conn.close()

def log_trade(trade):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (trade['id'], trade['account'], trade['symbol'], trade['side'],
               trade['entry'], trade['exit'], trade['size'],
               trade['entry_time'], trade['exit_time'], trade['pnl']))
    conn.commit()
    conn.close()

# === ML NeuralNet Scoring ===
class PairScorer:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500)
        self.train_dummy()

    def train_dummy(self):
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        self.model.fit(X, y)

    def score(self, volatility, spread, volume):
        return self.model.predict([[volatility, spread, volume]])[0]

# === BLOOD BOT ===
class BloodBot:
    def __init__(self, idx, creds):
        self.idx = idx
        self.apiKey = creds["apiKey"]
        self.secret = creds["secret"]
        self.console = Console()
        self.exchange = ccxt.gateio({
            'apiKey': self.apiKey,
            'secret': self.secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        self.exchange.load_markets()
        self.start_time = time.time()
        self.total_pnl = 0
        self.trades = []
        self.active_symbol = PAIR_CANDIDATES[0]

    def refresh_pair(self, scorer):
        best_score = -9999
        best_pair = self.active_symbol

        for pair in PAIR_CANDIDATES:
            try:
                ticker = self.exchange.fetch_ticker(pair)
                spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                vol = ticker['baseVolume']
                returns = np.random.uniform(0.001, 0.02)  # simulate volatility
                score = scorer.score(returns, spread, vol)
                if score > best_score:
                    best_score = score
                    best_pair = pair
            except Exception as e:
                self.console.print(f"[red]Pair check error {pair}: {e}[/red]")

        self.active_symbol = best_pair
        self.console.print(f"[bold magenta]Account {self.idx}: Switched to {self.active_symbol} (score {best_score:.4f})[/bold magenta]")

    def execute_trade(self):
        try:
            ticker = self.exchange.fetch_ticker(self.active_symbol)
            mid_price = ticker['last']
            if not mid_price:
                return
            
            size = MIN_TRADE_USD / mid_price
            side = 'buy' if time.time() % 2 > 1 else 'sell'
            price = mid_price * (1 - ORDER_SPACING if side == "buy" else 1 + ORDER_SPACING)

            self.exchange.set_leverage(LEVERAGE, self.active_symbol.replace("_", "/"))

            order = self.exchange.create_limit_order(
                self.active_symbol.replace("_", "/"),
                side,
                size,
                price,
                params={"postOnly": True, "reduceOnly": False}
            )

            # Simulate instant fill for now
            entry = price
            exit = price
            pnl = (exit - entry) * size if side == "buy" else (entry - exit) * size

            trade = {
                "id": str(time.time_ns()),
                "account": self.idx,
                "symbol": self.active_symbol,
                "side": side,
                "entry": entry,
                "exit": exit,
                "size": size,
                "entry_time": time.time(),
                "exit_time": time.time(),
                "pnl": pnl
            }

            self.total_pnl += pnl
            self.trades.append(trade)
            log_trade(trade)

        except Exception as e:
            self.console.print(f"[red]Trade error: {e}[/red]")

    def cps(self):
        elapsed = max(1, time.time() - self.start_time)
        return (self.total_pnl / elapsed) * 100

# === MAIN ENGINE ===
def run_bot(bot, scorer):
    last_switch = time.time()
    while True:
        if time.time() - last_switch >= PAIR_REFRESH:
            bot.refresh_pair(scorer)
            last_switch = time.time()
        bot.execute_trade()
        time.sleep(2)

def render_dashboard(bots):
    table = Table(title="💀 BLOOD COLOSSUS V3: SWARM LIVE")
    table.add_column("Account", style="magenta")
    table.add_column("Active Pair", style="cyan")
    table.add_column("CPS (¢/s)", style="green")
    table.add_column("Total PnL ($)", style="yellow")

    for bot in bots:
        table.add_row(
            f"Acct #{bot.idx+1}",
            bot.active_symbol,
            f"{bot.cps():.4f}",
            f"${bot.total_pnl:.2f}"
        )

    return table

def main():
    init_db()
    scorer = PairScorer()
    bots = []

    for idx, acct in enumerate(ACCOUNTS):
        bot = BloodBot(idx, acct)
        bots.append(bot)
        threading.Thread(target=run_bot, args=(bot, scorer), daemon=True).start()

    with Live(render_dashboard(bots), refresh_per_second=2) as live:
        while True:
            live.update(render_dashboard(bots))
            time.sleep(2)

if __name__ == "__main__":
    console.print("[bold red]🚀 BLOOD COLOSSUS V3 ACTIVATED 🚀[/bold red]")
    main()
