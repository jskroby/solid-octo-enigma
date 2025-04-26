#!/usr/bin/env python3
# hyperbot_monolith_cartman.py — Farm, Skim, Stack BTC, Save Ledger, Survive.

import os, sys, time, math, asyncio, logging, signal, json
from aiohttp import web
from prometheus_client import start_http_server, Gauge, Counter
import ccxt.async_support as ccxt
from ccxt.base.errors import *
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import print as rprint
from datetime import datetime

# ────────────────────────────────────────────────────────────────────────────
# CONFIG & LOGGING
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hyperbot_cartman")
console = Console()

API_KEY = os.getenv("GATE_API_KEY")
API_SECRET = os.getenv("GATE_API_SECRET")
if not (API_KEY and API_SECRET):
    logger.critical("Missing API keys, fatty.")
    sys.exit(1)

SYMBOL_POOL = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT").split(",")
DEPOSIT_USD = float(os.getenv("DEPOSIT_USD", 100.0))
LEVERAGE = int(os.getenv("LEVERAGE", 50))
LOOP_INTERVAL_SEC = float(os.getenv("LOOP_INTERVAL_SEC", 0.5))
LOCK_TIME_SEC = float(os.getenv("LOCK_TIME_SEC", 10))
TUNE_INTERVAL_SEC = float(os.getenv("TUNE_INTERVAL_SEC", 1))
MIN_VOL_USD_24H = float(os.getenv("MIN_VOL_USD_24H", 50000))
SKIM_INTERVAL_SEC = float(os.getenv("SKIM_INTERVAL_SEC", 300))
SKIM_PERCENT = 0.8
LEDGER_FILE = "profit_ledger.json"

cps_gauge = Gauge("hyperbot_cps_cents", "Current cents per second")
pnl_counter = Counter("hyperbot_profit_usd", "Total profit in USD")
trade_counter = Counter("hyperbot_trades", "Total trades executed")

SHUTDOWN = asyncio.Event()

# ────────────────────────────────────────────────────────────────────────────
# HTTP Health Server
# ────────────────────────────────────────────────────────────────────────────
async def health(request):
    return web.Response(text="OK")

async def start_http_servers():
    app = web.Application()
    app.router.add_get("/healthz", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, port=8000)
    await site.start()
    start_http_server(8001)
    logger.info("HTTP servers started")

# ────────────────────────────────────────────────────────────────────────────
# LEDGER MANAGEMENT
# ────────────────────────────────────────────────────────────────────────────
def load_ledger():
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "r") as f:
            return json.load(f)
    return {"total_btc_bought": 0.0, "total_usd_skimmed": 0.0, "last_skim_time": None}

def save_ledger(ledger):
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=4)

ledger = load_ledger()

# ────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ────────────────────────────────────────────────────────────────────────────
class SteppedEngine:
    def __init__(self, symbols, ex):
        self.symbols = symbols
        self.deposit = DEPOSIT_USD
        self.profit = 0.0
        self.t0 = time.time()
        self.head_notional = (DEPOSIT_USD * LEVERAGE) / len(symbols)
        self.ex = ex
        logger.info(f"Engine initialized: deposit={self.deposit}")

    def calc_cps(self):
        elapsed = max(time.time() - self.t0, 1e-6)
        return (self.profit / elapsed) * 100

    def record_profit(self, gross):
        logger.debug(f"Recording profit: gross={gross}")
        self.profit += gross
        cps_gauge.set(self.calc_cps())
        pnl_counter.inc(gross)
        trade_counter.inc()

    def get_balance(self):
        return self.deposit + self.profit

    async def skim_profits(self):
        if self.profit > 1.0:
            skim_amount = self.profit * SKIM_PERCENT
            if skim_amount < 5.0:
                logger.info(f"Skim amount too small (${skim_amount:.2f}), skipping.")
                return
            logger.info(f"Skimming ${skim_amount:.2f} to BTC")
            try:
                market = await self.ex.fetch_ticker('BTC/USDT')
                btc_price = market['last']
                btc_amount = skim_amount / btc_price
                await self.ex.create_market_buy_order('BTC/USDT', btc_amount)
                logger.info(f"Bought {btc_amount:.6f} BTC for ${skim_amount:.2f}")

                # Update ledger
                ledger["total_btc_bought"] += btc_amount
                ledger["total_usd_skimmed"] += skim_amount
                ledger["last_skim_time"] = datetime.utcnow().isoformat() + "Z"
                save_ledger(ledger)

                rprint("[bold green]💸💸 SKIM SUCCESS! BTC STACKED! 💸💸[/bold green]")
                rprint("[bold yellow]🚀🚀 PROFIT STOLEN FROM THE MARKET 🚀🚀[/bold yellow]")

            except Exception as e:
                logger.error(f"Skimming failed: {e}")
                return
            self.deposit += self.profit * (1 - SKIM_PERCENT)
            self.profit = 0.0

# ────────────────────────────────────────────────────────────────────────────
# GRID WORKER
# ────────────────────────────────────────────────────────────────────────────
class GridWorker:
    def __init__(self, symbol, engine):
        self.symbol = symbol
        self.engine = engine
        self.spread_pct = float(os.getenv("BASE_SPREAD_PCT", 0.001))
        self.ex = engine.ex

    async def run(self):
        await self.ex.set_leverage(LEVERAGE, self.symbol)
        while not SHUTDOWN.is_set():
            try:
                ticker = await self.ex.fetch_ticker(self.symbol)
                mid = (ticker['bid'] + ticker['ask']) / 2
                size = self.engine.head_notional / mid
                buy = await self.safe_create_order('buy', size, mid*(1-self.spread_pct))
                sell = await self.safe_create_order('sell', size, mid*(1+self.spread_pct))
                if buy and sell:
                    await asyncio.sleep(LOOP_INTERVAL_SEC)
                    open_ids = {o['id'] for o in await self.ex.fetch_open_orders(self.symbol)}
                    if buy['id'] not in open_ids and sell['id'] not in open_ids:
                        gross = size * self.spread_pct * mid
                        self.engine.record_profit(gross)
                await self.ex.cancel_all_orders(self.symbol)
                await asyncio.sleep(LOOP_INTERVAL_SEC)
            except Exception as e:
                logger.warning(f"[{self.symbol}] Error: {e}")
                await asyncio.sleep(1)
        await self.ex.close()

    async def safe_create_order(self, side, amount, price):
        try:
            order = await self.ex.create_order(self.symbol, 'limit', side, amount, price, {'postOnly': True})
            return order
        except InvalidOrder as e:
            logger.warning(f"[{self.symbol}] Invalid Order: {e}")
            return None
        except Exception as e:
            logger.error(f"[{self.symbol}] Order error: {e}")
            return None

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
async def main():
    await start_http_servers()

    ex = ccxt.gateio({
        'apiKey': API_KEY, 'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
    })
    await ex.load_markets()

    tickers = await ex.fetch_tickers(SYMBOL_POOL)
    symbols = [s for s in SYMBOL_POOL if tickers.get(s, {}).get('quoteVolume', 0) >= MIN_VOL_USD_24H]
    logger.info(f"Filtered active symbols: {symbols}")

    engine = SteppedEngine(symbols, ex)
    workers = [GridWorker(s, engine) for s in symbols]

    tasks = [asyncio.create_task(w.run()) for w in workers]

    async def monitor_and_skim():
        with Live(console=console, refresh_per_second=5) as live:
            prev_profit = 0.0
            while not SHUTDOWN.is_set():
                table = Table(title="💰 Hyperbot Profit Engine 💰", style="bold green")
                table.add_column("Symbol", justify="left")
                table.add_column("Balance", justify="right")
                table.add_column("CPS (¢/s)", justify="right")
                table.add_column("Profit (Δ)", justify="right")

                current_profit = engine.get_balance() - DEPOSIT_USD
                profit_delta = current_profit - prev_profit

                for sym in symbols:
                    table.add_row(
                        sym,
                        f"${engine.get_balance():.2f}",
                        f"{engine.calc_cps():.2f}",
                        f"{'+' if profit_delta >= 0 else ''}{profit_delta:.4f} USD"
                    )

                live.update(table)
                prev_profit = current_profit
                await asyncio.sleep(1)

    tasks.append(asyncio.create_task(monitor_and_skim()))

    async def skimmer():
        while not SHUTDOWN.is_set():
            await asyncio.sleep(SKIM_INTERVAL_SEC)
            await engine.skim_profits()

    tasks.append(asyncio.create_task(skimmer()))

    await asyncio.gather(*tasks)

# ────────────────────────────────────────────────────────────────────────────
# SHUTDOWN HANDLER
# ────────────────────────────────────────────────────────────────────────────
def shutdown_handler():
    logger.info("Shutdown requested, setting SHUTDOWN event.")
    SHUTDOWN.set()

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Hyperbot Cartman shutting down... later, fatass.")
