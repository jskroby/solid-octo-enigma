#!/usr/bin/env python3
# hyperbot_monolith_healed.py — Production‑grade monolith with self‑tuned 3¢/s stepping and robust error handling (with debug logs)

import os
import sys
import time
import math
import asyncio
import logging
import signal
from threading import Thread
from collections import deque

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection, InvalidOrder, AuthenticationError
from prometheus_client import start_http_server, Gauge, Counter
from aiohttp import web
from rich.console import Console
from rich.live import Live
from rich.table import Table

# ────────────────────────────────────────────────────────────────────────────
# CONFIG & LOGGING
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hyperbot_monolith")
console = Console()

API_KEY    = os.getenv("GATE_API_KEY")
API_SECRET = os.getenv("GATE_API_SECRET")
if not (API_KEY and API_SECRET):
    logger.critical("Missing API_KEY or API_SECRET in environment")
    sys.exit(1)

SYMBOL_POOL       = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT").split(",")
DEPOSIT_USD       = float(os.getenv("DEPOSIT_USD", 100.0))
LEVERAGE          = int(os.getenv("LEVERAGE", 50))
TARGET_CPS_CENTS  = float(os.getenv("TARGET_CPS_CENTS", 3.0))  # 3 ¢/s
LOCK_TIME_SEC     = float(os.getenv("LOCK_TIME_SEC", 10))
TUNE_INTERVAL_SEC = float(os.getenv("TUNE_INTERVAL_SEC", 1))
LOOP_INTERVAL_SEC = float(os.getenv("LOOP_INTERVAL_SEC", 0.5))
MIN_VOL_USD_24H   = float(os.getenv("MIN_VOL_USD_24H", 50000))

# Prometheus metrics
cps_gauge   = Gauge("hyperbot_cps_cents", "Current cents per second")
pnl_counter = Counter("hyperbot_profit_usd", "Total profit in USD")
trade_counter = Counter("hyperbot_trades", "Total trades executed")

SHUTDOWN = asyncio.Event()

# Health endpoint
async def health(request):
    logger.debug("Health check received")
    return web.Response(text="OK")

def start_http():
    app = web.Application()
    app.router.add_get("/healthz", health)
    web.run_app(app, port=8000)

Thread(target=start_http, daemon=True).start()
start_http_server(8001)
logger.info("HTTP and Prometheus endpoints started")

# ────────────────────────────────────────────────────────────────────────────
# SAFE CCXT WRAPPERS
# ────────────────────────────────────────────────────────────────────────────
async def safe_fetch_ticker(ex, symbol):
    logger.debug(f"Fetching ticker for {symbol}")
    try:
        ticker = await ex.fetch_ticker(symbol)
        logger.debug(f"Received ticker for {symbol}: bid={ticker['bid']} ask={ticker['ask']}")
        return ticker
    except (NetworkError, RequestTimeout) as e:
        logger.warning(f"[{symbol}] network issue, retrying: {e}")
        await asyncio.sleep(1)
        return await safe_fetch_ticker(ex, symbol)
    except (ExchangeNotAvailable, DDoSProtection) as e:
        logger.error(f"[{symbol}] exchange unavailable: {e}")
        await asyncio.sleep(5)
        return await safe_fetch_ticker(ex, symbol)
    except AuthenticationError:
        logger.critical(f"[{symbol}] authentication failed, exiting")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"[{symbol}] unexpected error fetching ticker: {e}")
        raise

async def safe_create_order(ex, symbol, side, amount, price):
    logger.debug(f"Placing {side} order for {symbol}: amount={amount}, price={price}")
    try:
        order = await ex.create_order(symbol, 'limit', side, amount, price, {'postOnly': True})
        logger.info(f"Order placed {side} {symbol}: id={order['id']}")
        return order
    except InvalidOrder as e:
        logger.warning(f"[{symbol}] invalid order: {e}")
        return None
    except Exception as e:
        logger.error(f"[{symbol}] error placing order: {e}")
        return None

async def filter_active_symbols(ex, symbols):
    logger.debug(f"Filtering symbols based on volume: {symbols}")
    tickers = await ex.fetch_tickers(symbols)
    good = []
    for sym, t in tickers.items():
        vol = t.get('quoteVolume', 0)
        if vol >= MIN_VOL_USD_24H:
            good.append(sym)
        else:
            logger.info(f"Skipping {sym}, low 24h volume: ${vol:.0f}")
    logger.info(f"Active symbols after filtering: {good}")
    return good

# ────────────────────────────────────────────────────────────────────────────
# STEPPED ENGINE WITH CPS LOCKING (in cents/sec)
# ────────────────────────────────────────────────────────────────────────────
class SteppedEngine:
    def __init__(self, symbols):
        self.symbols = symbols
        self.deposit = DEPOSIT_USD
        self.profit = 0.0
        self.t0 = time.time()
        self.head_notional = (DEPOSIT_USD * LEVERAGE) / len(symbols)
        self.step = 0
        self.lock_start = None
        logger.info(f"Engine initialized: deposit={self.deposit}, head_notional={self.head_notional}")

    def calc_cps(self) -> float:
        elapsed = max(time.time() - self.t0, 1e-6)
        cps = (self.profit / elapsed) * 100
        return cps

    def record_profit(self, gross: float):
        logger.debug(f"Recording profit: gross={gross}")
        self.profit += gross
        cps = self.calc_cps()
        cps_gauge.set(cps)
        pnl_counter.inc(gross)
        trade_counter.inc()
        logger.info(f"Profit recorded: total_profit={self.profit:.6f}, cps={cps:.3f}¢/s")
        self._check_step(cps)

    def _check_step(self, cps: float):
        target = [1, 2, 3][self.step]
        logger.debug(f"Checking step: current cps={cps:.3f}, target={target}¢/s, step={self.step}")
        if cps >= target:
            if self.lock_start is None:
                self.lock_start = time.time()
                logger.debug(f"Lock start time set for target {target}¢/s")
            elif time.time() - self.lock_start >= LOCK_TIME_SEC and self.step < 2:
                self.step += 1
                self.lock_start = None
                logger.info(f"🔒 Locked {target}¢/s → advancing to {(self.step+1)}¢/s")
        else:
            self.lock_start = None

        factor = target / max(cps, 1e-6)
        cap = DEPOSIT_USD * LEVERAGE
        new_notional = max(1.0, min(self.head_notional * math.sqrt(factor), cap))
        logger.debug(f"Tuning notional: old={self.head_notional}, new={new_notional}")
        self.head_notional = new_notional

    def get_balance(self) -> float:
        balance = self.deposit + self.profit
        logger.debug(f"Current balance: {balance}")
        return balance

# ────────────────────────────────────────────────────────────────────────────
# GRID WORKER
# ────────────────────────────────────────────────────────────────────────────
class GridWorker:
    def __init__(self, symbol: str, engine: SteppedEngine):
        self.symbol = symbol
        self.engine = engine
        self.spread_pct = float(os.getenv("BASE_SPREAD_PCT", 0.001))
        self.ex = None
        logger.info(f"GridWorker created for {symbol}")

    async def init_exchange(self):
        self.ex = ccxt.gateio({
            'apiKey': API_KEY, 'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
        })
        await self.ex.load_markets()
        await self.ex.set_leverage(LEVERAGE, self.symbol)
        logger.info(f"Exchange initialized for {self.symbol}")

    async def run(self):
        await self.init_exchange()
        while not SHUTDOWN.is_set():
            try:
                ticker = await safe_fetch_ticker(self.ex, self.symbol)
                mid = (ticker['bid'] + ticker['ask']) / 2
                size = self.engine.head_notional / mid
                logger.debug(f"Placing grid for {self.symbol}: size={size}, spread_pct={self.spread_pct}")
                buy = await safe_create_order(self.ex, self.symbol, 'buy', size, mid*(1-self.spread_pct))
                sell = await safe_create_order(self.ex, self.symbol, 'sell', size, mid*(1+self.spread_pct))
                if buy and sell:
                    await asyncio.sleep(LOOP_INTERVAL_SEC)
                    open_ids = {o['id'] for o in await self.ex.fetch_open_orders(self.symbol)}
                    if buy['id'] not in open_ids and sell['id'] not in open_ids:
                        gross = size * self.spread_pct * mid
                        logger.debug(f"Orders filled for {self.symbol}, gross profit={gross}")
                        self.engine.record_profit(gross)
                await self.ex.cancel_all_orders(self.symbol)
                await asyncio.sleep(LOOP_INTERVAL_SEC)
            except Exception as e:
                logger.warning(f"[{self.symbol}] run error: {e}")
                await asyncio.sleep(1)
        await self.ex.close()
        logger.info(f"GridWorker stopped for {self.symbol}")

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
async def main():
    logger.info("Starting main()")
    raw_ex = ccxt.gateio({
        'apiKey': API_KEY, 'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType':'swap','defaultSettle':'usdt'}
    })
    ex = await raw_ex.load_markets()
    symbols = await filter_active_symbols(ex, SYMBOL_POOL)
    logger.info(f"Filtered symbols: {symbols}")
    engine = SteppedEngine(symbols)
    tasks = [asyncio.create_task(GridWorker(s, engine).run()) for s in symbols]
    async def monitor_cps():
        with Live(console=console, refresh_per_second=1) as live:
            while not SHUTDOWN.is_set():
                table = Table()
                table.add_column("Symbol")
                table.add_column("Balance")
                table.add_column("CPS ¢/s")
                for sym in symbols:
                    table.add_row(sym, f"${engine.get_balance():.2f}", f"{engine.calc_cps():.2f}")
                live.update(table)
                await asyncio.sleep(1)
    tasks.append(asyncio.create_task(monitor_cps()))
    await asyncio.gather(*tasks)

# signal handling
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: SHUTDOWN.set())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down hyperbot_monolith...")

