#!/usr/bin/env python3
# hyperbot_studio_v2_fixed.py — Monolith: GA + Live Grid + Streamlit UI (Fixed)

import os, time, math, asyncio, signal, logging
from threading import Thread
from collections import deque
import ccxt.async_support as ccxt
import websockets, json
import pandas as pd, numpy as np
from deap import base, creator, tools, algorithms
from prometheus_client import start_http_server, Gauge, Counter
from aiohttp import web
import streamlit as st
import plotly.express as px

# ────────────────────────────────────────────────────────────────────────────
# GLOBAL SETUP
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hyperbot_v2")
SHUTDOWN = asyncio.Event()
engine = None  # will hold our SteppedEngine instance

# Prometheus metrics
Gauge("cps", "¢/s").set_function(lambda: engine.calc_cps()*100 if engine else 0)
pnl_ctr = Counter("profit", "USD profit")
trade_ctr = Counter("trades", "trade count")
Thread(target=lambda: start_http_server(8001), daemon=True).start()

# Health endpoint
async def health(req):
    return web.Response(text="OK")
def start_http():
    app = web.Application()
    app.router.add_get("/healthz", health)
    web.run_app(app, port=8000)
Thread(target=start_http, daemon=True).start()

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📐 Strategy & UI Settings")
SYMBOLS     = st.sidebar.text_input("Symbols",     "BTC/USDT:USDT,ETH/USDT:USDT").split(",")
DEPOSIT     = st.sidebar.number_input("Deposit USD", 100.0, step=10.0)
LEVERAGE    = st.sidebar.number_input("Leverage",    50,    step=1)
GRID_LVLS   = st.sidebar.slider("Grid Levels",  2, 12, 5)
BASE_SP_PCT = st.sidebar.slider("Base Spread %", 0.05, 1.0, 0.1)/100
SLIPP_PCT   = st.sidebar.slider("Slippage %",    0.01, 0.2, 0.05)/100
FEE_TAKER   = st.sidebar.slider("Taker Fee %",   0.005,0.05,0.016)/100
FEE_MAKER   = st.sidebar.slider("Maker Fee %",   0.005,0.05,0.015)/100

TUNE_INT    = st.sidebar.number_input("Tune Interval (s)", 1.0)
LOOP_INT    = st.sidebar.number_input("Loop Interval (s)", 0.5)

# GA parameters
st.sidebar.markdown("### 🔬 GA Backtest")
TIMEFRAME   = st.sidebar.selectbox("Backtest TF", ["1m","5m"], index=0)
HISTORY     = st.sidebar.number_input("History Bars", 500, step=100)
POP_SIZE    = st.sidebar.number_input("Pop Size", 20, step=5)
GENS        = st.sidebar.number_input("Generations", 10, step=1)
CXPB, MUTPB = 0.6, 0.3
BUDGET      = DEPOSIT
FEE_RT      = (FEE_TAKER + FEE_MAKER)

# Stepped CPS targets
STEPS       = [0.01, 0.02, 0.03]
LOCK_TIME   = st.sidebar.number_input("Step Lock Time (s)", 10.0)

# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT LAYOUT
# ────────────────────────────────────────────────────────────────────────────
st.title("🧠 HyperBot Studio V2 (Fixed)")
tab1, tab2 = st.tabs(["GA Backtest", "Live Grid"])

# ────────────────────────────────────────────────────────────────────────────
# GA BACKTEST TAB
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    if st.button("▶️ Run GA Backtest"):
        df_results = pd.DataFrame(columns=["Symbol","Grid%","TP%","Offset%","Fitness ¢/s"])
        progress = st.progress(0)
        fitness_curve = []
        for idx, sym in enumerate(SYMBOLS):
            st.write(f"**Backtesting {sym}**")
            prices = fetch_ws_closes(sym, TIMEFRAME, HISTORY)
            best, fit, history = run_ga(prices, POP_SIZE, GENS, CXPB, MUTPB)
            df_results.loc[idx] = [
                sym,
                f"{best['gw']*100:.3f}",
                f"{best['tp']*100:.3f}",
                f"{best['eo']*100:.3f}",
                f"{fit*100:.3f}"
            ]
            fitness_curve.append(fit*100)
            progress.progress((idx+1)/len(SYMBOLS))
        st.dataframe(df_results, use_container_width=True)
        st.plotly_chart(px.bar(df_results, x="Symbol", y="Fitness ¢/s", title="GA Fitness per Symbol"))
        st.plotly_chart(px.line(y=fitness_curve, labels={"y":"Best Fitness ¢/s","index":"Symbol Index"}))

# ────────────────────────────────────────────────────────────────────────────
# LIVE GRID TAB
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    def launch_live_grid():
        global engine
        engine = SteppedEngine(STEPS, LOCK_TIME, DEPOSIT, LEVERAGE)
        streams = [
            GridWorker(sym, engine, BASE_SP_PCT, GRID_LVLS, LOOP_INT,
                       FEE_TAKER, FEE_MAKER, SLIPP_PCT)
            for sym in SYMBOLS
        ]
        Thread(target=lambda: asyncio.run(start_live(streams, engine)), daemon=True).start()

    if st.button("▶️ Launch Live Grid"):
        launch_live_grid()

    cps_metric = st.empty()
    pnl_metric = st.empty()
    bal_text   = st.empty()

    if engine:
        def live_update():
            while engine:
                cps = engine.calc_cps()*100
                pnl = engine.profit
                bal = engine.get_balance()
                cps_metric.metric("CPS (¢/s)", f"{cps:.3f}")
                pnl_metric.metric("Total PnL (USD)", f"{pnl:.2f}")
                bal_text.write(f"**Balance:** {bal:.4f} USDT")
                time.sleep(1)
        Thread(target=live_update, daemon=True).start()

# ────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES & CLASSES
# ────────────────────────────────────────────────────────────────────────────
def fetch_ws_closes(symbol, tf, limit):
    uri = "wss://fx-ws.gateio.ws/ws/spot"
    async def _inner():
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "time": int(time.time()),
                "channel": "spot.candlesticks",
                "event":   "subscribe",
                "payload": [symbol.replace("/","_"), str(int(tf[:-1])*60), str(limit)]
            }))
            data = json.loads(await ws.recv())["result"]
            df = pd.DataFrame(data, columns=["ts","o","h","l","c","v"])
            df.ts = pd.to_datetime(df.ts, unit="s")
            return df.set_index("ts")["c"].astype(float)
    return asyncio.run(_inner())

def run_ga(prices, pop, gens, cxpb, mutpb):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,), overwrite=True)
    creator.create("Individual", dict, fitness=creator.FitnessMax, overwrite=True)
    tb = base.Toolbox()
    tb.register("gw", np.random.uniform,    0.001, 0.02)
    tb.register("tp", np.random.uniform,    0.001, 0.02)
    tb.register("eo", np.random.uniform, 0.0005, 0.01)
    def make(): return creator.Individual({"gw":tb.gw(),"tp":tb.tp(),"eo":tb.eo()})
    tb.register("individual", make)
    tb.register("population", tools.initRepeat, list, make)
    pop = tb.population(n=pop)
    history = []
    def evalfn(ind):
        fit = simulate(prices, ind)
        return (fit,)
    tb.register("evaluate", evalfn)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", lambda ind: ind.update(make()))
    tb.register("select", tools.selBest)
    for _ in range(gens):
        offspring,_ = algorithms.varAnd(pop, tb, cxpb, mutpb)
        fits = list(map(evalfn, offspring))
        for ind,fit in zip(offspring, fits): ind.fitness.values = fit
        pop = tb.select(offspring, k=len(pop))
        history.append(tools.selBest(pop,1)[0].fitness.values[0]*100)
    best = tools.selBest(pop,1)[0]
    return best, best.fitness.values[0], history

def simulate(prices, ind):
    arr = prices.values
    gw,tp,eo = ind["gw"],ind["tp"],ind["eo"]
    bal,pos,entry = BUDGET,0.0,0.0
    t0,t1 = 0,0
    for i,p in enumerate(arr):
        if pos==0 and np.random.rand()<0.5:
            entry,pos = p*(1+eo), bal/(p*(1+eo)); bal=0; t0=i
        elif pos>0 and (p>=entry*(1+tp) or p<=entry*(1-gw)):
            gross = pos*p; fee=gross*FEE_RT
            bal = gross-fee; pos=0; entry=0; t1=i
    if pos>0:
        gross=pos*arr[-1]; bal=gross-gross*FEE_RT; t1=len(arr)-1
    pnl = bal - BUDGET
    return pnl/max(t1-t0,1)

class SteppedEngine:
    def __init__(self, steps, lock_time, deposit, lev):
        self.steps      = steps
        self.lock_time  = lock_time
        self.deposit    = deposit          # store initial deposit
        self.profit     = 0.0
        self.t0         = time.time()
        self.notional   = (deposit * lev) / len(SYMBOLS)
        self.step       = 0
        self.lock       = None

    def calc_cps(self):
        return self.profit / max(time.time() - self.t0, 1e-6)

    def record(self, gross):
        fee  = gross * (FEE_TAKER + FEE_MAKER)
        slip = gross * SLIPP_PCT
        net  = gross - fee - slip
        self.profit += net
        trade_ctr.inc()
        cps = self.calc_cps()
        tgt = self.steps[self.step]
        if cps >= tgt:
            if not self.lock:
                self.lock = time.time()
            elif time.time() - self.lock >= self.lock_time and self.step < len(self.steps) - 1:
                self.step += 1
                self.lock = None
        else:
            self.lock = None
        factor = tgt / max(cps, 1e-6)
        cap    = DEPOSIT * LEVERAGE
        self.notional = max(1, min(self.notional * factor, cap))

    def get_balance(self) -> float:
        """
        Returns current USDT-equivalent balance: initial deposit + net PnL.
        """
        return self.deposit + self.profit

class GridWorker:
    def __init__(self, sym, eng, base_sp, lvls, loop, ft, fm, sp):
        self.sym    = sym
        self.e      = eng
        self.spread = base_sp
        self.lvls   = lvls
        self.loop   = loop
        self.ft     = ft
        self.fm     = fm
        self.spct   = sp
        self.ex     = None

    async def run(self):
        self.ex = ccxt.gateio({
            "apiKey":API_KEY,"secret":API_SECRET,
            "enableRateLimit":True,
            "options":{"defaultType":"swap","defaultSettle":"usdt"}
        })
        await self.ex.set_leverage(LEVERAGE, self.sym)
        while not SHUTDOWN.is_set():
            try:
                t = await self.ex.fetch_ticker(self.sym)
                mid = (t["bid"] + t["ask"]) / 2
                size = self.e.notional / mid
                oids = []
                for i in range(1, self.lvls + 1):
                    off = (self.spread / self.lvls) * i
                    b = await self.ex.create_order(self.sym, "limit", "buy",  size, mid*(1-off), {"postOnly":True})
                    s = await self.ex.create_order(self.sym, "limit", "sell", size, mid*(1+off), {"postOnly":True})
                    oids += [b["id"], s["id"]]
                await asyncio.sleep(self.loop)
                open_ids = {o["id"] for o in await self.ex.fetch_open_orders(self.sym)}
                for oid in oids:
                    if oid not in open_ids:
                        gross = size * (self.spread / self.lvls)
                        self.e.record(gross)
                await self.ex.cancel_all_orders(self.sym)
            except Exception as e:
                logger.warning(f"{self.sym} error: {e}")
                await asyncio.sleep(1)
        await self.ex.close()

async def start_live(workers, engine):
    tasks = [asyncio.create_task(w.run()) for w in workers]
    await asyncio.gather(*tasks)

# Clean shutdown handlers
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: SHUTDOWN.set())
