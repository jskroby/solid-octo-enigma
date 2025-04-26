#!/usr/bin/env python3
# hyperbot_studio.py — Monolith: GA + Live Grid + Streamlit UI

import os, time, math, asyncio, signal, logging
from collections import deque
from threading import Thread
import ccxt.async_support as ccxt
import websockets
import json
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from prometheus_client import start_http_server, Gauge, Counter
from aiohttp import web
import streamlit as st
import plotly.express as px
from rich.console import Console

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("GATE_API_KEY","")
API_SECRET   = os.getenv("GATE_API_SECRET","")
SYMBOLS      = st.sidebar.text_input("Symbols", "BTC/USDT:USDT,ETH/USDT:USDT").split(",")
DEPOSIT      = st.sidebar.number_input("Deposit USD",  value=100.0)
LEVERAGE     = st.sidebar.number_input("Leverage",     value=50)
GRID_LVLS    = st.sidebar.slider("Grid Levels", 2, 10, 5)
BASE_SP_PCT  = st.sidebar.slider("Base Spread %", 0.05, 1.0, 0.1) / 100
SLIPP_PCT    = 0.05/100
FEE_TAKER    = 0.016/100
FEE_MAKER    = 0.015/100
TARGET_STEPS = [0.01,0.02,0.03]
LOCK_TIME    = st.sidebar.number_input("Lock Time (s)", value=10.0)
TUNE_INT     = st.sidebar.number_input("Tune Interval (s)", value=1.0)
LOOP_INT     = st.sidebar.number_input("Loop Interval (s)", value=0.5)

# GA params
TIMEFRAME    = st.sidebar.selectbox("Backtest TF", ["1m","5m"], index=0)
HISTORY      = st.sidebar.number_input("History Bars", 1000)
POP_SIZE     = st.sidebar.number_input("GA Pop Size", 20)
GENS         = st.sidebar.number_input("GA Generations", 10)
CXPB, MUTPB  = 0.6, 0.3
BUDGET       = 100.0
FEE_RT       = 0.07/100  # round-trip

console = Console()
logging.basicConfig(level=logging.INFO)

# Prometheus
cps_gauge = Gauge("cps", "Current ¢/s")
pnl_ctr   = Counter("profit", "Profit USD")
trade_ctr = Counter("trades", "Trades count")

SHUTDOWN = asyncio.Event()

# ────────────────────────────────────────────────────────────────────────────
# HEALTH AND METRICS
# ────────────────────────────────────────────────────────────────────────────
async def health(request): return web.Response(text="OK")
def start_http():
    app = web.Application()
    app.router.add_get("/healthz", health)
    web.run_app(app, port=8000)

Thread(target=lambda: start_http_server(8001), daemon=True).start()
Thread(target=start_http, daemon=True).start()

# ────────────────────────────────────────────────────────────────────────────
# GA BACKTESTER
# ────────────────────────────────────────────────────────────────────────────
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("gw",    np.random.uniform, 0.001, 0.02)
toolbox.register("tp",    np.random.uniform, 0.001, 0.02)
toolbox.register("eo",    np.random.uniform, 0.0005,0.01)
def make_ind(): return creator.Individual({"gw":toolbox.gw(),"tp":toolbox.tp(),"eo":toolbox.eo()})
toolbox.register("individual", make_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fetch_ws(symbol):
    uri = "wss://fx-ws.gateio.ws/ws/spot"
    async def _inner():
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "time":int(time.time()),
                "channel":"spot.candlesticks",
                "event":"subscribe",
                "payload":[symbol.replace("/","_"), str(int(TIMEFRAME[:-1])*60), str(HISTORY)]
            }))
            msg = await ws.recv()
            data = json.loads(msg)["result"]
            df = pd.DataFrame(data,columns=["ts","o","h","l","c","v"])
            df.ts = pd.to_datetime(df.ts,unit="s")
            return df.set_index("ts")["c"].astype(float)
    return asyncio.run(_inner())

def simulate(prices, ind):
    gw, tp, eo = ind["gw"],ind["tp"],ind["eo"]
    bal, pos, entry = BUDGET, 0, 0
    t0, t1 = 0,0
    for i,p in enumerate(prices):
        if pos==0:
            if np.random.rand()<0.5:
                entry = p*(1+eo); pos = bal/entry; bal=0; t0=i
        else:
            if p>=entry*(1+tp) or p<=entry*(1-gw):
                gross = pos*p
                fee   = gross*FEE_RT
                bal   = gross-fee; pos=0; entry=0; t1=i
    if pos>0:
        gross=pos*prices[-1]; bal=gross-gross*FEE_RT; t1=len(prices)-1
    pnl = bal-BUDGET
    return pnl/max(t1-t0,1)

def eval_ind(ind, prices):
    return (simulate(prices,ind),)

def ga_opt(symbol):
    console.print(f"▶️ Backtesting {symbol}")
    prices = fetch_ws(symbol)
    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("evaluate", lambda ind: eval_ind(ind,prices))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", lambda ind: ind.update(make_ind()))
    toolbox.register("select", tools.selBest)
    pop,_ = algorithms.eaSimple(pop,toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENS, verbose=False)
    best = tools.selBest(pop,1)[0]
    fit  = eval_ind(best,prices)[0]
    return best, fit

# ────────────────────────────────────────────────────────────────────────────
# LIVE GRID TRADER
# ────────────────────────────────────────────────────────────────────────────
class Stepped:
    def __init__(self):
        self.tp=0; 
        self.profit=0; self.t0=time.time()
        self.notional=(DEPOSIT*LEVERAGE)/len(SYMBOLS)
        self.step=0; self.lock=None
        self.grids=[]

    def record(self,gross):
        fee=gross*(FEE_MAKER+FEE_TAKER); slip=gross*SLIPP_PCT
        net=gross-fee-slip; self.profit+=net
        cps=self.profit/max(time.time()-self.t0,1e-6)
        cps_gauge.set(cps*100); pnl_ctr.inc(net); trade_ctr.inc()
        tgt=TARGET_STEPS[self.step]
        if cps>=tgt:
            if not self.lock: self.lock=time.time()
            elif time.time()-self.lock>=LOCK_TIME and self.step<len(TARGET_STEPS)-1:
                self.step+=1; self.lock=None
        else: self.lock=None
        factor=tgt/max(cps,1e-6)
        cap=DEPOSIT*LEVERAGE
        self.notional=max(1,min(self.notional*factor,cap))
        for g in self.grids:
            g.spread=max(BASE_SP_PCT*0.5,min(g.spread*math.sqrt(factor),BASE_SP_PCT*2))
        return cps*100

class Grid:
    def __init__(self,sym,e:Stepped):
        self.sym=sym; self.e=e; self.spread=BASE_SP_PCT; self.ex=None

    async def init(self):
        self.ex=ccxt.gateio({"apiKey":API_KEY,"secret":API_SECRET,"enableRateLimit":True,
                             "options":{"defaultType":"swap","defaultSettle":"usdt"}})
        await self.ex.set_leverage(LEVERAGE,self.sym)

    async def run(self):
        await self.init()
        while not SHUTDOWN.is_set():
            try:
                t=await self.ex.fetch_ticker(self.sym)
                mid=(t["bid"]+t["ask"])/2; size=self.e.notional/mid
                oids=[]
                for i in range(1,GRID_LVLS+1):
                    off=(self.spread/GRID_LVLS)*i
                    b=await self.ex.create_order(self.sym,"limit","buy", size,mid*(1-off),{"postOnly":True})
                    s=await self.ex.create_order(self.sym,"limit","sell",size,mid*(1+off),{"postOnly":True})
                    oids+= [b["id"],s["id"]]
                await asyncio.sleep(LOOP_INT)
                open_ids={o["id"] for o in await self.ex.fetch_open_orders(self.sym)}
                for oid in oids:
                    if oid not in open_ids:
                        gross=size*(self.spread/GRID_LVLS)
                        self.e.record(gross)
                await self.ex.cancel_all_orders(self.sym)
            except: await asyncio.sleep(1)
        await self.ex.close()

# ────────────────────────────────────────────────────────────────────────────
# BALANCE PRINTER
# ────────────────────────────────────────────────────────────────────────────
async def print_bal():
    ex=ccxt.gateio({"apiKey":API_KEY,"secret":API_SECRET,"enableRateLimit":True,
                   "options":{"defaultType":"swap","defaultSettle":"usdt"}})
    while not SHUTDOWN.is_set():
        b=await ex.fetch_balance(); st.write(f"**Balance:** {b['total'].get('USDT',0):.4f} USDT")
        await asyncio.sleep(30)

# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────
st.title("🧠 HyperBot Studio")

if st.button("Run GA Backtest"):
    results=[]
    for s in SYMBOLS:
        ind,fit=ga_opt(s)
        results.append((s,ind,fit))
    df=pd.DataFrame([{
        "Symbol": sym,
        "Grid %": f"{ind['gw']*100:.3f}",
        "TP %":   f"{ind['tp']*100:.3f}",
        "Offset %":f"{ind['eo']*100:.3f}",
        "Fitness ¢/s":f"{fit*100:.4f}"
    } for sym,ind,fit in results])
    st.dataframe(df)
    st.plotly_chart(px.bar(df,x="Symbol",y="Fitness ¢/s"))

if st.button("Launch Live Grid"):
    engine=Stepped()
    async def live():
        tasks=[asyncio.create_task(Grid(s,engine).run()) for s in SYMBOLS]
        tasks.append(asyncio.create_task(print_bal()))
        while not SHUTDOWN.is_set(): await asyncio.sleep(TUNE_INT)
        for t in tasks: t.cancel()
    asyncio.run(live())

# ────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ────────────────────────────────────────────────────────────────────────────
for sig in (signal.SIGINT,signal.SIGTERM):
    signal.signal(sig, lambda *_: SHUTDOWN.set())
