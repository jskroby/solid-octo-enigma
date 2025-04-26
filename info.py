# spread_sniper_daemon.py

import websocket
import threading
import json
import time
import ccxt
from rich.console import Console
from rich.table import Table

console = Console()

# === Global State ===
spread_data = {}
symbols_list = []

# === Load Symbols ===
exchange = ccxt.gateio()
markets = exchange.load_markets()

# Only Spot + Futures (USDT)
for sym in markets.keys():
    if ('/USDT' in sym and (markets[sym]['type'] == 'spot' or markets[sym]['type'] == 'swap')):
        symbols_list.append(sym.replace('/', '_'))

console.print(f"[bold green]Tracking {len(symbols_list)} symbols...[/bold green]")

# === WebSocket Handler ===
def subscribe_ws(symbol):
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if "channel" in data and data["channel"] == "spot.order_book_update":
                bids = data["result"]["bids"]
                asks = data["result"]["asks"]
                if bids and asks:
                    bid = float(bids[0][0])
                    ask = float(asks[0][0])
                    spread = (ask - bid) / bid if bid else 0
                    spread_data[symbol] = spread
        except Exception as e:
            console.print(f"[red]WS Error: {e}[/red]")

    def on_open(ws):
        payload = {
            "time": int(time.time()),
            "channel": "spot.order_book_update",
            "event": "subscribe",
            "payload": [symbol, "100ms"]
        }
        ws.send(json.dumps(payload))

    ws = websocket.WebSocketApp(
        "wss://api.gateio.ws/ws/v4/",
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()

# === Start WebSockets ===
for sym in symbols_list:
    threading.Thread(target=subscribe_ws, args=(sym,), daemon=True).start()
    time.sleep(0.1)  # not to kill the server

# === Main Display ===
def show_spreads():
    while True:
        table = Table(title="🧠 Best Spread KPI Tracker")
        table.add_column("Symbol")
        table.add_column("Spread %")
        
        sorted_spreads = sorted(spread_data.items(), key=lambda x: x[1])

        for sym, spread in sorted_spreads[:15]:
            table.add_row(sym, f"{spread*100:.4f}%")

        console.clear()
        console.print(table)

        time.sleep(5)

if __name__ == "__main__":
    console.print("[bold yellow]🛰️ Spread Sniper Daemon Started...[/bold yellow]")
    show_spreads()
