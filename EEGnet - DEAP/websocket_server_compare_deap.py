import asyncio
import json
import logging

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

clients = set()
latest = {
    "EEG": None,
    "Morphcast": None,
}


async def register(ws):
    clients.add(ws)
    logging.info(f"Client connected: {ws.remote_address}")


async def unregister(ws):
    if ws in clients:
        clients.remove(ws)
    logging.info(f"Client disconnected: {ws.remote_address}")


async def broadcast(payload, sender=None):
    msg = json.dumps(payload)
    to_remove = []
    for c in clients:
        if sender is not None and c == sender:
            continue
        try:
            await c.send(msg)
        except websockets.exceptions.ConnectionClosed:
            to_remove.append(c)
    for c in to_remove:
        await unregister(c)


async def handler(ws):
    await register(ws)
    try:
        await ws.send(json.dumps({"type": "state", "latest": latest}))
        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            source = str(data.get("source", "")).strip().lower()
            if source == "eeg":
                latest["EEG"] = data
            elif source == "morphcast":
                latest["Morphcast"] = data

            await broadcast({"type": "update", "data": data, "latest": latest}, sender=ws)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await unregister(ws)


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8767, ping_interval=30):
        logging.info("WebSocket compare server started on ws://localhost:8767")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
