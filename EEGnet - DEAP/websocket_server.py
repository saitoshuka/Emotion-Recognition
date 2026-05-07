import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)

# Set to store connected clients
clients = set()

async def register(websocket):
    clients.add(websocket)
    logging.info(f"Client connected: {websocket.remote_address}")

async def unregister(websocket):
    clients.remove(websocket)
    logging.info(f"Client disconnected: {websocket.remote_address}")

async def handler(websocket, path):
    # Register new client
    await register(websocket)
    try:
        # Main loop to handle messages from this client
        async for message in websocket:
            logging.info(f"Received message from {websocket.remote_address}: {message}")
            # Broadcast message to all other clients
            await broadcast(message, websocket)
    except websockets.exceptions.ConnectionClosedError:
        pass  # Connection closed unexpectedly
    finally:
        # Unregister client when done
        await unregister(websocket)

async def broadcast(message, sender):
    # Send message to all clients except the sender
    for client in clients:
        if client != sender:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosedError:
                logging.warning(f"Failed to send message to {client.remote_address}")

async def main():
    async with websockets.serve(handler, "localhost", 8767, ping_interval=30):
        logging.info("WebSocket server started on ws://localhost:8767")
        await asyncio.Future()  # Run forever or until interrupted

if __name__ == "__main__":
    asyncio.run(main())

