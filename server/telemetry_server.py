"""
server/telemetry_server.py
Zero-Cloud Local Telemetry & Visualization Gateway for SIH 2026 PS-168.
Serves the offline dashboard via HTTP and manages live real-time WebSocket telemetry.
Runs 100% locally with NO Internet required.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import http.server
import threading
import websockets
from typing import Set, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves local dashboard files (HTML, CSS, JS, tiles) without internet."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def log_message(self, format, *args):
        # Suppress routine static asset HTTP logs to keep console clean
        pass


class LocalTelemetryServer:
    """
    Local WebSocket & Telemetry Relay Gateway.
    Receives SensorFrame / NavigationState from Flutter phone and broadcasts to Live Dashboard.
    """
    def __init__(self, host: str = "0.0.0.0", ws_port: int = 8765, http_port: int = 8080):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port

        self.phone_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.dashboard_clients: Set[websockets.WebSocketServerProtocol] = set()

        self.latest_state: Dict[str, Any] = {}
        self.total_packets_received: int = 0
        self.is_simulated_blackout: bool = False

    async def register(self, websocket, path=None, *args, **kwargs):
        """Registers connected clients (phone vs dashboard)."""
        self.dashboard_clients.add(websocket)
        print(f"  [+] Client Connected: {websocket.remote_address} | Total Viewers: {len(self.dashboard_clients)}")

        # Send latest state immediately upon connection
        if self.latest_state:
            try:
                await websocket.send(json.dumps(self.latest_state))
            except Exception:
                pass

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.dashboard_clients.discard(websocket)
            self.phone_clients.discard(websocket)
            print(f"  [-] Client Disconnected: {websocket.remote_address} | Remaining: {len(self.dashboard_clients)}")

    async def handle_message(self, sender_ws, message: str):
        """Processes incoming telemetry packets or dashboard control commands."""
        try:
            payload = json.loads(message)
            msg_type = payload.get("type", "")

            # 1. Incoming Telemetry from Flutter Phone
            if msg_type in ["navigationState", "sensorFrame"]:
                self.phone_clients.add(sender_ws)
                self.total_packets_received += 1
                self.latest_state = payload

                # Broadcast to all connected dashboard viewers
                if self.dashboard_clients:
                    disconnected = set()
                    for client in list(self.dashboard_clients):
                        if client != sender_ws:
                            try:
                                await client.send(message)
                            except Exception:
                                disconnected.add(client)
                    self.dashboard_clients.difference_update(disconnected)

            # 2. Control Command from Dashboard (e.g. Trigger Demo Blackout)
            elif msg_type == "COMMAND_BLACKOUT_TOGGLE":
                self.is_simulated_blackout = payload.get("active", False)
                print(f"\n[Command] Simulated GNSS Blackout set to: {self.is_simulated_blackout}")
                
                # Forward blackout command to phone clients
                cmd_msg = json.dumps({
                    "type": "SET_SIMULATED_BLACKOUT",
                    "active": self.is_simulated_blackout
                })
                for phone in list(self.phone_clients):
                    try:
                        await phone.send(cmd_msg)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[Error] Failed to process message: {e}")

    def start_http_server(self):
        """Starts background multi-threaded HTTP server for dashboard."""
        server_address = (self.host, self.http_port)
        # Use ThreadingHTTPServer so simultaneous requests/downloads never block
        if hasattr(http.server, 'ThreadingHTTPServer'):
            httpd = http.server.ThreadingHTTPServer(server_address, DashboardHTTPRequestHandler)
        else:
            httpd = http.server.HTTPServer(server_address, DashboardHTTPRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        print(f"[*] Local Offline Dashboard served at: http://localhost:{self.http_port}/ (or http://{self.host}:{self.http_port}/)")

    async def run(self):
        self.start_http_server()
        print(f"[*] Local WebSocket Telemetry Gateway listening on: ws://{self.host}:{self.ws_port}/telemetry")
        print(f"[*] ZERO INTERNET REQUIRED — Local LAN / Hotspot Ready.\n")

        async with websockets.serve(self.register, self.host, self.ws_port):
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="SIH 2026 Local Telemetry Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    parser.add_argument("--ws_port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--http_port", type=int, default=8080, help="HTTP dashboard port")
    args = parser.parse_args()

    print("=" * 80)
    print("SIH 2026 PS-168: LOCAL TELEMETRY & OFFLINE DASHBOARD GATEWAY")
    print("=" * 80)

    server = LocalTelemetryServer(host=args.host, ws_port=args.ws_port, http_port=args.http_port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[!] Server stopped by user.")


if __name__ == "__main__":
    main()
