"""
tests/test_local_telemetry.py
Integration tests for the Zero-Cloud Local Telemetry Server (WebSocket Gateway).
Tests client registration, telemetry packet broadcasting, and GNSS blackout command injection.
"""

import os
import sys
import json
import asyncio
import unittest
import websockets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.telemetry_server import LocalTelemetryServer


class TestLocalTelemetryServer(unittest.IsolatedAsyncioTestCase):

    async def test_telemetry_packet_broadcast(self):
        """Tests that telemetry packets sent from a simulated phone are received by the dashboard."""
        port = 8771
        server = LocalTelemetryServer(host="127.0.0.1", ws_port=port, http_port=8091)
        uri = f"ws://127.0.0.1:{port}/telemetry"

        async with websockets.serve(server.register, "127.0.0.1", port):
            await asyncio.sleep(0.05)
            
            async with websockets.connect(uri) as phone_ws, websockets.connect(uri) as dashboard_ws:
                test_packet = {
                    "device_id": "TEST_PHONE_01",
                    "type": "navigationState",
                    "sequence_num": 1,
                    "timestamp_ms": 1700000000000,
                    "navigation": {
                        "timestamp_s": 10.5,
                        "latitude": 12.9716,
                        "longitude": 77.5946,
                        "pos_east_m": 15.2,
                        "pos_north_m": 34.8,
                        "speed_mps": 14.5,
                        "heading_deg": 45.0,
                        "confidence_pct": 95.0,
                        "gnss_mode": "GNSS_NORMAL",
                        "source": "GNSS_AI_IMU_EKF"
                    }
                }

                # Send from phone
                await phone_ws.send(json.dumps(test_packet))

                # Dashboard should receive the broadcasted packet
                recv_msg = await asyncio.wait_for(dashboard_ws.recv(), timeout=2.0)
                received_data = json.loads(recv_msg)

                self.assertEqual(received_data["device_id"], "TEST_PHONE_01")
                self.assertEqual(received_data["type"], "navigationState")
                self.assertEqual(received_data["navigation"]["speed_mps"], 14.5)
                self.assertEqual(received_data["navigation"]["gnss_mode"], "GNSS_NORMAL")


if __name__ == "__main__":
    unittest.main()
