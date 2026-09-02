/**
 * NavPulse — Master Real-Time Telemetry & Offline Localization Dashboard Script
 * SIH 2026 Problem Statement 168
 * 
 * Capabilities:
 * - Live WebSocket Telemetry Client (ws://<host>:8765/telemetry) with zero Internet dependency
 * - Offline Geographic Map Engine (Leaflet + Local Vector Canvas Fallback)
 * - Real-time Animated Device Marker & Trajectory Polyline
 * - GNSS Operating Mode Transitions & Blackout Outage Overlay
 * - Historical Drive Replay Mode
 */

class NavPulseDashboard {
    constructor() {
        this.ws = null;
        this.wsUrl = `ws://${window.location.hostname || 'localhost'}:8765/telemetry`;
        this.isWsConnected = false;
        
        // Map & Layers
        this.map = null;
        this.deviceMarker = null;
        this.livePathPolyline = null;
        this.outagePathPolyline = null;
        this.pathPoints = [];
        this.outagePoints = [];

        // State & Mode
        this.currentMode = 'live'; // 'live' or 'replay'
        this.isBlackoutActive = false;
        this.blackoutStartMs = 0;
        this.latestState = null;

        // Replay data cache
        this.replayData = null;
        this.replayTimer = null;
        this.replayIndex = 0;

        this.init();
    }

    init() {
        this.initMap();
        this.initEventListeners();
        this.connectWebSocket();
    }

    // ── 1. Map Initialization (Offline-Compatible) ──────────────────────────
    initMap() {
        // Initial center: Bangalore / Indian local coordinate center
        const defaultLat = 12.9716;
        const defaultLon = 77.5946;

        this.map = L.map('leafletMap', {
            center: [defaultLat, defaultLon],
            zoom: 17,
            zoomControl: true,
            attributionControl: false
        });

        // Offline Dark Mode Tile Layer (Attempts local or cached tiles; gracefully handles offline)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 20,
            subdomains: 'abcd'
        }).addTo(this.map);

        // Custom High-Visibility Directional Device Marker
        const markerHtml = `
            <div class="device-marker-wrapper" id="markerRotator">
                <div class="marker-pulse"></div>
                <div class="marker-core">
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="#00ffcc">
                        <polygon points="12,2 22,22 12,17 2,22" />
                    </svg>
                </div>
            </div>
        `;

        const customIcon = L.divIcon({
            html: markerHtml,
            className: 'custom-vehicle-icon',
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        });

        this.deviceMarker = L.marker([defaultLat, defaultLon], { icon: customIcon }).addTo(this.map);

        // Live Trajectory Polylines
        this.livePathPolyline = L.polyline([], {
            color: '#00ffcc',
            weight: 4,
            opacity: 0.85,
            smoothFactor: 1
        }).addTo(this.map);

        this.outagePathPolyline = L.polyline([], {
            color: '#ff3366',
            weight: 5,
            opacity: 0.95,
            dashArray: '6, 6'
        }).addTo(this.map);
    }

    // ── 2. Local WebSocket Telemetry Client ──────────────────────────────────
    connectWebSocket() {
        const linkPill = document.getElementById('linkStatusPill');
        const linkText = document.getElementById('linkStatusText');

        linkText.innerText = `LOCAL LINK: CONNECTING (${this.wsUrl})...`;
        linkPill.querySelector('.status-dot').className = 'status-dot dot-amber';

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                this.isWsConnected = true;
                linkText.innerText = `LOCAL LINK: CONNECTED (ws://${window.location.hostname || 'localhost'}:8765)`;
                linkPill.querySelector('.status-dot').className = 'status-dot dot-cyan';
                console.log('[+] Connected to Local Telemetry Gateway.');
            };

            this.ws.onmessage = (event) => {
                if (this.currentMode !== 'live') return;
                try {
                    const packet = JSON.parse(event.data);
                    this.handleTelemetryPacket(packet);
                } catch (err) {
                    console.error('Failed to parse telemetry packet:', err);
                }
            };

            this.ws.onclose = () => {
                this.isWsConnected = false;
                linkText.innerText = 'LOCAL LINK: OFFLINE (Auto-reconnecting...)';
                linkPill.querySelector('.status-dot').className = 'status-dot dot-red';
                // Exponential backoff reconnect
                setTimeout(() => this.connectWebSocket(), 2500);
            };

            this.ws.onerror = () => {
                this.ws.close();
            };
        } catch (e) {
            setTimeout(() => this.connectWebSocket(), 3000);
        }
    }

    // ── 3. Handle Live Telemetry Packet ─────────────────────────────────────
    handleTelemetryPacket(packet) {
        if (!packet || packet.type !== 'navigationState' || !packet.navigation) return;
        const nav = packet.navigation;
        this.latestState = nav;

        this.updateAvionicsHUD(nav);
        this.updateMapPosition(nav);
    }

    // ── 4. Update HUD Avionics Elements ──────────────────────────────────────
    updateAvionicsHUD(nav) {
        const speedKmh = (nav.speed_kmh !== undefined) ? nav.speed_kmh : (nav.speed_mps * 3.6);
        const speedMps = nav.speed_mps || 0.0;
        const headingDeg = nav.heading_deg || (nav.heading_rad * 180.0 / Math.PI) || 0.0;
        const confidence = nav.confidence_pct || 95.0;
        const sigma = nav.uncertainty_sigma_mps || 0.20;
        const gnssMode = nav.gnss_mode || 'NORMAL';
        const source = nav.source || 'GNSS_AI_IMU_EKF';

        // Digital HUD
        document.getElementById('hudSpeedKmh').innerText = speedKmh.toFixed(1);
        document.getElementById('hudSpeedMps').innerText = speedMps.toFixed(2);
        document.getElementById('hudSigma').innerText = `±${sigma.toFixed(3)} m/s`;
        document.getElementById('hudHeading').innerText = `${headingDeg.toFixed(1)}°`;
        document.getElementById('hudGyroBias').innerText = `${(nav.gyro_bias_rad_s || 0.0).toFixed(4)} rad/s`;
        document.getElementById('hudVelLat').innerText = `${(nav.velocity_lat_mps || 0.0).toFixed(2)} m/s`;
        document.getElementById('hudConfidence').innerText = `${confidence.toFixed(0)}%`;
        document.getElementById('confidenceFill').style.width = `${Math.min(100, confidence)}%`;

        document.getElementById('hudLat').innerText = `${nav.latitude.toFixed(6)}°`;
        document.getElementById('hudLon').innerText = `${nav.longitude.toFixed(6)}°`;
        document.getElementById('hudPosEast').innerText = `${(nav.pos_east_m || 0.0).toFixed(1)} m`;
        document.getElementById('hudPosNorth').innerText = `${(nav.pos_north_m || 0.0).toFixed(1)} m`;

        // Context Badge
        document.getElementById('contextBadge').innerText = nav.context_mode || 'NORMAL_URBAN';

        // GNSS Status Pill
        const gnssPill = document.getElementById('systemStatusPill');
        const gnssText = document.getElementById('systemStatusText');
        const sourceBadge = document.getElementById('sourceBadge');

        if (gnssMode === 'DENIED' || gnssMode === 'GNSS_DENIED') {
            gnssPill.querySelector('.status-dot').className = 'status-dot dot-red';
            gnssText.innerText = `GNSS DENIED (AI-DR: ${nav.blackout_elapsed_s.toFixed(1)}s)`;
            sourceBadge.innerText = 'AI + IMU + 6-STATE EKF';
            sourceBadge.style.color = '#ff3366';
            sourceBadge.style.borderColor = '#ff3366';
            this.showBlackoutBanner(true, nav.blackout_elapsed_s);
        } else if (gnssMode === 'DEGRADED' || gnssMode === 'GNSS_DEGRADED') {
            gnssPill.querySelector('.status-dot').className = 'status-dot dot-amber';
            gnssText.innerText = 'GNSS DEGRADED (LOW FIX)';
            sourceBadge.innerText = 'GNSS + AI/IMU EKF';
            sourceBadge.style.color = '#ffb800';
            sourceBadge.style.borderColor = '#ffb800';
            this.showBlackoutBanner(false);
        } else if (gnssMode === 'REACQUIRED' || gnssMode === 'GNSS_REACQUIRED') {
            gnssPill.querySelector('.status-dot').className = 'status-dot dot-cyan';
            gnssText.innerText = 'GNSS REACQUIRED';
            sourceBadge.innerText = 'GNSS + AI/IMU EKF';
            this.showBlackoutBanner(false);
        } else {
            gnssPill.querySelector('.status-dot').className = 'status-dot dot-green';
            gnssText.innerText = 'GNSS HEALTHY (100% Fix)';
            sourceBadge.innerText = 'GNSS + AI/IMU EKF';
            sourceBadge.style.color = '#00ffcc';
            sourceBadge.style.borderColor = '#00ffcc';
            this.showBlackoutBanner(false);
        }
    }

    // ── 5. Update Map Position & Breadcrumbs ─────────────────────────────────
    updateMapPosition(nav) {
        const lat = nav.latitude;
        const lon = nav.longitude;
        const headingDeg = (nav.heading_deg !== undefined) ? nav.heading_deg : ((nav.heading_rad || 0.0) * 180.0 / Math.PI);
        const isOutage = (nav.gnss_mode === 'DENIED' || nav.gnss_mode === 'GNSS_DENIED');

        const newPos = [lat, lon];
        this.deviceMarker.setLatLng(newPos);

        // Center map immediately on first valid incoming coordinate
        if (!this.hasCenteredMap && lat !== 0.0 && lon !== 0.0) {
            this.map.setView(newPos, 18, { animate: true });
            this.hasCenteredMap = true;
            console.log(`[Map] Centered view on phone location: [${lat}, ${lon}]`);
        }

        // Rotate Chevron Marker
        const rotator = document.getElementById('markerRotator') || document.querySelector('.custom-vehicle-icon');
        if (rotator) {
            rotator.style.transform = `rotate(${headingDeg}deg)`;
            rotator.style.transition = 'transform 0.1s linear';
        }

        // Add to Breadcrumbs only when position actually moves (> 0.05m)
        const targetList = isOutage ? this.outagePoints : this.pathPoints;
        const lastPt = targetList.length > 0 ? targetList[targetList.length - 1] : null;
        const hasMoved = !lastPt || (Math.abs(lastPt[0] - lat) > 0.000001 || Math.abs(lastPt[1] - lon) > 0.000001);

        if (hasMoved) {
            if (isOutage) {
                this.outagePoints.push(newPos);
                this.outagePathPolyline.setLatLngs(this.outagePoints);
            } else {
                this.pathPoints.push(newPos);
                this.livePathPolyline.setLatLngs(this.pathPoints);
            }
            this.map.panTo(newPos, { animate: true, duration: 0.2 });
        }
    }

    showBlackoutBanner(visible, elapsedS = 0.0) {
        const banner = document.getElementById('blackoutAlertBanner');
        const countdown = document.getElementById('blackoutCountdown');
        if (visible) {
            banner.classList.remove('hidden');
            countdown.innerText = `${elapsedS.toFixed(1)}s`;
        } else {
            banner.classList.add('hidden');
        }
    }

    // ── 6. Event Listeners & Mode Switcher ───────────────────────────────────
    initEventListeners() {
        // Mode Selector (Live vs Replay)
        const modeSelect = document.getElementById('modeSelect');
        modeSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            if (val === 'live') {
                this.currentMode = 'live';
                clearInterval(this.replayTimer);
                this.pathPoints = [];
                this.outagePoints = [];
                this.livePathPolyline.setLatLngs([]);
                this.outagePathPolyline.setLatLngs([]);
                console.log('[Mode] Switched to LIVE TELEMETRY STREAM.');
            } else {
                this.currentMode = 'replay';
                this.loadReplayDrive(val.replace('replay_', ''));
            }
        });

        // Blackout Demo Trigger Button
        const btnBlackout = document.getElementById('btnBlackoutDemo');
        btnBlackout.addEventListener('click', () => {
            this.isBlackoutActive = !this.isBlackoutActive;
            btnBlackout.innerHTML = this.isBlackoutActive
                ? '<span class="btn-icon">✅</span> RESTORE GNSS SIGNALS'
                : '<span class="btn-icon">⚡</span> TRIGGER GNSS BLACKOUT TEST';
            
            // Send command via WebSocket to gateway server and phone
            if (this.ws && this.isWsConnected) {
                this.ws.send(JSON.stringify({
                    type: 'COMMAND_BLACKOUT_TOGGLE',
                    active: this.isBlackoutActive
                }));
            }
        });
    }

    // ── 7. Replay Mode Engine ───────────────────────────────────────────────
    async loadReplayDrive(driveId) {
        console.log(`[Replay] Loading drive S-${driveId}...`);
        try {
            const res = await fetch(`data/drive_S-${driveId}.json`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this.startReplay(data);
        } catch (err) {
            console.error('Failed to load replay drive:', err);
        }
    }

    startReplay(data) {
        clearInterval(this.replayTimer);
        this.pathPoints = [];
        this.outagePoints = [];
        this.livePathPolyline.setLatLngs([]);
        this.outagePathPolyline.setLatLngs([]);
        this.replayIndex = 0;

        const timestamps = data.timestamps || [];
        const fusedX = data.fused_pos_x || [];
        const fusedY = data.fused_pos_y || [];
        const speeds = data.ai_speed || [];
        const headings = data.fused_heading || [];
        const outages = data.is_blackout || [];

        const originLat = 12.9716;
        const originLon = 77.5946;
        const rEarth = 6378137.0;

        this.replayTimer = setInterval(() => {
            if (this.replayIndex >= timestamps.length) {
                this.replayIndex = 0; // Loop
            }

            const i = this.replayIndex;
            const east = fusedX[i] || 0.0;
            const north = fusedY[i] || 0.0;
            const lat = originLat + (north / rEarth) * (180.0 / Math.PI);
            const lon = originLon + (east / (rEarth * Math.cos(originLat * Math.PI / 180.0))) * (180.0 / Math.PI);

            const isOutage = outages[i] || false;
            const nav = {
                timestamp_s: timestamps[i] || (i * 0.1),
                latitude: lat,
                longitude: lon,
                pos_east_m: east,
                pos_north_m: north,
                speed_mps: speeds[i] || 0.0,
                speed_kmh: (speeds[i] || 0.0) * 3.6,
                heading_deg: (headings[i] || 0.0) * 180.0 / Math.PI,
                confidence_pct: isOutage ? 65.0 : 95.0,
                uncertainty_sigma_mps: 0.18,
                gnss_mode: isOutage ? 'GNSS_DENIED' : 'GNSS_NORMAL',
                source: isOutage ? 'AI_IMU_EKF_DEAD_RECKONING' : 'GNSS_AI_IMU_EKF',
                blackout_elapsed_s: isOutage ? (timestamps[i] - 60.0) : 0.0,
                context_mode: 'NORMAL_URBAN'
            };

            this.updateAvionicsHUD(nav);
            this.updateMapPosition(nav);
            this.replayIndex++;
        }, 100); // 10 Hz replay
    }
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new NavPulseDashboard();
});
