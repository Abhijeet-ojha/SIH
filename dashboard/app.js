/**
 * NAVPULSE DASHBOARD APPLICATION LOGIC
 * High-precision trajectory player, Leaflet map renderer & telemetry avionics HUD
 */

(function () {
    "use strict";

    // Application State
    let allDrives = {};
    let currentDriveKey = "S-Vfa02";
    let currentDrive = null;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playSpeed = 1.0;
    let animationFrameId = null;
    let lastRenderTime = 0;

    // Leaflet Map & Layer References
    let map = null;
    let polylineGt = null;
    let polylineNaive = null;
    let polylineFused = null;
    let blackoutPolygon = null;
    let vehicleMarker = null;
    let uncertaintyCircle = null;

    // Canvas Chart Reference
    let chartCanvas = null;
    let chartCtx = null;
    const CHART_HISTORY_LENGTH = 100;
    let speedHistory = [];

    // DOM Elements
    const driveSelect = document.getElementById("driveSelect");
    const systemStatusPill = document.getElementById("systemStatusPill");
    const systemStatusText = document.getElementById("systemStatusText");
    const blackoutAlertBanner = document.getElementById("blackoutAlertBanner");
    const blackoutCountdown = document.getElementById("blackoutCountdown");

    // Telemetry HUD Elements
    const speedKmhEl = document.getElementById("speedKmh");
    const speedMpsEl = document.getElementById("speedMps");
    const gtSpeedMpsEl = document.getElementById("gtSpeedMps");
    const speedGaugeBar = document.getElementById("speedGaugeBar");
    const fusedErrValEl = document.getElementById("fusedErrVal");
    const naiveErrValEl = document.getElementById("naiveErrVal");
    const fusedImprovementValEl = document.getElementById("fusedImprovementVal");
    const contextModePill = document.getElementById("contextModePill");
    const contextModeIcon = document.getElementById("contextModeIcon");
    const contextModeLabel = document.getElementById("contextModeLabel");
    const aiSigmaValEl = document.getElementById("aiSigmaVal");
    const alphaBlendValEl = document.getElementById("alphaBlendVal");
    const gyroBiasValEl = document.getElementById("gyroBiasVal");
    const headingDegValEl = document.getElementById("headingDegVal");

    // Playback Elements
    const btnPlayPause = document.getElementById("btnPlayPause");
    const playIcon = document.getElementById("playIcon");
    const pauseIcon = document.getElementById("pauseIcon");
    const btnReset = document.getElementById("btnReset");
    const btnStepBack = document.getElementById("btnStepBack");
    const btnStepForward = document.getElementById("btnStepForward");
    const scrubberTrack = document.getElementById("scrubberTrack");
    const scrubberProgress = document.getElementById("scrubberProgress");
    const scrubberHandle = document.getElementById("scrubberHandle");
    const blackoutRegionHighlight = document.getElementById("blackoutRegionHighlight");
    const timeCurrentEl = document.getElementById("timeCurrent");
    const timeTotalEl = document.getElementById("timeTotal");
    const speedBtns = document.querySelectorAll(".btn-speed");

    // Layer Toggles
    const toggleGt = document.getElementById("toggleGt");
    const toggleFused = document.getElementById("toggleFused");
    const toggleNaive = document.getElementById("toggleNaive");

    // ── 1. Initialization ───────────────────────────────────────────────────
    async function init() {
        initMap();
        initChart();
        setupEventListeners();
        await loadDrivesData();
    }

    // ── 2. Leaflet Map Setup ─────────────────────────────────────────────────
    function initMap() {
        // Initialize Leaflet Map
        map = L.map("leafletMap", {
            zoomControl: false,
            attributionControl: false
        }).setView([52.4068, -1.5197], 15);

        // Dark Matter tiles for sleek high-tech aesthetics
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            maxZoom: 19,
            subdomains: "abcd"
        }).addTo(map);

        L.control.zoom({ position: "bottomright" }).addTo(map);

        // Custom Animated Vehicle Marker Icon
        const vehicleIcon = L.divIcon({
            className: "custom-vehicle-icon",
            html: `<div class="vehicle-marker" id="vehicleMarkerInner"><div class="vehicle-arrow"></div></div>`,
            iconSize: [28, 28],
            iconAnchor: [14, 14]
        });

        vehicleMarker = L.marker([52.4068, -1.5197], { icon: vehicleIcon, zIndexOffset: 1000 }).addTo(map);

        // Dynamic Uncertainty Bubble
        uncertaintyCircle = L.circle([52.4068, -1.5197], {
            radius: 2,
            color: "#00f0ff",
            fillColor: "#00f0ff",
            fillOpacity: 0.15,
            weight: 1
        }).addTo(map);
    }

    // ── 3. Data Loading ─────────────────────────────────────────────────────
    async function loadDrivesData() {
        try {
            const response = await fetch("data/drives.json");
            if (!response.ok) throw new Error("Failed to load drives.json");
            allDrives = await response.json();
            
            // Populate select options if needed or select default
            selectDrive(currentDriveKey);
        } catch (err) {
            console.error("Data load error:", err);
            systemStatusText.innerText = "DATA LOAD ERROR";
            systemStatusPill.classList.add("status-blackout");
        }
    }

    // ── 4. Drive Selection & Route Rendering ────────────────────────────────
    function selectDrive(driveKey) {
        currentDriveKey = driveKey;
        currentDrive = allDrives[driveKey];
        if (!currentDrive || !currentDrive.frames.length) return;

        currentFrameIdx = 0;
        speedHistory = [];
        pausePlayback();

        // Clear existing polylines
        if (polylineGt) map.removeLayer(polylineGt);
        if (polylineNaive) map.removeLayer(polylineNaive);
        if (polylineFused) map.removeLayer(polylineFused);
        if (blackoutPolygon) map.removeLayer(blackoutPolygon);

        const gtLatLngs = currentDrive.frames.map(f => [f.gt_lat, f.gt_lon]);
        const naiveLatLngs = currentDrive.frames.map(f => [f.naive_lat, f.naive_lon]);
        const fusedLatLngs = currentDrive.frames.map(f => [f.fused_lat, f.fused_lon]);

        // Draw Full Polylines
        polylineGt = L.polyline(gtLatLngs, {
            color: "#00e676",
            weight: 3.5,
            opacity: 0.85,
            dashArray: "4, 6"
        }).addTo(map);

        polylineNaive = L.polyline(naiveLatLngs, {
            color: "#ff3366",
            weight: 2.5,
            opacity: 0.75
        }).addTo(map);

        polylineFused = L.polyline(fusedLatLngs, {
            color: "#00f0ff",
            weight: 4.5,
            opacity: 0.95
        }).addTo(map);

        // Highlight Blackout Zone on Route
        const blStart = currentDrive.blackout_start_sec;
        const blEnd = currentDrive.blackout_end_sec;
        const blFrames = currentDrive.frames.filter(f => f.t >= blStart && f.t <= blEnd);
        if (blFrames.length > 1) {
            const blLatLngs = blFrames.map(f => [f.gt_lat, f.gt_lon]);
            blackoutPolygon = L.polyline(blLatLngs, {
                color: "#ffb300",
                weight: 9,
                opacity: 0.35,
                lineCap: "round"
            }).addTo(map);
        }

        // Fit map bounds smoothly
        map.fitBounds(polylineFused.getBounds(), { padding: [40, 40] });

        // Update Timeline scrubber blackout region width
        const totalDur = currentDrive.total_duration_sec;
        const blLeftPct = (blStart / totalDur) * 100.0;
        const blWidthPct = ((blEnd - blStart) / totalDur) * 100.0;
        blackoutRegionHighlight.style.left = `${blLeftPct}%`;
        blackoutRegionHighlight.style.width = `${blWidthPct}%`;

        timeTotalEl.innerText = formatTime(totalDur);

        // Render Frame 0
        renderFrame(0);
    }

    // ── 5. Playback Controller ──────────────────────────────────────────────
    function togglePlayPause() {
        if (isPlaying) {
            pausePlayback();
        } else {
            startPlayback();
        }
    }

    function startPlayback() {
        if (!currentDrive) return;
        isPlaying = true;
        playIcon.classList.add("hidden");
        pauseIcon.classList.remove("hidden");
        lastRenderTime = performance.now();
        animationFrameId = requestAnimationFrame(playbackLoop);
    }

    function pausePlayback() {
        isPlaying = false;
        playIcon.classList.remove("hidden");
        pauseIcon.classList.add("hidden");
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    }

    function resetPlayback() {
        pausePlayback();
        currentFrameIdx = 0;
        speedHistory = [];
        renderFrame(0);
    }

    function stepFrames(deltaSec) {
        if (!currentDrive) return;
        const deltaFrames = Math.round(deltaSec * currentDrive.fps);
        currentFrameIdx = Math.max(0, Math.min(currentDrive.frames.length - 1, currentFrameIdx + deltaFrames));
        renderFrame(currentFrameIdx);
    }

    function playbackLoop(currentTime) {
        if (!isPlaying || !currentDrive) return;

        const dtSec = (currentTime - lastRenderTime) / 1000.0;
        lastRenderTime = currentTime;

        // Advance frames based on play speed
        const frameIncrement = dtSec * currentDrive.fps * playSpeed;
        currentFrameIdx += frameIncrement;

        if (currentFrameIdx >= currentDrive.frames.length - 1) {
            currentFrameIdx = currentDrive.frames.length - 1;
            renderFrame(Math.floor(currentFrameIdx));
            pausePlayback();
            return;
        }

        renderFrame(Math.floor(currentFrameIdx));
        animationFrameId = requestAnimationFrame(playbackLoop);
    }

    // ── 6. Real-Time Telemetry & Frame Rendering ────────────────────────────
    function renderFrame(idx) {
        if (!currentDrive || !currentDrive.frames[idx]) return;
        const frame = currentDrive.frames[idx];

        // 1. Move Vehicle Marker on Map
        const pos = [frame.fused_lat, frame.fused_lon];
        vehicleMarker.setLatLng(pos);

        // Rotate Heading Arrow
        const headingDeg = (frame.fused_h * 180.0 / Math.PI) % 360;
        const innerMarker = document.getElementById("vehicleMarkerInner");
        if (innerMarker) {
            innerMarker.style.transform = `rotate(${headingDeg}deg)`;
        }

        // Dynamic Uncertainty Bubble
        const sigmaM = Math.max(1.5, frame.ai_sigma_v * 4.0);
        uncertaintyCircle.setLatLng(pos);
        uncertaintyCircle.setRadius(sigmaM);

        // 2. Update HUD Velocities
        const kmh = frame.fused_v * 3.6;
        speedKmhEl.innerText = kmh.toFixed(1);
        speedMpsEl.innerText = `${frame.fused_v.toFixed(1)} m/s`;
        gtSpeedMpsEl.innerText = frame.gt_v.toFixed(1);

        // Gauge Bar fill (0 - 120 km/h)
        const gaugePct = Math.min(100, (kmh / 120.0) * 100);
        speedGaugeBar.style.width = `${gaugePct}%`;

        // 3. Update Error Comparators
        fusedErrValEl.innerText = `${frame.fused_err.toFixed(2)} m`;
        naiveErrValEl.innerText = `${frame.naive_err.toFixed(1)} m`;

        if (frame.naive_err > 0.1) {
            const imp = Math.max(0, ((frame.naive_err - frame.fused_err) / frame.naive_err) * 100);
            fusedImprovementValEl.innerText = `${imp.toFixed(1)}% reduction vs naive`;
        }

        // 4. Update Context Engine & Blackout Status
        if (frame.is_blackout) {
            systemStatusPill.classList.add("status-blackout");
            systemStatusText.innerText = "GNSS BLACKOUT (DENIED)";
            blackoutAlertBanner.classList.remove("hidden");

            const remSec = Math.max(0, currentDrive.blackout_end_sec - frame.t);
            blackoutCountdown.innerText = `${remSec.toFixed(1)}s`;
        } else {
            systemStatusPill.classList.remove("status-blackout");
            systemStatusText.innerText = "GNSS HEALTHY (100% Fix)";
            blackoutAlertBanner.classList.add("hidden");
        }

        // Context Badge
        contextModeLabel.innerText = frame.context_mode;
        if (frame.context_mode === "STANDSTILL") {
            contextModeIcon.innerText = "🛑";
            contextModePill.style.color = "#ff3366";
            contextModePill.style.borderColor = "rgba(255, 51, 102, 0.4)";
        } else if (frame.context_mode === "PREDICTIVE_TUNNEL_BLACKOUT") {
            contextModeIcon.innerText = "🚇";
            contextModePill.style.color = "#ffb300";
            contextModePill.style.borderColor = "rgba(255, 179, 0, 0.4)";
        } else {
            contextModeIcon.innerText = "🟢";
            contextModePill.style.color = "#00e676";
            contextModePill.style.borderColor = "rgba(0, 230, 118, 0.3)";
        }

        // Context Sub-stats
        aiSigmaValEl.innerText = `±${frame.ai_sigma_v.toFixed(3)} m/s`;
        const alpha = 0.25 / (1.0 + 1.5 * Math.max(0, frame.ai_sigma_v));
        alphaBlendValEl.innerText = alpha.toFixed(3);
        gyroBiasValEl.innerText = `${frame.fused_bg_mrad >= 0 ? "+" : ""}${frame.fused_bg_mrad.toFixed(3)} mrad/s`;
        headingDegValEl.innerText = `${headingDeg.toFixed(1)}°`;

        // 5. Timeline Scrubber Progress
        const totalDur = currentDrive.total_duration_sec;
        const progressPct = (frame.t / totalDur) * 100.0;
        scrubberProgress.style.width = `${progressPct}%`;
        scrubberHandle.style.left = `${progressPct}%`;
        timeCurrentEl.innerText = formatTime(frame.t);

        // 6. Update Chart Buffer & Draw
        speedHistory.push({
            t: frame.t,
            gt_v: frame.gt_v,
            ai_v: frame.ai_v,
            sigma: frame.ai_sigma_v
        });
        if (speedHistory.length > CHART_HISTORY_LENGTH) {
            speedHistory.shift();
        }
        drawTelemetryChart();
    }

    // ── 7. Strip Chart Canvas Renderer ───────────────────────────────────────
    function initChart() {
        chartCanvas = document.getElementById("telemetryChart");
        chartCtx = chartCanvas.getContext("2d");
    }

    function drawTelemetryChart() {
        if (!chartCtx || speedHistory.length < 2) return;
        const w = chartCanvas.width;
        const h = chartCanvas.height;

        chartCtx.clearRect(0, 0, w, h);

        // Grid lines
        chartCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        chartCtx.lineWidth = 1;
        for (let y = 0; y <= h; y += 30) {
            chartCtx.beginPath();
            chartCtx.moveTo(0, y);
            chartCtx.lineTo(w, y);
            chartCtx.stroke();
        }

        const maxV = 25.0; // 25 m/s (~90 km/h) max scale
        const stepX = w / (CHART_HISTORY_LENGTH - 1);

        // 1. Draw ±2σ Uncertainty Band Fill
        chartCtx.fillStyle = "rgba(0, 240, 255, 0.12)";
        chartCtx.beginPath();
        for (let i = 0; i < speedHistory.length; i++) {
            const x = i * stepX;
            const upperV = Math.min(maxV, speedHistory[i].ai_v + 2 * speedHistory[i].sigma);
            const y = h - (upperV / maxV) * h;
            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);
        }
        for (let i = speedHistory.length - 1; i >= 0; i--) {
            const x = i * stepX;
            const lowerV = Math.max(0, speedHistory[i].ai_v - 2 * speedHistory[i].sigma);
            const y = h - (lowerV / maxV) * h;
            chartCtx.lineTo(x, y);
        }
        chartCtx.closePath();
        chartCtx.fill();

        // 2. Draw Ground Truth Speed (Green)
        chartCtx.strokeStyle = "#00e676";
        chartCtx.lineWidth = 2;
        chartCtx.beginPath();
        for (let i = 0; i < speedHistory.length; i++) {
            const x = i * stepX;
            const y = h - (speedHistory[i].gt_v / maxV) * h;
            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);
        }
        chartCtx.stroke();

        // 3. Draw AI Predicted Speed (Cyan)
        chartCtx.strokeStyle = "#00f0ff";
        chartCtx.lineWidth = 2.5;
        chartCtx.beginPath();
        for (let i = 0; i < speedHistory.length; i++) {
            const x = i * stepX;
            const y = h - (speedHistory[i].ai_v / maxV) * h;
            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);
        }
        chartCtx.stroke();
    }

    // ── 8. Event Listeners & UI Helpers ─────────────────────────────────────
    function setupEventListeners() {
        driveSelect.addEventListener("change", (e) => {
            selectDrive(e.target.value);
        });

        btnPlayPause.addEventListener("click", togglePlayPause);
        btnReset.addEventListener("click", resetPlayback);
        btnStepBack.addEventListener("click", () => stepFrames(-1.0));
        btnStepForward.addEventListener("click", () => stepFrames(1.0));

        // Speed Multiplier Buttons
        speedBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                speedBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                playSpeed = parseFloat(btn.dataset.speed);
            });
        });

        // Scrubber click & drag
        scrubberTrack.addEventListener("click", (e) => {
            if (!currentDrive) return;
            const rect = scrubberTrack.getBoundingClientRect();
            const clickRatio = (e.clientX - rect.left) / rect.width;
            currentFrameIdx = Math.max(0, Math.min(currentDrive.frames.length - 1, Math.floor(clickRatio * currentDrive.frames.length)));
            renderFrame(currentFrameIdx);
        });

        // Layer Toggles
        toggleGt.addEventListener("change", (e) => {
            if (!polylineGt) return;
            if (e.target.checked) map.addLayer(polylineGt);
            else map.removeLayer(polylineGt);
        });

        toggleFused.addEventListener("change", (e) => {
            if (!polylineFused) return;
            if (e.target.checked) map.addLayer(polylineFused);
            else map.removeLayer(polylineFused);
        });

        toggleNaive.addEventListener("change", (e) => {
            if (!polylineNaive) return;
            if (e.target.checked) map.addLayer(polylineNaive);
            else map.removeLayer(polylineNaive);
        });

        // Keyboard Shortcuts
        window.addEventListener("keydown", (e) => {
            if (e.code === "Space") {
                e.preventDefault();
                togglePlayPause();
            } else if (e.code === "ArrowLeft") {
                stepFrames(-1.0);
            } else if (e.code === "ArrowRight") {
                stepFrames(1.0);
            }
        });
    }

    function formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = (sec % 60).toFixed(1);
        return `${String(m).padStart(2, "0")}:${s < 10 ? "0" : ""}${s}s`;
    }

    // Launch Dashboard on DOM Ready
    document.addEventListener("DOMContentLoaded", init);
})();
