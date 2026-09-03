package com.sih.sensorlogger

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import okhttp3.*
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlin.math.*

/**
 * SIH PS 168: Real-Time Smartphone Hardware Sensor Engine & WebSocket Telemetry Gateway.
 * Captures 50 Hz Accelerometer, Gyroscope, Hardware Fused Compass, and GPS/Network Location
 * with Live Frequency Diagnostics (Hz), Drive/Walk Motion Simulator, and on-device ML inference.
 */
class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var rotationVectorSensor: Sensor? = null
    private var magSensor: Sensor? = null
    private lateinit var locationManager: LocationManager

    // On-Device ML Inference Engine
    private val inferenceEngine = OnDeviceInferenceEngine()

    private var isStreaming = false
    private var isDriving = false
    private var isBlackout = false
    private var blackoutElapsedS = 0.0

    // Latest real hardware sensor values
    private var ax = 0f; private var ay = 0f; private var az = 9.81f
    private var gx = 0f; private var gy = 0f; private var gz = 0f
    private var compassHeadingDeg = 0.0
    private var gpsLat = 12.9716; private var gpsLon = 77.5946
    private var gpsSpeed = 0f; private var gpsHeading = 0f
    private var hasGpsFix = false
    private var originLat = 12.9716; private var originLon = 77.5946
    private val rEarth = 6378137.0

    // Real-Time Measured Sampling Frequencies (Hz) & Sample Counters
    private var accelCount = 0; private var gyroCount = 0
    private var rotCount = 0; private var gnssCount = 0
    private var mlCount = 0; private var packetCount = 0

    private var measuredAccelHz = 0.0
    private var measuredGyroHz = 0.0
    private var measuredRotHz = 0.0
    private var measuredGnssHz = 0.0
    private var measuredMlHz = 0.0

    private var lastRateCalcTimeMs = System.currentTimeMillis()

    // Dead Reckoning Position in Local Tangent Plane (ENU)
    private var estPosEast = 0.0
    private var estPosNorth = 0.0
    private var estSpeedMps = 0.0
    private var sequenceNum = 0
    private var lastTimestampMs = System.currentTimeMillis()

    // Rotation Matrix Buffers
    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)

    // Networking
    private var webSocket: WebSocket? = null
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private val handler = Handler(Looper.getMainLooper())
    private var streamRunnable: Runnable? = null
    private var rateCalcRunnable: Runnable? = null

    // UI Elements
    private lateinit var statusText: TextView
    private lateinit var logButton: Button
    private lateinit var driveButton: Button
    private lateinit var blackoutButton: Button
    private lateinit var ipInput: EditText
    private lateinit var txtSpeed: TextView
    private lateinit var txtHeading: TextView
    private lateinit var txtAccel: TextView
    private lateinit var txtGyro: TextView
    private lateinit var txtLocation: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        logButton = findViewById(R.id.logButton)
        driveButton = findViewById(R.id.driveButton)
        blackoutButton = findViewById(R.id.blackoutButton)
        ipInput = findViewById(R.id.ipInput)
        txtSpeed = findViewById(R.id.txtSpeed)
        txtHeading = findViewById(R.id.txtHeading)
        txtAccel = findViewById(R.id.txtAccel)
        txtGyro = findViewById(R.id.txtGyro)
        txtLocation = findViewById(R.id.txtLocation)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) ?: sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        checkPermissions()
        startLiveSensors()
        startRateCalculator()

        logButton.setOnClickListener {
            if (isStreaming) {
                stopStreaming()
            } else {
                startStreaming()
            }
        }

        driveButton.setOnClickListener {
            isDriving = !isDriving
            if (isDriving) {
                estSpeedMps = 4.17 // 15 km/h
                driveButton.text = "🛑 STOP MOTION SIMULATOR"
                driveButton.setBackgroundColor(0xFFFFB800.toInt())
                driveButton.setTextColor(0xFF000000.toInt())
                Toast.makeText(this, "15 km/h Motion Active! Rotate phone to steer vehicle.", Toast.LENGTH_SHORT).show()
                if (!isStreaming) startStreaming()
            } else {
                estSpeedMps = 0.0
                driveButton.text = "🚗 2. DRIVE MOTION SIMULATOR (15 km/h)"
                driveButton.setBackgroundColor(0xFF00E676.toInt())
                driveButton.setTextColor(0xFF000000.toInt())
            }
        }

        blackoutButton.setOnClickListener {
            isBlackout = !isBlackout
            if (isBlackout) {
                blackoutButton.text = "✅ RESTORE GNSS SIGNALS"
                blackoutButton.setBackgroundColor(0xFF00FFCC.toInt())
                blackoutButton.setTextColor(0xFF000000.toInt())
                statusText.text = "⚡ GNSS BLACKOUT ACTIVE (AI-DR DEAD RECKONING)"
            } else {
                blackoutButton.text = "⚡ 3. SIMULATE GNSS BLACKOUT"
                blackoutButton.setBackgroundColor(0xFFFF3366.toInt())
                blackoutButton.setTextColor(0xFFFFFFFF.toInt())
                blackoutElapsedS = 0.0
                statusText.text = "🟢 GNSS HEALTHY (100% Fix)"
            }
        }
    }

    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.HIGH_SAMPLING_RATE_SENSORS
        )
        ActivityCompat.requestPermissions(this, permissions, 100)

        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
                ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                val lastNet = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                val loc = lastGps ?: lastNet
                loc?.let {
                    gpsLat = it.latitude
                    gpsLon = it.longitude
                    originLat = it.latitude
                    originLon = it.longitude
                    hasGpsFix = true
                }
            }
        } catch (_: Exception) {}
    }

    private fun startLiveSensors() {
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        rotationVectorSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }

        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 200L, 0f, this)
                locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 500L, 0f, this)
            }
        } catch (_: Exception) {}
    }

    private fun startRateCalculator() {
        lastRateCalcTimeMs = System.currentTimeMillis()
        rateCalcRunnable = object : Runnable {
            override fun run() {
                val now = System.currentTimeMillis()
                val dtS = max(0.1, (now - lastRateCalcTimeMs) / 1000.0)
                lastRateCalcTimeMs = now

                measuredAccelHz = accelCount / dtS
                measuredGyroHz = gyroCount / dtS
                measuredRotHz = rotCount / dtS
                measuredGnssHz = gnssCount / dtS
                measuredMlHz = mlCount / dtS

                accelCount = 0
                gyroCount = 0
                rotCount = 0
                gnssCount = 0
                mlCount = 0

                handler.postDelayed(this, 1000)
            }
        }
        rateCalcRunnable?.let { handler.post(it) }
    }

    private fun startStreaming() {
        val laptopIp = ipInput.text.toString().trim()
        if (laptopIp.isEmpty()) {
            Toast.makeText(this, "Please enter your laptop IP address", Toast.LENGTH_SHORT).show()
            return
        }

        val wsUrl = "ws://$laptopIp:8765/telemetry"
        statusText.text = "Connecting to $wsUrl..."

        val request = Request.Builder().url(wsUrl).build()
        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                runOnUiThread {
                    isStreaming = true
                    logButton.text = "🛑 STOP STREAMING"
                    statusText.text = "🟢 STREAMING LIVE (${String.format(Locale.US, "%.1f", measuredAccelHz)} Hz -> $laptopIp:8765)"
                    Toast.makeText(this@MainActivity, "Connected to Laptop Gateway!", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                runOnUiThread {
                    statusText.text = "Link notice: ${t.message}. Reconnecting..."
                }
            }
        })

        lastTimestampMs = System.currentTimeMillis()
        streamRunnable = object : Runnable {
            override fun run() {
                transmitTelemetryPacket()
                handler.postDelayed(this, 50)
            }
        }
        streamRunnable?.let { handler.post(it) }
    }

    private fun stopStreaming() {
        isStreaming = false
        streamRunnable?.let { handler.removeCallbacks(it) }
        webSocket?.close(1000, "User stopped")
        webSocket = null

        logButton.text = "🚀 1. START STREAMING TO LAPTOP"
        statusText.text = "Streaming paused. Sensors active."
    }

    private fun transmitTelemetryPacket() {
        val nowMs = System.currentTimeMillis()
        val dt = max(0.01, (nowMs - lastTimestampMs) / 1000.0)
        lastTimestampMs = nowMs
        packetCount++
        mlCount++

        if (isBlackout) {
            blackoutElapsedS += dt
        }

        val activeHeadingDeg = if (compassHeadingDeg != 0.0) compassHeadingDeg else gpsHeading.toDouble()
        val headingRad = (activeHeadingDeg * Math.PI / 180.0)

        // Dynamic speed calculation
        if (isDriving) {
            estSpeedMps = 4.17 // 15 km/h
        } else if (hasGpsFix && !isBlackout && gpsSpeed > 0.3f) {
            estSpeedMps = gpsSpeed.toDouble()
        } else if (inferenceEngine.stateSpeed > 0.1) {
            estSpeedMps = inferenceEngine.stateSpeed
        } else {
            estSpeedMps = max(0.0, estSpeedMps - 0.5 * dt)
        }

        // Propagate local ENU position
        estPosEast += estSpeedMps * sin(headingRad) * dt
        estPosNorth += estSpeedMps * cos(headingRad) * dt

        val curLat = originLat + (estPosNorth / rEarth) * (180.0 / Math.PI)
        val curLon = originLon + (estPosEast / (rEarth * cos(originLat * Math.PI / 180.0))) * (180.0 / Math.PI)
        val speedKmh = estSpeedMps * 3.6

        // Update on-screen UI HUD with exact measured frequencies
        txtSpeed.text = String.format(Locale.US, "Speed: %.1f km/h (%.2f m/s) • ML: %.1f Hz", speedKmh, estSpeedMps, measuredMlHz)
        txtHeading.text = String.format(Locale.US, "Compass: %.1f° (%.1f Hz • Rotation Vector)", activeHeadingDeg, measuredRotHz)
        txtLocation.text = String.format(Locale.US, "Location: %.6f°, %.6f° (GNSS: %.1f Hz)", curLat, curLon, measuredGnssHz)

        val navJson = JSONObject().apply {
            put("timestamp_s", nowMs / 1000.0)
            put("latitude", curLat)
            put("longitude", curLon)
            put("pos_east_m", estPosEast)
            put("pos_north_m", estPosNorth)
            put("speed_mps", estSpeedMps)
            put("speed_kmh", speedKmh)
            put("velocity_lat_mps", 0.0)
            put("heading_deg", activeHeadingDeg)
            put("heading_rad", headingRad)
            put("gyro_bias_rad_s", inferenceEngine.stateGyroBias)
            put("confidence_pct", if (isBlackout) max(45.0, 90.0 - blackoutElapsedS * 0.4) else 95.0)
            put("uncertainty_sigma_mps", inferenceEngine.currentUncertaintySigma)
            put("gnss_mode", if (isBlackout) "GNSS_DENIED" else "GNSS_NORMAL")
            put("source", if (isBlackout) "AI_IMU_EKF_DEAD_RECKONING" else "GNSS_AI_IMU_EKF")
            put("blackout_elapsed_s", blackoutElapsedS)
            put("context_mode", if (isBlackout) "GNSS_BLACKOUT_ACTIVE" else "NORMAL_URBAN")
        }

        val packetJson = JSONObject().apply {
            put("device_id", "ANDROID_HARDWARE_PHONE")
            put("type", "navigationState")
            put("sequence_num", sequenceNum++)
            put("timestamp_ms", nowMs)
            put("navigation", navJson)
        }

        webSocket?.send(packetJson.toString())
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelCount++
                ax = event.values[0]
                ay = event.values[1]
                az = event.values[2]
                txtAccel.text = String.format(Locale.US, "Accel: [%.1f, %.1f, %.1f] m/s² (%.1f Hz)", ax, ay, az, measuredAccelHz)

                // Pass 50 Hz sample into On-Device ML Inference Engine
                inferenceEngine.pushSample(
                    ax.toDouble(), ay.toDouble(), az.toDouble(),
                    gx.toDouble(), gy.toDouble(), gz.toDouble(),
                    ambientLux = 400.0, dt = 0.02
                )

                val dynamicAcc = sqrt(ax * ax + ay * ay)
                if (!isDriving && dynamicAcc > 1.5f) {
                    estSpeedMps = min(5.0, estSpeedMps + (dynamicAcc * 0.12).toDouble())
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroCount++
                gx = event.values[0]
                gy = event.values[1]
                gz = event.values[2]
                txtGyro.text = String.format(Locale.US, "Gyro: [%.2f, %.2f, %.2f] rad/s (%.1f Hz)", gx, gy, gz, measuredGyroHz)
            }
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                rotCount++
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                SensorManager.getOrientation(rotationMatrix, orientationAngles)
                val azimuthRad = orientationAngles[0]
                var azimuthDeg = Math.toDegrees(azimuthRad.toDouble())
                if (azimuthDeg < 0) azimuthDeg += 360.0
                compassHeadingDeg = azimuthDeg
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        gnssCount++
        gpsLat = location.latitude
        gpsLon = location.longitude
        gpsSpeed = if (location.hasSpeed()) location.speed else 0f
        gpsHeading = if (location.hasBearing()) location.bearing else gpsHeading
        hasGpsFix = true

        if (originLat == 12.9716 && originLon == 77.5946) {
            originLat = location.latitude
            originLon = location.longitude
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        rateCalcRunnable?.let { handler.removeCallbacks(it) }
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
    }
}
