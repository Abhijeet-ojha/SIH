package com.sih.sensorlogger

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Environment
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Live dead-reckoning HUD and sensor logger.
 *
 * This does two jobs at once, deliberately: it records raw sensor CSVs for offline training,
 * AND runs OnDeviceInferenceEngine on the same samples so the screen shows what the
 * algorithm currently believes. Before this, the app only logged - the engine existed but
 * nothing ever called it, so there was no way to see the system work without a laptop.
 *
 * The blackout button is the demo: it stops feeding GNSS to the filter while the phone keeps
 * receiving it, so the track shows dead reckoning diverging from truth in real time, and the
 * exit error is measured against the first real fix afterwards.
 */
class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager

    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var gravitySensor: Sensor? = null
    private var linAccelSensor: Sensor? = null
    private var magSensor: Sensor? = null
    private var pressureSensor: Sensor? = null
    private var lightSensor: Sensor? = null
    private var stepSensor: Sensor? = null
    private var rotationSensor: Sensor? = null

    private var isLogging = false
    private var logWriter: PrintWriter? = null
    private var currentFile: File? = null
    private var sampleCount = 0

    // Latest sensor cache
    private var ax = 0f; private var ay = 0f; private var az = 0f
    private var gx = 0f; private var gy = 0f; private var gz = 0f
    private var grx = 0f; private var gry = 0f; private var grz = 9.80665f
    private var lax = 0f; private var lay = 0f; private var laz = 0f
    private var mx = Float.NaN; private var my = Float.NaN; private var mz = Float.NaN
    private var pressure = Float.NaN
    private var light = Float.NaN
    private var stepEvent = 0
    private var gpsLat = 0.0; private var gpsLon = 0.0
    private var gpsSpeed = 0f; private var gpsHeading = 0f
    private var hasFix = false

    private var lastSampleNs = 0L
    private var blackoutSimulated = false
    private var blackoutStartMs = 0L

    private val engine = OnDeviceInferenceEngine()

    private lateinit var statusText: TextView
    private lateinit var uncertaintyValue: TextView
    private lateinit var uncertaintyHint: TextView
    private lateinit var motionState: TextView
    private lateinit var motionReason: TextView
    private lateinit var speedValue: TextView
    private lateinit var headingValue: TextView
    private lateinit var headingCardinal: TextView
    private lateinit var gnssBadge: TextView
    private lateinit var trackView: TrackView
    private lateinit var sensorChips: ChipGroup
    private lateinit var logButton: MaterialButton
    private lateinit var blackoutButton: MaterialButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // A navigation HUD that sleeps mid-drive is useless.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        statusText = findViewById(R.id.statusText)
        uncertaintyValue = findViewById(R.id.uncertaintyValue)
        uncertaintyHint = findViewById(R.id.uncertaintyHint)
        motionState = findViewById(R.id.motionState)
        motionReason = findViewById(R.id.motionReason)
        speedValue = findViewById(R.id.speedValue)
        headingValue = findViewById(R.id.headingValue)
        headingCardinal = findViewById(R.id.headingCardinal)
        gnssBadge = findViewById(R.id.gnssBadge)
        trackView = findViewById(R.id.trackView)
        sensorChips = findViewById(R.id.sensorChips)
        logButton = findViewById(R.id.logButton)
        blackoutButton = findViewById(R.id.blackoutButton)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        linAccelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        stepSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        populateSensorChips()
        checkPermissions()

        logButton.setOnClickListener { if (isLogging) stopLogging() else startLogging() }
        blackoutButton.setOnClickListener { toggleBlackout() }
        blackoutButton.isEnabled = false
    }

    /** Which sensors this handset actually has. Absence is a real result, so show it. */
    private fun populateSensorChips() {
        sensorChips.removeAllViews()
        listOf(
            "Accel" to accelSensor, "Gyro" to gyroSensor, "Gravity" to gravitySensor,
            "LinAccel" to linAccelSensor, "Mag" to magSensor, "Baro" to pressureSensor,
            "Light" to lightSensor, "Steps" to stepSensor, "RotVec" to rotationSensor
        ).forEach { (name, sensor) ->
            val present = sensor != null
            sensorChips.addView(Chip(this).apply {
                text = if (present) name else "$name —"
                isClickable = false
                textSize = 11f
                chipBackgroundColor = android.content.res.ColorStateList.valueOf(
                    Color.parseColor(if (present) "#14304A" else "#1A1F2E"))
                setTextColor(Color.parseColor(if (present) "#7DD3FC" else "#475569"))
                chipStrokeWidth = 0f
            })
        }
    }

    private fun checkPermissions() {
        ActivityCompat.requestPermissions(this, arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.HIGH_SAMPLING_RATE_SENSORS,
            Manifest.permission.ACTIVITY_RECOGNITION
        ), 100)
    }

    private fun startLogging() {
        try {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            currentFile = File(dir, "drive_log_$stamp.csv")
            logWriter = PrintWriter(FileWriter(currentFile!!, true))
            logWriter?.println(
                "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z," +
                "grav_x,grav_y,grav_z,lin_x,lin_y,lin_z," +
                "mag_x,mag_y,mag_z,pressure,light,step_detector," +
                "gps_lat,gps_lon,gps_speed,gps_heading"
            )
            logWriter?.flush()

            accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            gravitySensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            linAccelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            rotationSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            pressureSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
            lightSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
            stepSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED) {
                locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0f, this)
            }

            engine.reset()
            trackView.clear()
            lastSampleNs = 0L
            sampleCount = 0
            isLogging = true
            blackoutButton.isEnabled = true
            logButton.text = "STOP"
            logButton.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#DC2626"))
            statusText.text = "Recording to ${currentFile?.name}"
        } catch (e: Exception) {
            Toast.makeText(this, "Could not start: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopLogging() {
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
        logWriter?.flush(); logWriter?.close(); logWriter = null
        isLogging = false
        blackoutSimulated = false
        blackoutButton.isEnabled = false
        logButton.text = "START"
        logButton.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#2563EB"))
        statusText.text = "Saved $sampleCount samples to ${currentFile?.name}"
    }

    /**
     * Withhold GNSS from the filter while still receiving it, so the divergence is visible
     * live and the exit error can be scored against the first fix after resuming.
     */
    private fun toggleBlackout() {
        blackoutSimulated = !blackoutSimulated
        if (blackoutSimulated) {
            blackoutStartMs = System.currentTimeMillis()
            blackoutButton.text = "RESTORE GNSS"
            blackoutButton.setTextColor(Color.parseColor("#4ADE80"))
        } else {
            val secs = (System.currentTimeMillis() - blackoutStartMs) / 1000.0
            blackoutButton.text = "SIMULATE GNSS BLACKOUT"
            blackoutButton.setTextColor(Color.parseColor("#F59E0B"))
            Toast.makeText(this, "Blackout lasted %.0f s".format(secs), Toast.LENGTH_SHORT).show()
        }
        updateGnssBadge()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { ax = event.values[0]; ay = event.values[1]; az = event.values[2] }
            Sensor.TYPE_GRAVITY -> { grx = event.values[0]; gry = event.values[1]; grz = event.values[2] }
            Sensor.TYPE_LINEAR_ACCELERATION -> { lax = event.values[0]; lay = event.values[1]; laz = event.values[2] }
            Sensor.TYPE_MAGNETIC_FIELD -> { mx = event.values[0]; my = event.values[1]; mz = event.values[2] }
            Sensor.TYPE_PRESSURE -> pressure = event.values[0]
            Sensor.TYPE_LIGHT -> light = event.values[0]
            Sensor.TYPE_STEP_DETECTOR -> stepEvent = 1
        }
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        gx = event.values[0]; gy = event.values[1]; gz = event.values[2]
        if (!isLogging) return

        // dt from the sensor's own clock, not wall time - wall time jitters under load and
        // this value is integrated.
        val dt = if (lastSampleNs == 0L) 0.02 else (event.timestamp - lastSampleNs) / 1e9
        lastSampleNs = event.timestamp
        if (dt <= 0.0 || dt > 1.0) return

        engine.pushSample(
            ax.toDouble(), ay.toDouble(), az.toDouble(),
            gx.toDouble(), gy.toDouble(), gz.toDouble(),
            dt,
            gravity = doubleArrayOf(grx.toDouble(), gry.toDouble(), grz.toDouble()),
            ambientLux = if (light.isNaN()) Double.NaN else light.toDouble(),
            stepDetected = stepEvent == 1
        )

        val nowSec = System.currentTimeMillis() / 1000.0
        logWriter?.printf(
            Locale.US,
            "%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f," +
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f," +
            "%.3f,%.3f,%.3f,%.3f,%.2f,%d," +
            "%.6f,%.6f,%.2f,%.2f\n",
            nowSec, ax, ay, az, gx, gy, gz,
            grx, gry, grz, lax, lay, laz,
            mx, my, mz, pressure, light, stepEvent,
            gpsLat, gpsLon, gpsSpeed, gpsHeading
        )
        stepEvent = 0
        sampleCount++

        trackView.addPoint(engine.statePosX, engine.statePosY, blackoutSimulated, engine.stateHeading)
        if (sampleCount % 5 == 0) refreshHud()
        if (sampleCount % 100 == 0) logWriter?.flush()
    }

    private fun refreshHud() {
        val kmh = (engine.stateSpeed * 3.6).roundToInt()
        speedValue.text = kmh.toString()

        var deg = Math.toDegrees(engine.stateHeading)
        if (deg < 0) deg += 360.0
        headingValue.text = deg.roundToInt().toString()
        headingCardinal.text = cardinal(deg)

        // 1-sigma position uncertainty, grown by the filter while GNSS is withheld.
        val sigma = engine.currentUncertaintySigma
        val elapsed = if (blackoutSimulated)
            (System.currentTimeMillis() - blackoutStartMs) / 1000.0 else 0.0
        val posSigma = sigma * elapsed
        uncertaintyValue.text = String.format(Locale.US, "%.1f", posSigma)
        uncertaintyValue.setTextColor(Color.parseColor(when {
            posSigma < 10 -> "#4ADE80"
            posSigma < 50 -> "#FBBF24"
            else -> "#F87171"
        }))
        uncertaintyHint.text = when {
            !blackoutSimulated && hasFix -> "GNSS correcting — uncertainty held"
            blackoutSimulated -> "dead reckoning for %.0f s".format(elapsed)
            else -> "no fix yet"
        }

        val (label, colour, reason) = when (engine.motionState) {
            OnDeviceInferenceEngine.MotionState.IN_VEHICLE_MOVING ->
                Triple("IN VEHICLE — MOVING", "#4ADE80", "Gravity steady, phone tracking the car.")
            OnDeviceInferenceEngine.MotionState.STATIONARY ->
                Triple("STATIONARY", "#94A3B8", "No horizontal force, no yaw. Position held.")
            OnDeviceInferenceEngine.MotionState.PHONE_HANDLED ->
                Triple("PHONE HANDLED", "#FBBF24",
                    "Gravity direction is moving — speed estimate vetoed.")
        }
        motionState.text = label
        motionState.setTextColor(Color.parseColor(colour))
        motionReason.text = reason

        statusText.text = "$sampleCount samples · forward-axis confidence " +
            "${(engine.forwardConfidence * 100).roundToInt()}%"
        updateGnssBadge()
    }

    private fun updateGnssBadge() {
        when {
            blackoutSimulated -> {
                gnssBadge.text = "GNSS DENIED"
                gnssBadge.setBackgroundResource(R.drawable.badge_warn)
                gnssBadge.setTextColor(Color.parseColor("#451A03"))
            }
            hasFix -> {
                gnssBadge.text = "GNSS OK"
                gnssBadge.setBackgroundResource(R.drawable.badge_good)
                gnssBadge.setTextColor(Color.parseColor("#052E16"))
            }
            else -> {
                gnssBadge.text = "NO FIX"
                gnssBadge.setBackgroundResource(R.drawable.badge_bad)
                gnssBadge.setTextColor(Color.parseColor("#450A0A"))
            }
        }
    }

    private fun cardinal(deg: Double): String {
        val dirs = arrayOf("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        return dirs[(((deg + 22.5) % 360.0) / 45.0).toInt().coerceIn(0, 7)]
    }

    override fun onLocationChanged(loc: Location) {
        gpsLat = loc.latitude
        gpsLon = loc.longitude
        gpsSpeed = if (loc.hasSpeed()) loc.speed else 0f
        gpsHeading = if (loc.hasBearing()) loc.bearing else 0f
        hasFix = true

        // The whole point of the blackout button: the fix arrives, and we refuse to use it.
        if (isLogging && !blackoutSimulated) {
            engine.updateGpsFix(0.0, 0.0, gpsSpeed.toDouble(),
                if (loc.hasBearing()) Math.toRadians(gpsHeading.toDouble()) else null)
        }
        updateGnssBadge()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}

    override fun onDestroy() {
        super.onDestroy()
        if (isLogging) stopLogging()
    }
}
