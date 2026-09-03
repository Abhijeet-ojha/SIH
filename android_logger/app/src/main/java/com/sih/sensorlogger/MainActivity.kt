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
import android.os.Environment
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * SIH PS 168: Lightweight Real-Time Smartphone Sensor Logger for OnePlus Nord CE3 / Android.
 * Captures 3-axis Accelerometer, 3-axis Gyroscope, and GPS Location directly to CSV.
 * Output format matches src/data_loader.py schema.
 */
class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    // The engine took an ambientLux argument and was fed a constant, while the barometer,
    // magnetometer and step detector - all present on any modern phone, all free from the
    // sensor hub - were never registered at all.
    private var gravitySensor: Sensor? = null
    private var linAccelSensor: Sensor? = null
    private var magSensor: Sensor? = null
    private var pressureSensor: Sensor? = null
    private var lightSensor: Sensor? = null
    private var stepSensor: Sensor? = null
    private var rotationSensor: Sensor? = null
    private lateinit var locationManager: LocationManager

    private var isLogging = false
    private var logWriter: PrintWriter? = null
    private var currentFile: File? = null

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

    private lateinit var statusText: TextView
    private lateinit var logButton: Button
    private var sampleCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        logButton = findViewById(R.id.logButton)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        linAccelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        pressureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        stepSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        checkPermissions()

        logButton.setOnClickListener {
            if (isLogging) {
                stopLogging()
            } else {
                startLogging()
            }
        }
    }

    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.HIGH_SAMPLING_RATE_SENSORS,
            Manifest.permission.ACTIVITY_RECOGNITION
        )
        ActivityCompat.requestPermissions(this, permissions, 100)
    }

    private fun startLogging() {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            currentFile = File(dir, "drive_log_${timestamp}.csv")
            logWriter = PrintWriter(FileWriter(currentFile!!, true))

            // CSV Header matching data_loader.py
            logWriter?.println(
                "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z," +
                "grav_x,grav_y,grav_z,lin_x,lin_y,lin_z," +
                "mag_x,mag_y,mag_z,pressure,light,step_detector," +
                "gps_lat,gps_lon,gps_speed,gps_heading"
            )
            logWriter?.flush()

            // Register sensors at 20-50 Hz (SENSOR_DELAY_GAME)
            accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            gravitySensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            linAccelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            magSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            rotationSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            // Barometer and light change slowly; polling them at game rate wastes battery
            // for no extra information.
            pressureSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
            lightSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
            // Step detector fires per step; there is no rate to choose.
            stepSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0f, this)
            }

            isLogging = true
            sampleCount = 0
            logButton.text = "STOP LOGGING"
            statusText.text = "Logging active to: ${currentFile?.name}"
            Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to start log: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopLogging() {
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)

        logWriter?.flush()
        logWriter?.close()
        logWriter = null

        isLogging = false
        logButton.text = "START LOGGING"
        statusText.text = "Saved ${sampleCount} samples to ${currentFile?.absolutePath}"
        Toast.makeText(this, "Log saved successfully", Toast.LENGTH_SHORT).show()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isLogging || event == null) return

        val nowSec = System.currentTimeMillis() / 1000.0

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { ax = event.values[0]; ay = event.values[1]; az = event.values[2] }
            Sensor.TYPE_GRAVITY -> { grx = event.values[0]; gry = event.values[1]; grz = event.values[2] }
            Sensor.TYPE_LINEAR_ACCELERATION -> { lax = event.values[0]; lay = event.values[1]; laz = event.values[2] }
            Sensor.TYPE_MAGNETIC_FIELD -> { mx = event.values[0]; my = event.values[1]; mz = event.values[2] }
            Sensor.TYPE_PRESSURE -> pressure = event.values[0]
            Sensor.TYPE_LIGHT -> light = event.values[0]
            Sensor.TYPE_STEP_DETECTOR -> stepEvent = 1
        }

        if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            gx = event.values[0]; gy = event.values[1]; gz = event.values[2]
            
            // Log sample on gyro arrival
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
            // One step event per row, not one sticky flag until the next step.
            stepEvent = 0
            sampleCount++
            if (sampleCount % 50 == 0) {
                logWriter?.flush()
                statusText.text = "Logged $sampleCount samples..."
            }
        }
    }

    override fun onLocationChanged(loc: Location) {
        gpsLat = loc.latitude
        gpsLon = loc.longitude
        gpsSpeed = if (loc.hasSpeed()) loc.speed else 0f
        gpsHeading = if (loc.hasBearing()) loc.bearing else 0f
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
