package com.sih2026.navpulse

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel

class MainActivity : FlutterActivity(), SensorEventListener {
    private var manager: SensorManager? = null
    private var sink: EventChannel.EventSink? = null
    private val matrix = FloatArray(9)
    private val orientation = FloatArray(3)

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        manager = getSystemService(SENSOR_SERVICE) as SensorManager
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "navpulse/relative_heading")
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                    manager?.unregisterListener(this@MainActivity)
                    sink = events
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    val sensor = manager?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
                        ?: manager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
                    if (sensor == null || manager?.registerListener(this@MainActivity, sensor, 20000) != true) {
                        events.error("NO_ROTATION", "Rotation sensor unavailable; using gyroscope fallback.", null)
                    }
                }
                override fun onCancel(arguments: Any?) {
                    manager?.unregisterListener(this@MainActivity)
                    sink = null
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            })
    }
    override fun onSensorChanged(event: SensorEvent) {
        SensorManager.getRotationMatrixFromVector(matrix, event.values)
        SensorManager.getOrientation(matrix, orientation)
        sink?.success(orientation[0].toDouble())
    }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onDestroy() {
        manager?.unregisterListener(this)
        sink = null
        super.onDestroy()
    }
}
