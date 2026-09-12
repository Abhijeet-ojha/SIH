package com.sih2026.navpulse

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.telephony.SmsManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

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

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "navpulse/safety")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "sendSms" -> {
                        @Suppress("UNCHECKED_CAST")
                        val recipients = (call.argument<List<String>>("recipients") ?: emptyList())
                        val body = call.argument<String>("body") ?: ""
                        result.success(sendSms(recipients, body))
                    }
                    else -> result.notImplemented()
                }
            }
    }
    /**
     * Emergency SMS.
     *
     * SMS rather than a data call because this app exists for the moments when there is no
     * data: a basement, a tunnel, a rural road. SMS rides the control channel and gets out
     * on a single bar.
     *
     * Two paths, and the fallback is the important one. Direct SmsManager send is a single
     * tap and needs no further interaction, which matters when the user may not be able to
     * look at the screen - but SEND_SMS is a restricted permission that may be refused or
     * unavailable. Rather than fail, we hand the message to the messaging app prefilled,
     * which needs no permission at all. The user then presses send. Slower, always works.
     */
    private fun sendSms(recipients: List<String>, body: String): Map<String, Any> {
        if (recipients.isEmpty()) return mapOf("status" to "failed", "reason" to "no recipients")

        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) {
            return try {
                val sms = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    getSystemService(SmsManager::class.java)
                } else {
                    @Suppress("DEPRECATION") SmsManager.getDefault()
                }
                for (number in recipients) {
                    // The body carries coordinates, an accuracy and a URL, so it will
                    // usually exceed one 160-character segment. Splitting is mandatory -
                    // an un-split long message is silently truncated by some carriers.
                    val parts = sms.divideMessage(body)
                    sms.sendMultipartTextMessage(number, null, parts, null, null)
                }
                mapOf("status" to "sent", "count" to recipients.size)
            } catch (e: Exception) {
                composeSms(recipients, body)
            }
        }

        // Ask for the permission for next time, then fall back for this time.
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.SEND_SMS), 201)
        return composeSms(recipients, body)
    }

    private fun composeSms(recipients: List<String>, body: String): Map<String, Any> {
        return try {
            val uri = Uri.parse("smsto:" + recipients.joinToString(";"))
            val intent = Intent(Intent.ACTION_SENDTO, uri).apply {
                putExtra("sms_body", body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            if (intent.resolveActivity(packageManager) != null) {
                startActivity(intent)
                mapOf("status" to "composed")
            } else {
                mapOf("status" to "failed", "reason" to "no messaging app")
            }
        } catch (e: Exception) {
            mapOf("status" to "failed", "reason" to (e.message ?: "unknown"))
        }
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
