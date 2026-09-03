package com.sih.sensorlogger

import org.json.JSONObject
import kotlin.math.*

/**
 * On-device dead-reckoning engine: frame alignment -> motion gate -> speed model -> EKF.
 *
 * What this replaces, and why:
 *
 *   val dynamicSpeed = max(0.0, (meanMagAcc - 9.80665) * 1.85 + stdAy * 4.20)
 *
 * That was the entire speed estimate - two hardcoded constants, not the trained model the
 * README described. Both of its terms rise monotonically with vibration (by Jensen's
 * inequality E[|g + noise|] > |g|, so agitating a stationary phone lifts the first term off
 * zero, and the second is a vibration statistic outright). Speed was clamped >= 0 and
 * integrated into position, so shaking the phone produced forward travel. It also assumed
 * accY was "forward" and gyroZ was "yaw", which is false the moment the phone is not
 * sitting in one exact orientation.
 *
 * The replacement:
 *   1. Gravity from the OS (Sensor.TYPE_GRAVITY) or a fallback low-pass, and every derived
 *      quantity projected onto it, so nothing depends on how the phone is held.
 *   2. A motion gate that vetoes the speed estimate when the phone is being handled.
 *   3. The exported gradient-boosted model (assets/ondevice_model.json), evaluated exactly
 *      as scripts/export_ondevice_model.py evaluates it.
 *   4. The same 6-state Joseph-form EKF that was benchmarked offline - not the previous
 *      fixed-gain 0.85/0.15 complementary blend.
 */
class OnDeviceInferenceEngine(modelJson: String? = null) {

    companion object {
        const val G0 = 9.80665
        const val WINDOW_SEC = 1.5
        const val STABILITY_WINDOW_SEC = 2.0
        const val GRAVITY_LP_HZ = 0.2

        // Motion gate thresholds. Physical, and meant to be retuned per mount - a phone in
        // a soft pocket vibrates differently from one in a rigid vent cradle.
        const val GRAV_STABILITY_MAX = 0.08
        const val TILT_RATE_MAX = 0.25
        const val STILL_ACC_RMS = 0.12
        const val STILL_YAW_RATE = 0.02
        const val DEBOUNCE_FRAMES = 5

        // Forward-axis estimation
        const val FWD_MIN_SPEED = 1.5
        const val FWD_MIN_ACCEL = 0.35
        const val FWD_MIN_EVENTS = 25
    }

    enum class MotionState { IN_VEHICLE_MOVING, STATIONARY, PHONE_HANDLED }

    // ── Streaming buffers ────────────────────────────────────────────────────
    private var capacity = 32
    private var aFwd = DoubleArray(capacity)
    private var aLat = DoubleArray(capacity)
    private var aVert = DoubleArray(capacity)
    private var aHorizMag = DoubleArray(capacity)
    private var yawRate = DoubleArray(capacity)
    private var gyroMag = DoubleArray(capacity)
    private var tiltRate = DoubleArray(capacity)
    private var gravStab = DoubleArray(capacity)
    private var count = 0
    private var head = 0

    private val gHatHist = ArrayDeque<DoubleArray>()
    private var stabilityCap = 20

    // Gravity estimate (fallback path when TYPE_GRAVITY is unavailable)
    private var gx = 0.0; private var gy = 0.0; private var gz = G0
    private var gravityInitialised = false

    // Forward-axis estimate: principal axis of horizontal acceleration during accel/brake.
    private val fwdCov = Array(3) { DoubleArray(3) }
    private var fwdEvents = 0
    private var fHat = doubleArrayOf(0.0, 1.0, 0.0)
    var forwardConfidence = 0.0
        private set
    private var lastGpsSpeed = 0.0

    // Gate state
    private var debounceRun = 0
    private var debounceVal = MotionState.STATIONARY
    var motionState = MotionState.STATIONARY
        private set
    val isStandstill: Boolean get() = motionState != MotionState.IN_VEHICLE_MOVING

    var currentUncertaintySigma = 0.2
        private set
    var isPredictiveTunnelAlert = false
        private set

    private val ekf = JosephEkf()
    val statePosX: Double get() = ekf.x[0]
    val statePosY: Double get() = ekf.x[1]
    val stateSpeed: Double get() = ekf.x[2]
    val stateHeading: Double get() = ekf.x[4]
    val stateGyroBias: Double get() = ekf.x[5]

    private val model: TreeModel? = modelJson?.let { TreeModel.fromJson(it) }

    // ── Public API ───────────────────────────────────────────────────────────

    /**
     * Push one IMU sample. Pass [gravity] from Sensor.TYPE_GRAVITY when available; the
     * sensor hub computes it for free and more accurately than a low-pass can.
     * Returns the model's speed estimate, or 0.0 while the gate is closed.
     */
    fun pushSample(
        ax: Double, ay: Double, az: Double,
        gxRaw: Double, gyRaw: Double, gzRaw: Double,
        dt: Double,
        gravity: DoubleArray? = null,
        ambientLux: Double = Double.NaN,
        stepDetected: Boolean = false
    ): Double {
        updateGravity(ax, ay, az, dt, gravity)

        val gn = sqrt(gx * gx + gy * gy + gz * gz).coerceAtLeast(1e-9)
        val ux = gx / gn; val uy = gy / gn; val uz = gz / gn

        // Linear acceleration, then split into vertical and horizontal about gravity.
        val lx = ax - gx; val ly = ay - gy; val lz = az - gz
        val vert = lx * ux + ly * uy + lz * uz
        val hx = lx - vert * ux; val hy = ly - vert * uy; val hz = lz - vert * uz
        val horizMag = sqrt(hx * hx + hy * hy + hz * hz)

        // Yaw is the gyro projected onto gravity. gyroZ never was yaw unless the phone
        // happened to be lying perfectly flat.
        val yaw = gxRaw * ux + gyRaw * uy + gzRaw * uz
        val gMag = sqrt(gxRaw * gxRaw + gyRaw * gyRaw + gzRaw * gzRaw)
        val tilt = sqrt(max(0.0, gMag * gMag - yaw * yaw))

        pushGravityHistory(doubleArrayOf(ux, uy, uz))
        val stability = gravityStability()

        // Forward axis: accumulate horizontal-acceleration covariance during genuine
        // accel/brake events while GPS is healthy, then take its principal axis.
        accumulateForwardAxis(hx, hy, hz)

        val fwd = hx * fHat[0] + hy * fHat[1] + hz * fHat[2]
        val lHat = cross(doubleArrayOf(ux, uy, uz), fHat)
        val lat = hx * lHat[0] + hy * lHat[1] + hz * lHat[2]

        store(fwd, lat, vert, horizMag, yaw, gMag, tilt, stability)

        if (count < capacity) return 0.0

        updateGate(stability, stepDetected)

        val predicted = if (motionState == MotionState.IN_VEHICLE_MOVING) {
            model?.predict(buildFeatures()) ?: 0.0
        } else {
            0.0
        }.coerceAtLeast(0.0)

        val stdFwd = std(aFwd)
        currentUncertaintySigma = when (motionState) {
            MotionState.STATIONARY -> 0.04
            MotionState.PHONE_HANDLED -> 5.0   // the estimate is meaningless; say so loudly
            else -> 0.15 + stdFwd * 0.35
        }

        isPredictiveTunnelAlert =
            !ambientLux.isNaN() && ambientLux < 100.0 && ekf.x[2] > 4.0

        ekf.predict(dt, predicted, currentUncertaintySigma, yaw, isStandstill)
        ekf.updateNhc()
        if (isStandstill) ekf.updateZupt()

        return predicted
    }

    /** GPS fix during a healthy window. Real EKF update, not a fixed-gain blend. */
    fun updateGpsFix(gpsX: Double, gpsY: Double, gpsSpeed: Double, gpsHeading: Double?) {
        lastGpsSpeed = gpsSpeed
        ekf.updateGps(gpsX, gpsY, gpsSpeed, gpsHeading)
    }

    fun reset() {
        count = 0; head = 0
        gHatHist.clear()
        gravityInitialised = false
        fwdEvents = 0; forwardConfidence = 0.0
        fHat = doubleArrayOf(0.0, 1.0, 0.0)
        for (r in fwdCov) r.fill(0.0)
        motionState = MotionState.STATIONARY
        debounceRun = 0
        ekf.reset()
    }

    // ── Internals ────────────────────────────────────────────────────────────

    /** Sizes the ring buffers from the observed sample rate on the first sample. */
    private fun configureFor(dt: Double) {
        val rate = if (dt > 0) 1.0 / dt else 10.0
        capacity = max(4, (WINDOW_SEC * rate).roundToInt())
        stabilityCap = max(2, (STABILITY_WINDOW_SEC * rate).roundToInt())
        aFwd = DoubleArray(capacity); aLat = DoubleArray(capacity)
        aVert = DoubleArray(capacity); aHorizMag = DoubleArray(capacity)
        yawRate = DoubleArray(capacity); gyroMag = DoubleArray(capacity)
        tiltRate = DoubleArray(capacity); gravStab = DoubleArray(capacity)
        count = 0; head = 0
    }

    private fun updateGravity(ax: Double, ay: Double, az: Double, dt: Double, g: DoubleArray?) {
        if (!gravityInitialised) {
            configureFor(dt)
            gx = g?.get(0) ?: ax; gy = g?.get(1) ?: ay; gz = g?.get(2) ?: az
            gravityInitialised = true
            return
        }
        if (g != null) {
            gx = g[0]; gy = g[1]; gz = g[2]
            return
        }
        val tau = 1.0 / (2.0 * PI * GRAVITY_LP_HZ)
        val alpha = if (dt > 0) dt / (dt + tau) else 0.0
        gx = (1 - alpha) * gx + alpha * ax
        gy = (1 - alpha) * gy + alpha * ay
        gz = (1 - alpha) * gz + alpha * az
    }

    private fun pushGravityHistory(u: DoubleArray) {
        gHatHist.addLast(u)
        while (gHatHist.size > stabilityCap) gHatHist.removeFirst()
    }

    /** Norm of the per-axis std of the gravity direction: ~0.01 cradled, 0.3+ shaken. */
    private fun gravityStability(): Double {
        if (gHatHist.size < 2) return 0.0
        var acc = 0.0
        for (axis in 0..2) {
            var m = 0.0
            for (v in gHatHist) m += v[axis]
            m /= gHatHist.size
            var s = 0.0
            for (v in gHatHist) { val d = v[axis] - m; s += d * d }
            acc += s / gHatHist.size
        }
        return sqrt(acc)
    }

    private fun accumulateForwardAxis(hx: Double, hy: Double, hz: Double) {
        val dv = ekf.x[2] - lastGpsSpeed
        if (abs(dv) < FWD_MIN_ACCEL || lastGpsSpeed < FWD_MIN_SPEED) return
        val v = doubleArrayOf(hx, hy, hz)
        for (i in 0..2) for (j in 0..2) fwdCov[i][j] += v[i] * v[j]
        fwdEvents++
        if (fwdEvents < FWD_MIN_EVENTS || fwdEvents % 10 != 0) return

        // Power iteration for the principal eigenvector. Cheap and converges in a handful
        // of steps for a 3x3 with a dominant axis, which is the case we care about.
        var e = doubleArrayOf(fHat[0], fHat[1], fHat[2])
        repeat(24) {
            val n = doubleArrayOf(
                fwdCov[0][0] * e[0] + fwdCov[0][1] * e[1] + fwdCov[0][2] * e[2],
                fwdCov[1][0] * e[0] + fwdCov[1][1] * e[1] + fwdCov[1][2] * e[2],
                fwdCov[2][0] * e[0] + fwdCov[2][1] * e[1] + fwdCov[2][2] * e[2]
            )
            val nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            if (nn > 1e-12) { e = doubleArrayOf(n[0] / nn, n[1] / nn, n[2] / nn) }
        }
        // Sign: forward acceleration must correlate with speeding up.
        val proj = hx * e[0] + hy * e[1] + hz * e[2]
        if (proj * dv < 0) { e[0] = -e[0]; e[1] = -e[1]; e[2] = -e[2] }
        fHat = e
        val trace = fwdCov[0][0] + fwdCov[1][1] + fwdCov[2][2]
        val along = e[0] * (fwdCov[0][0] * e[0] + fwdCov[0][1] * e[1] + fwdCov[0][2] * e[2]) +
                    e[1] * (fwdCov[1][0] * e[0] + fwdCov[1][1] * e[1] + fwdCov[1][2] * e[2]) +
                    e[2] * (fwdCov[2][0] * e[0] + fwdCov[2][1] * e[1] + fwdCov[2][2] * e[2])
        forwardConfidence = if (trace > 1e-12) along / trace else 0.0
    }

    private fun store(f: Double, l: Double, v: Double, hm: Double,
                      yaw: Double, gm: Double, tilt: Double, stab: Double) {
        aFwd[head] = f; aLat[head] = l; aVert[head] = v; aHorizMag[head] = hm
        yawRate[head] = yaw; gyroMag[head] = gm; tiltRate[head] = tilt; gravStab[head] = stab
        head = (head + 1) % capacity
        if (count < capacity) count++
    }

    private fun updateGate(stability: Double, stepDetected: Boolean) {
        val tiltRms = rms(tiltRate)
        val accRms = rms(aHorizMag)
        val yawAbs = absMean(yawRate)

        val handled = stability > GRAV_STABILITY_MAX || tiltRms > TILT_RATE_MAX || stepDetected
        val still = accRms < STILL_ACC_RMS && yawAbs < STILL_YAW_RATE

        val raw = when {
            handled -> MotionState.PHONE_HANDLED
            still -> MotionState.STATIONARY
            else -> MotionState.IN_VEHICLE_MOVING
        }
        if (raw == debounceVal) debounceRun++ else { debounceVal = raw; debounceRun = 1 }
        if (debounceRun >= DEBOUNCE_FRAMES) motionState = debounceVal
    }

    /** Must match ONDEVICE_FEATURES in scripts/export_ondevice_model.py, in order. */
    private fun buildFeatures(): DoubleArray = doubleArrayOf(
        mean(aFwd), std(aFwd), rms(aFwd),
        std(aLat), rms(aLat),
        std(aVert), rms(aVert),
        mean(aHorizMag), std(aHorizMag), rms(aHorizMag),
        absMean(yawRate), std(yawRate),
        mean(gyroMag), rms(tiltRate), mean(gravStab),
        std(aVert) * rms(aHorizMag)
    )

    private fun mean(a: DoubleArray): Double { var s = 0.0; for (v in a) s += v; return s / a.size }
    private fun absMean(a: DoubleArray): Double { var s = 0.0; for (v in a) s += abs(v); return s / a.size }
    private fun rms(a: DoubleArray): Double { var s = 0.0; for (v in a) s += v * v; return sqrt(s / a.size) }
    private fun std(a: DoubleArray): Double {
        val m = mean(a); var s = 0.0
        for (v in a) { val d = v - m; s += d * d }
        return sqrt(s / a.size)
    }
    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )
}

/**
 * Gradient-boosted tree ensemble, loaded from outputs/models/ondevice_model.json.
 * Evaluation is deliberately identical to eval_trees() in export_ondevice_model.py so the
 * parity test can be exact rather than approximate.
 */
class TreeModel(
    private val init: Double,
    private val learningRate: Double,
    private val features: List<String>,
    private val trees: List<Tree>
) {
    class Tree(
        val feature: IntArray, val threshold: DoubleArray,
        val left: IntArray, val right: IntArray, val value: DoubleArray
    )

    val numFeatures: Int get() = features.size

    fun predict(x: DoubleArray): Double {
        var out = init
        for (t in trees) {
            var node = 0
            while (t.feature[node] != -2) {
                node = if (x[t.feature[node]] <= t.threshold[node]) t.left[node] else t.right[node]
            }
            out += learningRate * t.value[node]
        }
        return out
    }

    companion object {
        fun fromJson(json: String): TreeModel {
            val o = JSONObject(json)
            val feats = o.getJSONArray("features").let { a ->
                List(a.length()) { a.getString(it) }
            }
            val treesJson = o.getJSONArray("trees")
            val trees = List(treesJson.length()) { i ->
                val t = treesJson.getJSONObject(i)
                fun ints(k: String) = t.getJSONArray(k).let { a -> IntArray(a.length()) { a.getInt(it) } }
                fun dbls(k: String) = t.getJSONArray(k).let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
                Tree(ints("feature"), dbls("threshold"), ints("left"), ints("right"), dbls("value"))
            }
            return TreeModel(o.getDouble("init"), o.getDouble("learning_rate"), feats, trees)
        }
    }
}

/**
 * 6-state Joseph-form EKF: [posX(E), posY(N), vFwd, vLat, heading, gyroBias].
 * Ported from src/fusion_ekf.py so the phone runs the filter that was benchmarked, rather
 * than the previous fixed-gain complementary blend.
 */
class JosephEkf {
    val x = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    private var P = diag(doubleArrayOf(4.0, 4.0, 1.0, 0.25, 0.05, 1e-4))

    private val qPos = 0.05; private val qVelBase = 0.12; private val qVelLat = 0.05
    private val qHeading = 0.003; private val qBias = 1e-6
    private val rGpsPos = 2.0; private val rGpsVel = 0.4; private val rGpsHeadingBase = 0.12
    private val nhcLateralVariance = 0.05 * 0.05
    private val rZupt = 0.04 * 0.04

    fun reset() {
        x.fill(0.0)
        P = diag(doubleArrayOf(4.0, 4.0, 1.0, 0.25, 0.05, 1e-4))
    }

    fun predict(dt: Double, vAi: Double, vAiStd: Double, yawRate: Double, isStationary: Boolean) {
        val px = x[0]; val py = x[1]; val vF = x[2]; val vL = x[3]; val th = x[4]; val bg = x[5]

        val alphaV: Double; val alphaLat: Double
        val vFwdEff: Double; val vLatEff: Double; val thNew: Double
        if (isStationary) {
            alphaV = 0.0; alphaLat = 0.0
            vFwdEff = 0.0; vLatEff = 0.0; thNew = th
        } else {
            alphaV = 0.25 / (1.0 + 1.5 * max(0.0, vAiStd))
            alphaLat = 0.10
            vFwdEff = (1.0 - alphaV) * vF + alphaV * vAi
            vLatEff = (1.0 - alphaLat) * vL
            thNew = wrap(th + (yawRate - bg) * dt)
        }

        x[0] = px + (vFwdEff * sin(thNew) + vLatEff * cos(thNew)) * dt
        x[1] = py + (vFwdEff * cos(thNew) - vLatEff * sin(thNew)) * dt
        x[2] = vFwdEff; x[3] = vLatEff; x[4] = thNew

        val qVelDyn = qVelBase + 1.5 * vAiStd * vAiStd
        val Q = diag(doubleArrayOf(qPos * qPos, qPos * qPos, qVelDyn,
                                   qVelLat * qVelLat, qHeading * qHeading, qBias * qBias))
        val F = eye()
        if (!isStationary) {
            val s = sin(thNew); val c = cos(thNew)
            F[0][2] = (1 - alphaV) * s * dt
            F[0][3] = (1 - alphaLat) * c * dt
            F[0][4] = (vFwdEff * c - vLatEff * s) * dt
            F[1][2] = (1 - alphaV) * c * dt
            F[1][3] = -(1 - alphaLat) * s * dt
            F[1][4] = -(vFwdEff * s + vLatEff * c) * dt
            F[2][2] = 1 - alphaV
            F[3][3] = 1 - alphaLat
            F[4][5] = -dt
        } else {
            F[2][2] = 0.0; F[3][3] = 0.0
        }
        P = add(mul(mul(F, P), transpose(F)), Q)
    }

    /** Non-holonomic constraint: a ground vehicle does not travel sideways. */
    fun updateNhc() = scalarUpdate(3, 0.0, nhcLateralVariance, intArrayOf(5))

    fun updateZupt() {
        scalarUpdate(2, 0.0, rZupt, intArrayOf(0, 1, 4, 5))
        scalarUpdate(3, 0.0, rZupt, intArrayOf(0, 1, 4, 5))
    }

    fun updateGps(gpsX: Double, gpsY: Double, gpsSpeed: Double, gpsHeading: Double?) {
        scalarUpdate(0, gpsX, rGpsPos * rGpsPos, intArrayOf(4, 5))
        scalarUpdate(1, gpsY, rGpsPos * rGpsPos, intArrayOf(4, 5))
        scalarUpdate(2, gpsSpeed, rGpsVel * rGpsVel, intArrayOf(4, 5))
        if (gpsHeading != null) {
            // Inverse-speed variance scaling: GPS course is noise at a standstill.
            val vRef = max(gpsSpeed, 0.2)
            val rH = rGpsHeadingBase * rGpsHeadingBase * (1.0 + (1.5 / vRef).pow(2))
            scalarUpdate(4, wrap(gpsHeading), rH, intArrayOf(0, 1, 2, 3), angular = true)
        }
    }

    /**
     * Single-measurement Joseph-form update on state index [idx].
     * [decoupled] lists state indices this measurement carries no observability of; their
     * gain rows are zeroed, matching the partitioned update in src/fusion_ekf.py.
     */
    private fun scalarUpdate(idx: Int, z: Double, r: Double, decoupled: IntArray,
                             angular: Boolean = false) {
        val n = 6
        val innovation = if (angular) wrap(z - x[idx]) else z - x[idx]
        val s = P[idx][idx] + r
        if (s <= 0.0) return
        val K = DoubleArray(n) { P[it][idx] / s }
        for (d in decoupled) K[d] = 0.0

        for (i in 0 until n) x[i] += K[i] * innovation
        if (angular) x[4] = wrap(x[4])

        // Joseph form: P = (I - KH) P (I - KH)^T + K R K^T
        val IKH = eye()
        for (i in 0 until n) IKH[i][idx] -= K[i]
        val KRKt = Array(n) { i -> DoubleArray(n) { j -> K[i] * r * K[j] } }
        P = add(mul(mul(IKH, P), transpose(IKH)), KRKt)
    }

    private fun wrap(a: Double): Double {
        var v = (a + PI) % (2 * PI)
        if (v < 0) v += 2 * PI
        return v - PI
    }

    private fun eye() = Array(6) { i -> DoubleArray(6) { j -> if (i == j) 1.0 else 0.0 } }
    private fun diag(d: DoubleArray) = Array(6) { i -> DoubleArray(6) { j -> if (i == j) d[i] else 0.0 } }
    private fun transpose(m: Array<DoubleArray>) = Array(6) { i -> DoubleArray(6) { j -> m[j][i] } }
    private fun add(a: Array<DoubleArray>, b: Array<DoubleArray>) =
        Array(6) { i -> DoubleArray(6) { j -> a[i][j] + b[i][j] } }
    private fun mul(a: Array<DoubleArray>, b: Array<DoubleArray>) = Array(6) { i ->
        DoubleArray(6) { j ->
            var s = 0.0
            for (k in 0 until 6) s += a[i][k] * b[k][j]
            s
        }
    }
}
