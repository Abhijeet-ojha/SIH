package com.sih.sensorlogger

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Live top-down track of the estimated path, auto-scaled to fit.
 *
 * The point of this view for a demo is the contrast: the track is drawn in one colour while
 * GNSS is available and another while it is not, so the dead-reckoned section is visually
 * obvious. That is the whole claim of the project, on screen, in real time.
 *
 * Deliberately a plain Canvas view rather than a map SDK - no API key, no network, works in
 * a basement, which is exactly where this system is supposed to earn its keep.
 */
class TrackView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private data class Pt(val x: Float, val y: Float, val blackout: Boolean)

    // ponytail: fixed-capacity ring, oldest points dropped. A drive is unbounded and the
    // screen is not; 4000 points at 10 Hz is ~7 minutes of visible history.
    private val capacity = 4000
    private val pts = ArrayDeque<Pt>()

    private val gnssPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#38BDF8"); style = Paint.Style.STROKE
        strokeWidth = 4f; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val blackoutPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F59E0B"); style = Paint.Style.STROKE
        strokeWidth = 5f; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F8FAFC"); style = Paint.Style.FILL
    }
    private val headRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#38BDF8"); style = Paint.Style.STROKE; strokeWidth = 2.5f
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1E293B"); style = Paint.Style.STROKE; strokeWidth = 1f
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#475569"); textSize = 26f
    }

    private var heading = 0.0

    fun addPoint(x: Double, y: Double, inBlackout: Boolean, headingRad: Double) {
        pts.addLast(Pt(x.toFloat(), y.toFloat(), inBlackout))
        while (pts.size > capacity) pts.removeFirst()
        heading = headingRad
        invalidate()
    }

    fun clear() {
        pts.clear()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        // Grid, so scale changes are visible rather than silent.
        val step = h / 4f
        var g = step
        while (g < h) { canvas.drawLine(0f, g, w, g, gridPaint); g += step }
        g = step
        while (g < w) { canvas.drawLine(g, 0f, g, h, gridPaint); g += step }

        if (pts.size < 2) {
            canvas.drawText("waiting for movement", 24f, h / 2f, labelPaint)
            return
        }

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (p in pts) {
            if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x
            if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y
        }
        val spanX = max(maxX - minX, 1f)
        val spanY = max(maxY - minY, 1f)
        // Equal scale on both axes, or the track shape lies about the turns.
        val pad = 28f
        val scale = min(( w - 2 * pad) / spanX, (h - 2 * pad) / spanY)
        val cx = (minX + maxX) / 2f
        val cy = (minY + maxY) / 2f

        fun sx(x: Float) = w / 2f + (x - cx) * scale
        // Screen y grows downward, north should grow upward.
        fun sy(y: Float) = h / 2f - (y - cy) * scale

        // Draw as runs so the blackout section keeps its own colour.
        val list = pts.toList()
        var i = 1
        while (i < list.size) {
            val seg = Path()
            val blackout = list[i].blackout
            seg.moveTo(sx(list[i - 1].x), sy(list[i - 1].y))
            while (i < list.size && list[i].blackout == blackout) {
                seg.lineTo(sx(list[i].x), sy(list[i].y))
                i++
            }
            canvas.drawPath(seg, if (blackout) blackoutPaint else gnssPaint)
        }

        // Current position, with a heading tick.
        val last = list.last()
        val hx = sx(last.x); val hy = sy(last.y)
        canvas.drawCircle(hx, hy, 7f, headPaint)
        canvas.drawCircle(hx, hy, 12f, headRingPaint)
        canvas.drawLine(hx, hy,
            hx + (sin(heading) * 22f).toFloat(),
            hy - (cos(heading) * 22f).toFloat(), headRingPaint)

        // Scale bar: a track with no scale is decoration.
        val metres = spanX.coerceAtLeast(spanY)
        canvas.drawText("${fmt(metres)} across", 16f, h - 14f, labelPaint)
    }

    private fun min(a: Float, b: Float) = if (a < b) a else b

    private fun fmt(m: Float): String =
        if (abs(m) >= 1000f) String.format("%.1f km", m / 1000f) else String.format("%.0f m", m)
}
