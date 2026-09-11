import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../state/navigation_state_provider.dart';
import 'theme.dart';

/// The map surface.
///
/// Deliberately drawn rather than tiled: this system's entire claim is that it works with
/// no infrastructure, so shipping a demo that needs a tile server and an API key would
/// undercut the point. It renders a grid graticule for scale reference plus the vehicle's
/// own track, which is the only geometry that actually matters here.
///
/// The one thing the map must communicate is WHERE GNSS stopped. That is drawn as a colour
/// change in the route itself - blue while satellites were correcting the filter, amber
/// while the position came from inertia alone - so divergence is visible without a legend.
class MapCanvas extends StatelessWidget {
  final List<TrackPoint> track;
  final double headingRad;
  final bool followHeading;
  final bool indoor;

  const MapCanvas({
    super.key,
    required this.track,
    required this.headingRad,
    this.followHeading = true,
    this.indoor = false,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _MapPainter(
        track: track,
        headingRad: headingRad,
        followHeading: followHeading,
        indoor: indoor,
      ),
      size: Size.infinite,
    );
  }
}

class _MapPainter extends CustomPainter {
  final List<TrackPoint> track;
  final double headingRad;
  final bool followHeading;
  final bool indoor;

  _MapPainter({
    required this.track,
    required this.headingRad,
    required this.followHeading,
    required this.indoor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = NavTheme.mapLand);

    // Puck sits low on the screen like a navigation app, so the road ahead gets the space.
    final focus = Offset(size.width / 2, size.height * 0.36);

    // Metres-per-pixel: zoom in when slow/short, out when the track is long.
    double span = indoor ? 12.0 : 60.0;
    double centerE = 0, centerN = 0;
    if (track.length > 1) {
      var minE = double.infinity, maxE = -double.infinity;
      var minN = double.infinity, maxN = -double.infinity;
      for (final p in track) {
        minE = math.min(minE, p.east);
        maxE = math.max(maxE, p.east);
        minN = math.min(minN, p.north);
        maxN = math.max(maxN, p.north);
      }
      centerE = (minE + maxE) / 2;
      centerN = (minN + maxN) / 2;
      span = math.max(
          math.max(maxE - minE, maxN - minN) * 1.8, indoor ? 12.0 : 60.0);
    }
    final scale = math.min(size.width, size.height * 0.40) / span;

    final last = track.isNotEmpty ? track.last : const TrackPoint(0, 0, false);
    final rot = followHeading ? -headingRad : 0.0;
    final cosR = math.cos(rot), sinR = math.sin(rot);

    Offset project(double e, double n) {
      final dx = (e - (indoor ? centerE : last.east)) * scale;
      final dy = (n - (indoor ? centerN : last.north)) * scale;
      // Rotate so the direction of travel points up the screen, then flip y for canvas.
      final rx = dx * cosR - dy * sinR;
      final ry = dx * sinR + dy * cosR;
      return Offset(focus.dx + rx, focus.dy - ry);
    }

    _drawGraticule(canvas, size, scale, span, project, last);
    _drawTrack(canvas, project);
    final origin = project(0, 0);
    canvas.drawCircle(origin, 5, Paint()..color = NavTheme.good);
    final startLabel = TextPainter(
        text: const TextSpan(
            text: 'START',
            style: TextStyle(color: NavTheme.good, fontSize: 11)),
        textDirection: TextDirection.ltr)
      ..layout();
    startLabel.paint(canvas, origin + const Offset(8, -16));
    canvas.save();
    final puck = project(last.east, last.north);
    canvas.translate(puck.dx, puck.dy);
    canvas.rotate(followHeading ? 0 : headingRad);
    _drawPuck(canvas, Offset.zero);
    canvas.restore();
    _drawScaleBar(canvas, size, scale);
  }

  /// A 50 m grid, aligned to the world rather than the screen, so rotation is legible and
  /// the viewer can judge distance without a tile basemap.
  void _drawGraticule(Canvas canvas, Size size, double scale, double span,
      Offset Function(double, double) project, TrackPoint last) {
    double step = 50.0;
    while (span / step > 12) {
      step *= 2;
    }
    while (span / step < 4) {
      step /= 2;
    }

    final paint = Paint()
      ..color = NavTheme.mapRoadCasing.withOpacity(0.35)
      ..strokeWidth = 1.0;

    final e0 = (last.east / step).floor() * step - step * 8;
    final n0 = (last.north / step).floor() * step - step * 8;
    for (var i = 0; i <= 16; i++) {
      final e = e0 + i * step;
      canvas.drawLine(project(e, n0), project(e, n0 + step * 16), paint);
      final n = n0 + i * step;
      canvas.drawLine(project(e0, n), project(e0 + step * 16, n), paint);
    }
  }

  void _drawTrack(Canvas canvas, Offset Function(double, double) project) {
    if (track.length < 2) return;

    final gnssPaint = Paint()
      ..color = NavTheme.accent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 9.0
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    final deniedPaint = Paint()
      ..color = NavTheme.denied
      ..style = PaintingStyle.stroke
      ..strokeWidth = 9.0
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    final casing = Paint()
      ..color = Colors.black.withOpacity(0.45)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 13.0
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    // Draw in runs so each GNSS state keeps its own colour, with a casing underneath for
    // contrast against the map ground.
    var i = 1;
    while (i < track.length) {
      final denied = track[i].denied;
      final path = Path()
        ..moveTo(project(track[i - 1].east, track[i - 1].north).dx,
            project(track[i - 1].east, track[i - 1].north).dy);
      while (i < track.length && track[i].denied == denied) {
        final p = project(track[i].east, track[i].north);
        path.lineTo(p.dx, p.dy);
        i++;
      }
      canvas.drawPath(path, casing);
      canvas.drawPath(path, denied ? deniedPaint : gnssPaint);
    }
  }

  /// The location puck: a directional cone plus a white-ringed dot, the way a navigation
  /// app shows heading confidence.
  void _drawPuck(Canvas canvas, Offset c) {
    final cone = Path()
      ..moveTo(c.dx, c.dy - 34)
      ..lineTo(c.dx - 15, c.dy + 6)
      ..lineTo(c.dx + 15, c.dy + 6)
      ..close();
    canvas.drawPath(
      cone,
      Paint()
        ..shader = LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            NavTheme.accent.withOpacity(0.55),
            NavTheme.accent.withOpacity(0.0),
          ],
        ).createShader(Rect.fromCircle(center: c, radius: 34)),
    );

    canvas.drawCircle(c, 13, Paint()..color = Colors.black.withOpacity(0.35));
    canvas.drawCircle(c, 11, Paint()..color = Colors.white);
    canvas.drawCircle(c, 8, Paint()..color = NavTheme.accent);
  }

  void _drawScaleBar(Canvas canvas, Size size, double scale) {
    // Pick a round distance that lands near 90 px.
    const candidates = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0];
    var metres = candidates.first;
    for (final c in candidates) {
      if (c * scale <= 110) metres = c;
    }
    final px = metres * scale;
    final y = size.height - 22;
    final x = size.width - px - 18;

    final p = Paint()
      ..color = Colors.white.withOpacity(0.75)
      ..strokeWidth = 2;
    canvas.drawLine(Offset(x, y), Offset(x + px, y), p);
    canvas.drawLine(Offset(x, y - 4), Offset(x, y + 4), p);
    canvas.drawLine(Offset(x + px, y - 4), Offset(x + px, y + 4), p);

    final tp = TextPainter(
      text: TextSpan(
        text: metres >= 1000
            ? '${(metres / 1000).toStringAsFixed(0)} km'
            : '${metres.toStringAsFixed(0)} m',
        style: const TextStyle(color: Colors.white70, fontSize: 11),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, Offset(x + px / 2 - tp.width / 2, y - 18));
  }

  @override
  bool shouldRepaint(covariant _MapPainter old) =>
      true; // Track data may be appended in place by the live sensor provider.
}
