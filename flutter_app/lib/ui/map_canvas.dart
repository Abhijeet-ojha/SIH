import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../safety/position_report.dart';
import '../state/navigation_state_provider.dart';
import 'theme.dart';
import '../models/tile_basemap.dart';

/// Where a real GNSS fix pinned the local frame to the Earth. Without one, the track is
/// only relative and no basemap can be placed under it.
class GeoAnchor {
  final double lat, lon, east, north;
  const GeoAnchor(this.lat, this.lon, this.east, this.north);
}

/// The camera: local east/north metres to screen pixels, and back.
///
/// Hoisted out of the painter because a tap has to be turned back into a position, and a
/// second copy of this arithmetic living in the gesture handler is how a map ends up
/// dropping pins where the user did not press.
class MapCamera {
  final Offset focus;
  final double scale; // pixels per metre
  final double centreE, centreN;
  final double rot;

  const MapCamera({
    required this.focus,
    required this.scale,
    required this.centreE,
    required this.centreN,
    required this.rot,
  });

  factory MapCamera.fit({
    required Size size,
    required List<TrackPoint> track,
    required bool indoor,
    required bool followHeading,
    required double headingRad,
  }) {
    final focus = Offset(size.width / 2, size.height * 0.36);
    final last = track.isNotEmpty ? track.last : const TrackPoint(0, 0, false);

    double span = indoor ? 12.0 : 60.0;
    var centreE = last.east, centreN = last.north;
    if (track.length > 1) {
      var minE = double.infinity, maxE = -double.infinity;
      var minN = double.infinity, maxN = -double.infinity;
      for (final p in track) {
        minE = math.min(minE, p.east);
        maxE = math.max(maxE, p.east);
        minN = math.min(minN, p.north);
        maxN = math.max(maxN, p.north);
      }
      span = math.max(
          math.max(maxE - minE, maxN - minN) * 1.8, indoor ? 12.0 : 60.0);
      if (indoor) {
        centreE = (minE + maxE) / 2;
        centreN = (minN + maxN) / 2;
      }
    }

    return MapCamera(
      focus: focus,
      scale: math.min(size.width, size.height * 0.40) / span,
      centreE: centreE,
      centreN: centreN,
      rot: followHeading ? -headingRad : 0.0,
    );
  }

  Offset project(double e, double n) {
    final dx = (e - centreE) * scale;
    final dy = (n - centreN) * scale;
    final c = math.cos(rot), s = math.sin(rot);
    // Rotate so the direction of travel points up the screen, then flip y for canvas.
    return Offset(focus.dx + dx * c - dy * s, focus.dy - (dx * s + dy * c));
  }

  /// Screen pixels back to local metres. Exact inverse of [project].
  (double, double) unproject(Offset p) {
    final rx = p.dx - focus.dx;
    final ry = focus.dy - p.dy;
    final c = math.cos(rot), s = math.sin(rot);
    final dx = rx * c + ry * s;
    final dy = -rx * s + ry * c;
    return (centreE + dx / scale, centreN + dy / scale);
  }
}

/// The map surface.
///
/// A bundled raster basemap is drawn underneath when one was fetched for the demo area
/// (see tool/fetch_tiles.py); where there is no coverage it falls back to a metre grid, so
/// the app never depends on tiles having been fetched. Either way nothing is requested at
/// run time - the offline claim is about running, not about building.
///
/// The one thing the map must communicate is WHERE GNSS stopped. That is drawn as a colour
/// change in the route itself - blue while satellites were correcting the filter, amber
/// while the position came from inertia alone - so divergence is visible without a legend.
class MapCanvas extends StatelessWidget {
  final List<TrackPoint> track;
  final double headingRad;
  final bool followHeading;
  final bool indoor;
  final TileBasemap? basemap;
  final GeoAnchor? anchor;

  /// Destination in local metres, if one is set.
  final (double, double)? destination;

  /// Long-press hands back local metres, ready for [NavigationStateProvider.setDestinationLocal].
  final void Function(double east, double north)? onPickDestination;

  const MapCanvas({
    super.key,
    required this.track,
    required this.headingRad,
    this.followHeading = true,
    this.indoor = false,
    this.basemap,
    this.anchor,
    this.destination,
    this.onPickDestination,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = Size(constraints.maxWidth, constraints.maxHeight);
        final camera = MapCamera.fit(
          size: size,
          track: track,
          indoor: indoor,
          followHeading: followHeading,
          headingRad: headingRad,
        );
        return GestureDetector(
          behavior: HitTestBehavior.opaque,
          onLongPressStart: onPickDestination == null
              ? null
              : (d) {
                  final (e, n) = camera.unproject(d.localPosition);
                  onPickDestination!(e, n);
                },
          child: CustomPaint(
            painter: _MapPainter(
              track: track,
              headingRad: headingRad,
              camera: camera,
              indoor: indoor,
              basemap: basemap,
              anchor: anchor,
              destination: destination,
            ),
            size: size,
          ),
        );
      },
    );
  }
}

class _MapPainter extends CustomPainter {
  final List<TrackPoint> track;
  final double headingRad;
  final MapCamera camera;
  final bool indoor;
  final TileBasemap? basemap;
  final GeoAnchor? anchor;
  final (double, double)? destination;

  _MapPainter({
    required this.track,
    required this.headingRad,
    required this.camera,
    required this.indoor,
    required this.basemap,
    required this.anchor,
    required this.destination,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = NavTheme.mapLand);

    final drewTiles = _drawTiles(canvas, size);
    if (!drewTiles) _drawGraticule(canvas, size);

    _drawTrack(canvas);
    _drawDestination(canvas);

    final origin = camera.project(0, 0);
    canvas.drawCircle(origin, 5, Paint()..color = NavTheme.good);
    _label(canvas, 'START', origin + const Offset(8, -16), NavTheme.good);

    final last = track.isNotEmpty ? track.last : const TrackPoint(0, 0, false);
    canvas.save();
    final puck = camera.project(last.east, last.north);
    canvas.translate(puck.dx, puck.dy);
    canvas.rotate(camera.rot == 0 ? headingRad : 0);
    _drawPuck(canvas, Offset.zero);
    canvas.restore();

    _drawScaleBar(canvas, size);
    if (drewTiles) {
      // Attribution is a condition of using OpenStreetMap data, not a nicety.
      _label(canvas, basemap!.attribution, Offset(8, size.height - 16),
          Colors.white70);
    }
  }

  /// Bundled tiles, geo-referenced through the GNSS anchor. Returns false when there is
  /// nothing to draw, so the caller can fall back to the grid.
  ///
  /// Only drawn north-up. Rotating a raster basemap turns every street label upside down
  /// half the time, and the app runs north-up anyway.
  bool _drawTiles(Canvas canvas, Size size) {
    final map = basemap;
    final a = anchor;
    if (map == null || a == null || camera.rot != 0) return false;

    var drew = false;
    final paint = Paint()..filterQuality = FilterQuality.medium;
    for (final t in map.tiles) {
      // Tile corners in geographic coordinates, then in the local metre frame.
      final north = TileBasemap.tileYToLat(t.y.toDouble(), map.zoom);
      final south = TileBasemap.tileYToLat(t.y + 1.0, map.zoom);
      final west = TileBasemap.tileXToLon(t.x.toDouble(), map.zoom);
      final east = TileBasemap.tileXToLon(t.x + 1.0, map.zoom);

      final (wE, nN) =
          PositionReport.enuFromLatLon(a.lat, a.lon, north, west);
      final (eE, sN) =
          PositionReport.enuFromLatLon(a.lat, a.lon, south, east);

      final topLeft = camera.project(a.east + wE, a.north + nN);
      final bottomRight = camera.project(a.east + eE, a.north + sN);
      final dst = Rect.fromPoints(topLeft, bottomRight);
      if (!dst.overlaps(Offset.zero & size)) continue;

      canvas.drawImageRect(
        t.image,
        Rect.fromLTWH(
            0, 0, t.image.width.toDouble(), t.image.height.toDouble()),
        dst,
        paint,
      );
      drew = true;
    }
    return drew;
  }

  /// A metre grid, aligned to the world rather than the screen, so rotation is legible and
  /// distance can be judged with no basemap at all.
  void _drawGraticule(Canvas canvas, Size size) {
    final span = math.min(size.width, size.height * 0.40) / camera.scale;
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

    final e0 = (camera.centreE / step).floor() * step - step * 8;
    final n0 = (camera.centreN / step).floor() * step - step * 8;
    for (var i = 0; i <= 16; i++) {
      final e = e0 + i * step;
      canvas.drawLine(camera.project(e, n0),
          camera.project(e, n0 + step * 16), paint);
      final n = n0 + i * step;
      canvas.drawLine(camera.project(e0, n),
          camera.project(e0 + step * 16, n), paint);
    }
  }

  void _drawTrack(Canvas canvas) {
    if (track.length < 2) return;

    Paint stroke(Color c, double w) => Paint()
      ..color = c
      ..style = PaintingStyle.stroke
      ..strokeWidth = w
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final gnssPaint = stroke(NavTheme.accent, 9.0);
    final deniedPaint = stroke(NavTheme.denied, 9.0);
    final casing = stroke(Colors.black.withOpacity(0.45), 13.0);

    // Draw in runs so each GNSS state keeps its own colour, with a casing underneath for
    // contrast against the map ground.
    var i = 1;
    while (i < track.length) {
      final denied = track[i].denied;
      final start = camera.project(track[i - 1].east, track[i - 1].north);
      final path = Path()..moveTo(start.dx, start.dy);
      while (i < track.length && track[i].denied == denied) {
        final p = camera.project(track[i].east, track[i].north);
        path.lineTo(p.dx, p.dy);
        i++;
      }
      canvas.drawPath(path, casing);
      canvas.drawPath(path, denied ? deniedPaint : gnssPaint);
    }
  }

  /// Destination pin plus a straight line from the current position.
  ///
  /// A STRAIGHT line, deliberately: there is no road graph on the device, so drawing a
  /// route down streets would be a drawing, not a route. The line is bearing and distance,
  /// which is what the system can actually justify.
  void _drawDestination(Canvas canvas) {
    final d = destination;
    if (d == null) return;
    final last = track.isNotEmpty ? track.last : const TrackPoint(0, 0, false);
    final from = camera.project(last.east, last.north);
    final to = camera.project(d.$1, d.$2);

    final dash = Paint()
      ..color = NavTheme.good.withOpacity(0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    final total = (to - from).distance;
    if (total > 1) {
      const on = 12.0, off = 9.0;
      final dir = (to - from) / total;
      for (var t = 0.0; t < total; t += on + off) {
        canvas.drawLine(
            from + dir * t, from + dir * math.min(t + on, total), dash);
      }
    }

    canvas.drawCircle(to, 9, Paint()..color = Colors.black.withOpacity(0.4));
    canvas.drawCircle(to, 7, Paint()..color = NavTheme.good);
    canvas.drawCircle(to, 3, Paint()..color = Colors.white);
    _label(canvas, 'DESTINATION', to + const Offset(11, -8), NavTheme.good);
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

  void _drawScaleBar(Canvas canvas, Size size) {
    // Pick a round distance that lands near 90 px.
    const candidates = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0];
    var metres = candidates.first;
    for (final c in candidates) {
      if (c * camera.scale <= 110) metres = c;
    }
    final px = metres * camera.scale;
    final y = size.height - 22;
    final x = size.width - px - 18;

    final p = Paint()
      ..color = Colors.white.withOpacity(0.75)
      ..strokeWidth = 2;
    canvas.drawLine(Offset(x, y), Offset(x + px, y), p);
    canvas.drawLine(Offset(x, y - 4), Offset(x, y + 4), p);
    canvas.drawLine(Offset(x + px, y - 4), Offset(x + px, y + 4), p);

    _label(
      canvas,
      metres >= 1000
          ? '${(metres / 1000).toStringAsFixed(0)} km'
          : '${metres.toStringAsFixed(0)} m',
      Offset(x + px / 2 - 14, y - 18),
      Colors.white70,
    );
  }

  void _label(Canvas canvas, String text, Offset at, Color colour) {
    TextPainter(
      text: TextSpan(
          text: text, style: TextStyle(color: colour, fontSize: 11)),
      textDirection: TextDirection.ltr,
    )
      ..layout()
      ..paint(canvas, at);
  }

  @override
  bool shouldRepaint(covariant _MapPainter old) =>
      true; // Track data may be appended in place by the live sensor provider.
}
