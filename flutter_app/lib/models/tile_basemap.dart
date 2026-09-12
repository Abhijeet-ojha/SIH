import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/services.dart';

/// A raster basemap baked into the APK.
///
/// The map was previously a bare graticule, justified as avoiding a tile server. That
/// reasoning confused build time with run time: needing a tile server while the app is
/// running would undercut the offline claim, fetching tiles once during the build does
/// not. These tiles are as offline as a drawn grid and let a viewer see which street the
/// track is actually on, which is the difference between a plausible demo and a
/// convincing one.
///
/// Scope is one area, fetched by tool/fetch_tiles.py. When no tiles are bundled - or the
/// session is outside the covered area - [tileAt] returns null and the canvas falls back
/// to its grid, so the app never depends on this having been run.
class TileBasemap {
  final String name;
  final String attribution;
  final int zoom;
  final Map<int, ui.Image> _images = {};

  TileBasemap._(this.name, this.attribution, this.zoom);

  bool get isEmpty => _images.isEmpty;

  static const String _manifestPath = 'assets/tiles/manifest.json';

  /// Load the bundled basemap, or null when none was bundled.
  ///
  /// Never throws: a demo phone with no tiles must still start.
  static Future<TileBasemap?> load({double? nearLat, double? nearLon}) async {
    try {
      final raw = await rootBundle.loadString(_manifestPath);
      final j = jsonDecode(raw) as Map<String, dynamic>;
      final tiles = (j['tiles'] as List).cast<Map<String, dynamic>>();
      if (tiles.isEmpty) return null;

      // Bundles may carry several zooms. Use the most detailed one available, which is
      // the one worth the pixels on a phone screen.
      final zoom = tiles.map((t) => t['z'] as int).reduce(math.max);

      final map = TileBasemap._(
        (j['name'] ?? 'Offline area').toString(),
        (j['attribution'] ?? '(c) OpenStreetMap contributors').toString(),
        zoom,
      );

      for (final t in tiles.where((t) => t['z'] == zoom)) {
        final x = t['x'] as int, y = t['y'] as int;
        try {
          final bytes = await rootBundle.load('assets/tiles/$zoom/$x/$y.png');
          map._images[_key(x, y)] = await _decode(bytes.buffer.asUint8List());
        } catch (_) {
          // One missing tile leaves a hole, not a crash.
        }
      }
      return map.isEmpty ? null : map;
    } catch (_) {
      return null; // no manifest bundled
    }
  }

  static Future<ui.Image> _decode(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    return (await codec.getNextFrame()).image;
  }

  static int _key(int x, int y) => x * 100000 + y;

  ui.Image? tileAt(int x, int y) => _images[_key(x, y)];

  Iterable<({int x, int y, ui.Image image})> get tiles =>
      _images.entries.map((e) => (
            x: e.key ~/ 100000,
            y: e.key % 100000,
            image: e.value,
          ));

  // ── Web Mercator, the slippy-map convention ──────────────────────────────

  static double lonToTileX(double lon, int z) =>
      (lon + 180.0) / 360.0 * (1 << z);

  static double latToTileY(double lat, int z) {
    final r = lat * math.pi / 180.0;
    // asinh(tan(lat)) is the Mercator y, and is numerically better behaved than the
    // log(tan + sec) form near the equator.
    return (1.0 - _asinh(math.tan(r)) / math.pi) / 2.0 * (1 << z);
  }

  static double tileXToLon(double x, int z) => x / (1 << z) * 360.0 - 180.0;

  static double tileYToLat(double y, int z) {
    final n = math.pi * (1.0 - 2.0 * y / (1 << z));
    return math.atan(_sinh(n)) * 180.0 / math.pi;
  }

  static double _sinh(double x) => (math.exp(x) - math.exp(-x)) / 2.0;
  static double _asinh(double x) => math.log(x + math.sqrt(x * x + 1.0));
}
