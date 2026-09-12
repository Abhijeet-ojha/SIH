import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/models/tile_basemap.dart';
import 'package:navpulse_localizer/safety/position_report.dart';
import 'package:navpulse_localizer/state/navigation_state_provider.dart';
import 'package:navpulse_localizer/ui/map_canvas.dart';

/// Map geometry has a specific failure mode: it looks plausible and is wrong. A flipped
/// sign puts the pin where the user did not press, or the basemap a street away from the
/// track, and neither is visible by eye until someone checks a coordinate. These are the
/// round trips that catch it.
void main() {
  group('Screen and metres are exact inverses', () {
    MapCamera camera({bool followHeading = false, double heading = 0}) =>
        MapCamera.fit(
          size: const Size(400, 800),
          track: const [TrackPoint(0, 0, false), TrackPoint(60, 80, false)],
          indoor: false,
          followHeading: followHeading,
          headingRad: heading,
        );

    test('unproject undoes project, north-up', () {
      final c = camera();
      for (final p in const [
        Offset(0, 0),
        Offset(400, 800),
        Offset(123, 456),
        Offset(200, 288),
      ]) {
        final (e, n) = c.unproject(p);
        final back = c.project(e, n);
        expect(back.dx, closeTo(p.dx, 1e-9));
        expect(back.dy, closeTo(p.dy, 1e-9));
      }
    });

    test('unproject undoes project with the map rotated', () {
      // A rotating camera is where a sign error hides, because north-up masks it.
      final c = camera(followHeading: true, heading: 1.1);
      expect(c.rot, isNot(0));
      const p = Offset(310, 190);
      final (e, n) = c.unproject(p);
      final back = c.project(e, n);
      expect(back.dx, closeTo(p.dx, 1e-9));
      expect(back.dy, closeTo(p.dy, 1e-9));
    });

    test('North is up and east is right on a north-up map', () {
      final c = camera();
      final origin = c.project(c.centreE, c.centreN);
      expect(c.project(c.centreE, c.centreN + 10).dy, lessThan(origin.dy));
      expect(c.project(c.centreE + 10, c.centreN).dx, greaterThan(origin.dx));
    });
  });

  group('Web Mercator round trips', () {
    test('Tile x/y and lon/lat invert each other', () {
      const z = 16;
      for (final ll in const [
        (12.9716, 77.5946),
        (28.6139, 77.2090),
        (-33.8688, 151.2093),
        (0.0, 0.0),
      ]) {
        final x = TileBasemap.lonToTileX(ll.$2, z);
        final y = TileBasemap.latToTileY(ll.$1, z);
        expect(TileBasemap.tileXToLon(x, z), closeTo(ll.$2, 1e-9));
        expect(TileBasemap.tileYToLat(y, z), closeTo(ll.$1, 1e-9));
      }
    });

    test('Tile y increases southwards, x eastwards', () {
      const z = 16;
      expect(TileBasemap.latToTileY(13.0, z),
          lessThan(TileBasemap.latToTileY(12.9, z)));
      expect(TileBasemap.lonToTileX(77.6, z),
          greaterThan(TileBasemap.lonToTileX(77.5, z)));
    });
  });

  group('Local metres and coordinates', () {
    test('enuFromLatLon inverts latLonFromEnu', () {
      const aLat = 12.9716, aLon = 77.5946;
      for (final en in const [(0.0, 0.0), (250.0, -400.0), (-1200.0, 900.0)]) {
        final (lat, lon) =
            PositionReport.latLonFromEnu(aLat, aLon, en.$1, en.$2);
        final (e, n) = PositionReport.enuFromLatLon(aLat, aLon, lat, lon);
        expect(e, closeTo(en.$1, 1e-6));
        expect(n, closeTo(en.$2, 1e-6));
      }
    });
  });

  group('Destination guidance', () {
    NavigationStateProvider walker() {
      final nav = NavigationStateProvider()..setIndoorMode(true);
      nav.pedestrian.east = 0;
      nav.pedestrian.north = 0;
      nav.pedestrian.heading = 0; // facing north
      return nav;
    }

    test('A destination due east of a north-facing walker is to the RIGHT', () {
      final nav = walker()..setDestinationLocal(100, 0);
      expect(nav.destinationDistanceM, closeTo(100, 1e-9));
      expect(nav.destinationRelativeBearingDeg, closeTo(90, 1e-9));
    });

    test('Due west is to the LEFT, and behind is a turnaround', () {
      final nav = walker()..setDestinationLocal(-100, 0);
      expect(nav.destinationRelativeBearingDeg, closeTo(-90, 1e-9));
      nav.setDestinationLocal(0, -100);
      expect(nav.destinationRelativeBearingDeg!.abs(), closeTo(180, 1e-9));
    });

    test('Turning the walker turns the instruction with them', () {
      final nav = walker()..setDestinationLocal(100, 0);
      nav.pedestrian.heading = math.pi / 2; // now facing east
      expect(nav.destinationRelativeBearingDeg, closeTo(0, 1e-9),
          reason: 'facing the destination must read as straight ahead');
    });

    test('A destination cannot be placed from coordinates with no GNSS anchor',
        () {
      // Refusing beats guessing: with no anchor there is no way to know where a
      // latitude sits in the local frame, and a silently wrong pin is worse than none.
      final nav = walker();
      expect(nav.setDestinationLatLon(12.98, 77.60), isFalse);
      expect(nav.hasDestination, isFalse);
    });

    test('Clearing removes it', () {
      final nav = walker()..setDestinationLocal(10, 10);
      expect(nav.hasDestination, isTrue);
      nav.clearDestination();
      expect(nav.hasDestination, isFalse);
      expect(nav.destinationDistanceM, isNull);
    });
  });

  test('With no tiles bundled the basemap is absent, not an error', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    // The repo ships an empty manifest so the build never depends on the fetch script
    // having been run.
    expect(await TileBasemap.load(), isNull);
  });
}
