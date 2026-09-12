import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/safety/position_report.dart';
import 'package:navpulse_localizer/safety/safety_service.dart';

/// The safety path has one property that matters more than any other: it must never send a
/// confident wrong location. A searcher acting on a fabricated coordinate is worse off than
/// one who was told nothing.
void main() {
  final t0 = DateTime.utc(2026, 9, 12, 14, 30);

  PositionReport anchored({
    double east = 0,
    double north = 0,
    double acc = 5,
    double age = 0,
    double dist = 0,
    double drift = 0,
  }) =>
      PositionReport.fromAnchor(
        anchorLat: 12.9716,
        anchorLon: 77.5946,
        anchorAccuracyM: acc,
        eastM: east,
        northM: north,
        fixAgeSeconds: age,
        distanceSinceFixM: dist,
        drUncertaintyM: drift,
        gpsAnchored: true,
        mode: 'walking',
        at: t0,
      );

  group('Local metres become real coordinates', () {
    test('A pure north offset moves latitude only, by the right amount', () {
      // 111 m north is almost exactly 0.001 degrees of latitude anywhere on Earth.
      final r = anchored(north: 111.195);
      expect(r.lat, closeTo(12.9726, 1e-4));
      expect(r.lon, closeTo(77.5946, 1e-9));
    });

    test('A pure east offset is scaled by cos(latitude)', () {
      // At 12.97 deg, a degree of longitude is cos(12.97) = 0.9745 of a degree of latitude,
      // so the same metres east move MORE degrees than the same metres north.
      final r = anchored(east: 111.195);
      expect(r.lat, closeTo(12.9716, 1e-9));
      final dLon = r.lon - 77.5946;
      expect(dLon, closeTo(0.001 / math.cos(12.9716 * math.pi / 180), 1e-5));
      expect(dLon, greaterThan(0.001),
          reason: 'longitude degrees must be shorter than latitude degrees away from '
              'the equator; a missing cos(lat) would make these equal');
    });

    test('Uncertainty combines the fix and the drift in quadrature', () {
      final r = anchored(acc: 3, drift: 4);
      expect(r.uncertaintyM, closeTo(5.0, 1e-9)); // 3-4-5
      // Adding them linearly would give 7 and overstate the circle after a long walk.
      expect(r.uncertaintyM, lessThan(7.0));
    });

    test('Pedestrian drift grows with distance walked', () {
      expect(PositionReport.pedestrianDrift(0), closeTo(0.3, 1e-9));
      expect(PositionReport.pedestrianDrift(100), closeTo(20.3, 1e-9));
      expect(PositionReport.pedestrianDrift(100),
          greaterThan(PositionReport.pedestrianDrift(50)));
    });
  });

  group('The message a human actually receives', () {
    test('Carries position, accuracy, staleness and a tappable link', () {
      final body = anchored(acc: 8, drift: 30, age: 45, dist: 150)
          .smsBody(name: 'Asha');
      expect(body, contains('SOS'));
      expect(body, contains('Asha'));
      expect(body, contains('12.971600'));
      expect(body, contains('maps.google.com'));
      // Staleness is mandatory: a 45-second-old position during a walk is a very
      // different instruction from a live one.
      expect(body, contains('45s ago'));
      expect(body, contains('within 31m'));
    });

    test('Long staleness reads in minutes, not hundreds of seconds', () {
      expect(anchored(age: 600).smsBody(), contains('10min ago'));
    });

    test('Staleness rounds up, so it is never understated', () {
      // 89 s is nearly a minute and a half. Rounding to nearest would print "1min" and
      // tell the recipient the position is fresher than it is.
      expect(anchored(age: 89).smsBody(), contains('2min ago'));
    });

    test('Quality wording tracks the circle', () {
      expect(anchored(acc: 5).qualityWord, 'good');
      expect(anchored(acc: 5, drift: 30).qualityWord, 'approximate');
      expect(anchored(acc: 5, drift: 200).qualityWord, 'rough');
    });

    test('With no fix it says so plainly instead of inventing a coordinate', () {
      final r = PositionReport(
        lat: double.nan,
        lon: double.nan,
        uncertaintyM: double.infinity,
        fixAgeSeconds: double.infinity,
        distanceSinceFixM: 42,
        gpsAnchored: false,
        mode: 'walking',
        at: t0,
      );
      expect(r.isShareable, isFalse);
      final body = r.smsBody();
      expect(body, contains('NO GPS FIX'));
      expect(body, isNot(contains('maps.google.com')),
          reason: 'a link to a fabricated coordinate would send help to the wrong place');
      expect(body, contains('42 m'));
    });
  });

  group('Sharing refuses to lie', () {
    test('An unanchored report is queued, never sent', () async {
      final s = SafetyService()..addContact('+911234567890');
      final r = PositionReport(
        lat: double.nan,
        lon: double.nan,
        uncertaintyM: double.infinity,
        fixAgeSeconds: double.infinity,
        distanceSinceFixM: 10,
        gpsAnchored: false,
        mode: 'walking',
        at: t0,
      );
      final res = await s.share(r);
      expect(res.outcome, ShareOutcome.queued);
      expect(s.pendingCount, 1);
    });

    test('With no contacts it refuses and says why', () async {
      final res = await SafetyService().share(anchored());
      expect(res.outcome, ShareOutcome.refused);
      expect(res.detail, contains('contact'));
    });

    test('A failed send is queued, never dropped', () async {
      // No platform channel in a unit test, so the send path throws
      // MissingPluginException - the same route a phone with no messaging app takes.
      final s = SafetyService()..addContact('+911234567890');
      final res = await s.share(anchored());
      expect(res.outcome, ShareOutcome.queued);
      expect(s.pendingCount, 1,
          reason: 'a safety message that silently evaporated is worse than no feature');
    });

    test('The queue is bounded, keeping the newest', () async {
      final s = SafetyService()..addContact('+911234567890');
      for (var i = 0; i < SafetyService.maxPending + 20; i++) {
        await s.share(anchored(north: i.toDouble()));
      }
      expect(s.pendingCount, SafetyService.maxPending);
    });

    test('Flushing drops unknown-position entries and keeps real ones', () async {
      final s = SafetyService()..addContact('+911234567890');
      await s.share(anchored(north: 10));
      await s.share(PositionReport(
        lat: double.nan,
        lon: double.nan,
        uncertaintyM: double.infinity,
        fixAgeSeconds: double.infinity,
        distanceSinceFixM: 5,
        gpsAnchored: false,
        mode: 'walking',
        at: t0,
      ));
      expect(s.pendingCount, 2);
      await s.flush();
      // Both were re-attempted; the anchored one re-queues (still no channel), the
      // unanchored one is discarded rather than spamming "position unknown".
      expect(s.pending.every((r) => r.isShareable), isTrue);
    });

    test('Every attempt is logged so the feature is auditable', () async {
      final s = SafetyService()..addContact('+911234567890');
      await s.share(anchored());
      expect(s.log, isNotEmpty);
    });
  });

  test('A report survives a JSON round trip for the offline queue', () {
    final r = anchored(east: 30, north: -20, acc: 7, drift: 12, age: 45, dist: 80);
    final back = PositionReport.fromJson(r.toJson());
    expect(back.lat, closeTo(r.lat, 1e-12));
    expect(back.lon, closeTo(r.lon, 1e-12));
    expect(back.uncertaintyM, closeTo(r.uncertaintyM, 1e-12));
    expect(back.gpsAnchored, r.gpsAnchored);
    expect(back.at, r.at);
  });
}
