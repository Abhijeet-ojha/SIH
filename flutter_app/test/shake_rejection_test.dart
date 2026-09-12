import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/localization/pedestrian_tracker.dart';

/// Reproduces: "the system thinks we are moving when the user is simply shaking the phone."
///
/// This is the same failure as the vehicle-mode tilt bug, one layer down. The step detector
/// keys on |a|, which is invariant to how the phone is held - good - but also completely
/// blind to WHICH WAY the acceleration points. A hand shaking at 2.5 Hz writes the same
/// magnitude signature as a 2 Hz gait, so every shake became distance on the map.
///
/// PRESENTATION.md previously handled this by asking the presenter not to shake the phone.
/// A judge handed a phone will shake it.
///
/// Convention: phone held screen-up, so gravity reads +g on body Z.

const double g = 9.80665;

void warmup(PedestrianTracker p, {int untilMs = 2000}) {
  for (var t = 0; t < untilMs; t += 20) {
    p.addAcceleration(0, 0, g, t);
  }
}

/// Real walking, as the existing suite models it: the body bobs vertically.
void walk(PedestrianTracker p, int startMs, int cycles, {double amp = 1.8}) {
  for (var t = startMs; t < startMs + cycles * 500; t += 20) {
    p.addAcceleration(
        0, 0, g + amp * math.sin(2 * math.pi * (t - startMs) / 500), t);
  }
}

void main() {
  group('Shaking the phone must not create distance', () {
    test('Side-to-side shake at walking cadence', () {
      // The commonest way a person shakes a phone: waving it horizontally. Gravity stays
      // put, so |a| still swings - sqrt(64 sin^2 + g^2) peaks about 2.8 m/s^2 above rest -
      // and at 2.5 Hz the intervals clear the old 300 ms refractory comfortably.
      final p = PedestrianTracker();
      warmup(p);
      for (var t = 2000; t < 12000; t += 20) {
        final s = 8.0 * math.sin(2 * math.pi * 2.5 * (t - 2000) / 1000);
        p.addAcceleration(s, 0, g, t);
      }
      expect(p.steps, 0,
          reason: 'a horizontal shake produced ${p.steps} steps / '
              '${p.distanceM.toStringAsFixed(1)} m of travel');
      expect(p.distanceM, 0);
    });

    test('Shake with wrist rotation, which is what a hand actually does', () {
      // Nobody shakes a phone without turning it. The gravity direction swings through the
      // body frame, which a walker holding a phone in front of them does not do.
      final p = PedestrianTracker();
      warmup(p);
      for (var t = 2000; t < 12000; t += 20) {
        final ph = 2 * math.pi * 2.2 * (t - 2000) / 1000;
        final tilt = 0.7 * math.sin(ph); // wrist rolling back and forth, ~40 deg
        final shake = 7.0 * math.sin(ph);
        // Gravity rotates with the phone; the shake rides on top of it.
        p.addAcceleration(
          g * math.sin(tilt) + shake * math.cos(tilt),
          0,
          g * math.cos(tilt) - shake * math.sin(tilt),
          t,
        );
      }
      expect(p.steps, 0,
          reason: 'a shake with wrist rotation produced ${p.steps} steps');
    });

    test('Fast shake, above any human cadence', () {
      final p = PedestrianTracker();
      warmup(p);
      for (var t = 2000; t < 12000; t += 10) {
        final s = 9.0 * math.sin(2 * math.pi * 5.0 * (t - 2000) / 1000);
        p.addAcceleration(0, 0, g + s, t);
      }
      expect(p.steps, 0, reason: '5 Hz shaking produced ${p.steps} steps');
    });

    test('Violent shake', () {
      final p = PedestrianTracker();
      warmup(p);
      final rng = math.Random(4);
      for (var t = 2000; t < 12000; t += 20) {
        p.addAcceleration(
          18.0 * (rng.nextDouble() - 0.5),
          18.0 * (rng.nextDouble() - 0.5),
          g + 18.0 * (rng.nextDouble() - 0.5),
          t,
        );
      }
      expect(p.steps, 0, reason: 'violent shaking produced ${p.steps} steps');
    });

    test('A brief burst at gait cadence commits nothing', () {
      // Two seconds of vertical oscillation at exactly walking cadence and amplitude.
      //
      // Being honest about what this test does and does not claim: a SUSTAINED, perfectly
      // vertical, gait-cadence, gait-amplitude oscillation is not distinguishable from
      // walking by an accelerometer, because to an accelerometer it is walking. Rejecting
      // that would need an independent measurement of translation - the very thing dead
      // reckoning exists to produce. What the gait lock buys is that a person has to keep
      // it up for ~2.5 s of steady rhythm before any distance is credited, and a
      // reflexive shake does not last that long or stay that regular.
      final p = PedestrianTracker();
      warmup(p);
      for (var t = 2000; t < 4000; t += 20) {
        p.addAcceleration(
            0, 0, g + 1.8 * math.sin(2 * math.pi * (t - 2000) / 500), t);
      }
      expect(p.distanceM, 0,
          reason: 'a 2 s burst committed ${p.distanceM.toStringAsFixed(2)} m');
    });
  });

  group('Real walking must still be counted', () {
    test('Sustained gait is detected and fully credited', () {
      final p = PedestrianTracker();
      warmup(p);
      walk(p, 2000, 20);
      expect(p.steps, inInclusiveRange(16, 20),
          reason: 'only ${p.steps} of ~20 steps detected');
      // Steps held back before the gait locked must be credited retroactively, or distance
      // and step count would disagree and the map would lag the walker.
      expect(p.distanceM, closeTo(p.steps * p.stepLengthM, 1e-9));
      expect(p.north, closeTo(p.steps * p.stepLengthM, 1e-9));
    });

    test('Gentle gait still passes at the default sensitivity', () {
      final p = PedestrianTracker();
      warmup(p);
      walk(p, 2000, 20, amp: 0.95);
      expect(p.steps, inInclusiveRange(16, 20));
    });

    test('Walking resumes cleanly after a pause', () {
      final p = PedestrianTracker();
      warmup(p);
      walk(p, 2000, 10);
      final afterFirst = p.steps;
      for (var t = 7000; t < 10000; t += 20) {
        p.addAcceleration(0, 0, g, t);
      }
      walk(p, 10000, 10);
      expect(p.steps, greaterThan(afterFirst + 5),
          reason: 'after a pause only ${p.steps - afterFirst} further steps counted');
    });
  });
}
