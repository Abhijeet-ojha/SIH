import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/localization/online_calibration.dart';

/// Learn while GPS is visible, spend it in the tunnel.
///
/// The tests below simulate a device with known, deliberately wrong characteristics - a
/// gyro reading 0.02 rad/s at rest with a 6% scale error, and a speed model reading 20%
/// low with a 1 m/s offset - drive it under GPS, and check that the calibration recovers
/// the truth and that using it shrinks the error a subsequent blackout would accumulate.
void main() {
  // What the simulated hardware actually does wrong.
  const trueBias = 0.02; // rad/s
  const trueScale = 1.06;
  const trueSpeedGain = 0.8;
  const trueSpeedOffset = 1.0;

  group('Cold start must behave exactly like no calibration', () {
    test('Every correction is the identity before anything is learned', () {
      final c = OnlineCalibration();
      expect(c.hasLearnedAnything, isFalse);
      expect(c.correctYawRate(0.5), closeTo(0.5, 1e-12));
      expect(c.correctSpeed(12.0), closeTo(12.0, 1e-12));
      expect(c.speedSigma, closeTo(OnlineCalibration.defaultSpeedSigma, 1e-12));
      expect(c.summary, contains('Not yet calibrated'));
    });

    test('A trickle of evidence moves the correction only a little', () {
      // Confidence-weighted blending means a handful of samples cannot swing the system.
      final c = OnlineCalibration();
      for (var i = 0; i < 10; i++) {
        c.observeStationary(trueBias, 0.02);
      }
      // 0.2 s of evidence against a 20 s horizon: barely any of the bias is applied.
      expect(c.yawBiasConfidence, lessThan(0.02));
      expect(c.correctYawRate(0.5), closeTo(0.5, 0.002));
    });
  });

  group('Learning from GPS', () {
    test('Gyro bias is recovered from stationary periods, without GPS', () {
      final c = OnlineCalibration();
      // 60 s at a red light. No fix required - this is why it still works in a canyon.
      for (var i = 0; i < 3000; i++) {
        c.observeStationary(trueBias, 0.02);
      }
      expect(c.yawBiasConfidence, greaterThan(0.9));
      expect(c.yawBias, closeTo(trueBias, 0.002));
      // And the correction removes it.
      expect(c.correctYawRate(trueBias), closeTo(0.0, 0.004));
    });

    test('Gyro scale is recovered from turns scored against GNSS course', () {
      final c = OnlineCalibration();
      // Pre-learn the bias so the scale windows are not polluted by it.
      for (var i = 0; i < 3000; i++) {
        c.observeStationary(trueBias, 0.02);
      }

      var course = 0.0;
      // Twelve 8 s windows, each containing a real turn of varying size.
      for (var w = 0; w < 12; w++) {
        final turnRate = (w.isEven ? 1 : -1) * (0.06 + 0.01 * (w % 4));
        for (var i = 0; i < 400; i++) {
          const dt = 0.02;
          // Truth advances at turnRate; the gyro reports it scaled and biased.
          course += turnRate * dt;
          final reported = turnRate / trueScale + trueBias;
          c.observeMovingWithGnss(
            compassYawRate: reported,
            dt: dt,
            gnssSpeed: 15.0,
            gnssAccuracy: 4.0,
            courseRad: course,
            mountDisturbed: false,
          );
        }
      }
      expect(c.headingWindows, greaterThan(6),
          reason: 'no usable turn windows were accepted');
      expect(c.yawScale, closeTo(trueScale, 0.03));
    });

    test('Speed gain and offset are recovered, and sigma reflects the fit', () {
      final c = OnlineCalibration();
      final rng = math.Random(7);
      for (var i = 0; i < 600; i++) {
        final truth = 5.0 + 20.0 * rng.nextDouble();
        // The model under-reads and carries an offset, plus a little noise.
        final model =
            (truth - trueSpeedOffset) / trueSpeedGain + rng.nextDouble() * 0.4 - 0.2;
        c.observeSpeed(
          modelSpeed: model,
          gnssSpeed: truth,
          gnssAccuracy: 4.0,
          mountDisturbed: false,
        );
      }
      expect(c.speedConfidence, greaterThan(0.9));
      expect(c.speedGain, closeTo(trueSpeedGain, 0.05));
      expect(c.speedOffset, closeTo(trueSpeedOffset, 0.3));
      // Having fitted the systematic part, the residual it reports to the EKF should be
      // small - far below the 5.7 m/s global default it started from.
      expect(c.speedSigma, lessThan(1.0),
          reason: 'sigma stayed at ${c.speedSigma}, so the filter would still '
              'distrust a source it can now predict well');
    });
  });

  group('The tunnel: what the learning is actually for', () {
    test('Learned heading correction shrinks blackout drift', () {
      // A 90 s blackout with the vehicle driving straight. Truth: zero turn.
      const blackoutSec = 90.0;
      const dt = 0.02;
      const reportedWhenStraight = trueBias; // gyro says this while truly going straight

      // Uncalibrated: the bias integrates directly into heading.
      final naive = reportedWhenStraight * blackoutSec;

      final c = OnlineCalibration();
      for (var i = 0; i < 3000; i++) {
        c.observeStationary(trueBias, 0.02);
      }
      var calibrated = 0.0;
      for (var i = 0; i < blackoutSec ~/ dt; i++) {
        calibrated += c.correctYawRate(reportedWhenStraight) * dt;
      }

      final naiveDeg = naive * 180 / math.pi;
      final calDeg = calibrated.abs() * 180 / math.pi;
      expect(calDeg, lessThan(naiveDeg * 0.1),
          reason: 'heading error over a 90 s tunnel: '
              '${naiveDeg.toStringAsFixed(1)}° uncalibrated vs '
              '${calDeg.toStringAsFixed(1)}° calibrated');
      // The uncalibrated case is genuinely bad - this is the thing being fixed.
      expect(naiveDeg, greaterThan(90.0));
    });
  });

  group('It must not learn from bad evidence', () {
    test('A disturbed mount contributes nothing', () {
      final c = OnlineCalibration();
      for (var i = 0; i < 2000; i++) {
        c.observeMovingWithGnss(
          compassYawRate: 0.5,
          dt: 0.02,
          gnssSpeed: 15.0,
          gnssAccuracy: 4.0,
          courseRad: 0.0, // course says straight, gyro says turning hard
          mountDisturbed: true,
        );
        c.observeSpeed(
          modelSpeed: 100.0,
          gnssSpeed: 15.0,
          gnssAccuracy: 4.0,
          mountDisturbed: true,
        );
      }
      expect(c.headingWindows, 0);
      expect(c.speedSamples, 0);
      expect(c.hasLearnedAnything, isFalse);
    });

    test('A poor GNSS fix contributes nothing', () {
      final c = OnlineCalibration();
      for (var i = 0; i < 2000; i++) {
        c.observeSpeed(
          modelSpeed: 100.0,
          gnssSpeed: 15.0,
          gnssAccuracy: 60.0, // urban canyon multipath
          mountDisturbed: false,
        );
      }
      expect(c.speedSamples, 0);
    });

    test('Straight-line driving teaches nothing about scale', () {
      // Dividing a near-zero measured turn by a near-zero predicted one is pure noise.
      final c = OnlineCalibration();
      var course = 0.0;
      for (var i = 0; i < 4000; i++) {
        course += 0.0005 * (i.isEven ? 1 : -1); // jitter, no real turn
        c.observeMovingWithGnss(
          compassYawRate: 0.0005 * (i.isEven ? 1 : -1),
          dt: 0.02,
          gnssSpeed: 25.0,
          gnssAccuracy: 3.0,
          courseRad: course,
          mountDisturbed: false,
        );
      }
      expect(c.headingWindows, 0);
      expect(c.yawScale, closeTo(1.0, 1e-9));
    });
  });

  test('Learned state survives a save/load round trip', () {
    final a = OnlineCalibration();
    for (var i = 0; i < 3000; i++) {
      a.observeStationary(trueBias, 0.02);
    }
    final store = InMemoryCalibrationStore()..save(a.toJson());

    final b = OnlineCalibration()..loadJson(store.load()!);
    expect(b.yawBias, closeTo(a.yawBias, 1e-9));
    expect(b.yawBiasConfidence, closeTo(a.yawBiasConfidence, 1e-9));
    expect(b.correctYawRate(0.3), closeTo(a.correctYawRate(0.3), 1e-9));
  });
}
