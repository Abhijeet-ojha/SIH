import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/sensors/sensor_quality_engine.dart';
import 'package:navpulse_localizer/models/sensor_quality.dart';

void main() {
  group('SensorRateTracker & QualityEngine Tests', () {
    test('Calculates actual sampling rate and jitter accurately', () {
      final tracker = SensorRateTracker(name: 'ACCELEROMETER', nominalHz: 50.0);

      // Simulate 50 Hz events (20 ms interval) with small jitter
      int t = 1000;
      final intervals = [20, 22, 19, 21, 20, 18, 20, 21, 19, 20];
      for (final dt in intervals) {
        t += dt;
        tracker.recordEvent(timestampMs: t);
      }

      tracker.computeRateMetrics(t);
      final diag = tracker.getDiagnostic(t, staleThresholdMs: 150, unavailableThresholdMs: 600);

      expect(diag.sensorName, equals('ACCELEROMETER'));
      expect(diag.actualHz, greaterThan(45.0));
      expect(diag.actualHz, lessThan(55.0));
      expect(diag.medianDeltaTMs, closeTo(20.0, 2.0));
      expect(diag.jitterMs, greaterThanOrEqualTo(0.0));
      expect(diag.freshnessStatus, equals(SensorFreshnessStatus.live));
      expect(diag.healthGrade, isIn([SensorHealthGrade.excellent, SensorHealthGrade.good]));
    });

    test('Transitions from LIVE to STALE to UNAVAILABLE based on age', () {
      final tracker = SensorRateTracker(name: 'GYROSCOPE', nominalHz: 50.0);
      tracker.recordEvent(timestampMs: 1000);
      tracker.computeRateMetrics(1000);

      // 50ms later -> LIVE
      final liveDiag = tracker.getDiagnostic(1050, staleThresholdMs: 150, unavailableThresholdMs: 600);
      expect(liveDiag.freshnessStatus, equals(SensorFreshnessStatus.live));

      // 300ms later -> STALE
      final staleDiag = tracker.getDiagnostic(1300, staleThresholdMs: 150, unavailableThresholdMs: 600);
      expect(staleDiag.freshnessStatus, equals(SensorFreshnessStatus.stale));

      // 1000ms later -> UNAVAILABLE
      final unavailDiag = tracker.getDiagnostic(2000, staleThresholdMs: 150, unavailableThresholdMs: 600);
      expect(unavailDiag.freshnessStatus, equals(SensorFreshnessStatus.unavailable));
    });

    test('Detects sensor anomalies and generates diagnostic report', () {
      final engine = SensorQualityEngine();

      engine.recordAccelEvent(0.1, 0.2, 9.81, 1000);
      engine.recordAccelEvent(double.nan, 0.2, 9.81, 1020); // NaN anomaly
      engine.recordAccelEvent(42.0, 0.0, 0.0, 1040); // Saturation anomaly

      final report = engine.generateReport(1050);

      expect(report.accel.sampleCount, equals(3));
      expect(report.recentAnomalies.length, greaterThanOrEqualTo(1));
      expect(report.activeFaults, contains('ACCEL_SATURATED'));
      expect(report.overallScore, lessThan(1.0));
    });
  });
}
