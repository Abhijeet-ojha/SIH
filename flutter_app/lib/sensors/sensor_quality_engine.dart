import 'dart:math' as math;

import '../models/sensor_quality.dart';

/// Tracks the ACTUAL delivery rate of one sensor stream.
///
/// Android's SENSOR_DELAY_GAME is a request, not a promise: the real rate varies by
/// handset, by thermal state, and by what else is running. Everything downstream integrates
/// these samples, so a stream quietly delivering 12 Hz instead of 50 Hz corrupts the
/// position estimate while looking fine. Measure it rather than assume it.
class SensorRateTracker {
  final String name;
  final double nominalHz;

  final List<int> _deltas = [];
  int _lastTimestampMs = 0;
  int _sampleCount = 0;
  double _actualHz = 0.0;
  double _medianDeltaTMs = 0.0;
  double _jitterMs = 0.0;

  static const int _historyLength = 64;

  SensorRateTracker({required this.name, required this.nominalHz});

  int get sampleCount => _sampleCount;
  int get lastTimestampMs => _lastTimestampMs;

  void recordEvent({required int timestampMs}) {
    if (_lastTimestampMs > 0) {
      final d = timestampMs - _lastTimestampMs;
      if (d > 0) {
        _deltas.add(d);
        while (_deltas.length > _historyLength) {
          _deltas.removeAt(0);
        }
      }
    }
    _lastTimestampMs = timestampMs;
    _sampleCount++;
  }

  void computeRateMetrics(int nowMs) {
    if (_deltas.isEmpty) {
      _actualHz = 0.0;
      _medianDeltaTMs = 0.0;
      _jitterMs = 0.0;
      return;
    }
    final sorted = List<int>.from(_deltas)..sort();
    _medianDeltaTMs = sorted[sorted.length ~/ 2].toDouble();
    _actualHz = _medianDeltaTMs > 0 ? 1000.0 / _medianDeltaTMs : 0.0;

    // Jitter as mean absolute deviation from the median - robust to the occasional long
    // gap in a way that a standard deviation is not.
    var acc = 0.0;
    for (final d in _deltas) {
      acc += (d - _medianDeltaTMs).abs();
    }
    _jitterMs = acc / _deltas.length;
  }

  SensorDiagnostic getDiagnostic(
    int nowMs, {
    int staleThresholdMs = 150,
    int unavailableThresholdMs = 600,
  }) {
    final age = _lastTimestampMs == 0 ? 1 << 30 : nowMs - _lastTimestampMs;
    final freshness = age >= unavailableThresholdMs
        ? SensorFreshnessStatus.unavailable
        : (age >= staleThresholdMs
            ? SensorFreshnessStatus.stale
            : SensorFreshnessStatus.live);

    return SensorDiagnostic(
      sensorName: name,
      actualHz: _actualHz,
      nominalHz: nominalHz,
      medianDeltaTMs: _medianDeltaTMs,
      jitterMs: _jitterMs,
      ageMs: age,
      sampleCount: _sampleCount,
      freshnessStatus: freshness,
      healthGrade: _grade(freshness),
    );
  }

  SensorHealthGrade _grade(SensorFreshnessStatus freshness) {
    if (freshness == SensorFreshnessStatus.unavailable)
      return SensorHealthGrade.dead;
    if (_sampleCount < 2 || nominalHz <= 0) return SensorHealthGrade.degraded;
    final ratio = _actualHz / nominalHz;
    final jitterFrac = _medianDeltaTMs > 0 ? _jitterMs / _medianDeltaTMs : 1.0;
    if (ratio > 0.85 && jitterFrac < 0.15) return SensorHealthGrade.excellent;
    if (ratio > 0.70 && jitterFrac < 0.30) return SensorHealthGrade.good;
    if (ratio > 0.45) return SensorHealthGrade.degraded;
    return SensorHealthGrade.poor;
  }

  void reset() {
    _deltas.clear();
    _lastTimestampMs = 0;
    _sampleCount = 0;
    _actualHz = _medianDeltaTMs = _jitterMs = 0.0;
  }
}

/// Watches all IMU streams for rate problems and for values that are physically impossible.
class SensorQualityEngine {
  final SensorRateTracker accel =
      SensorRateTracker(name: 'ACCELEROMETER', nominalHz: 50.0);
  final SensorRateTracker gyro =
      SensorRateTracker(name: 'GYROSCOPE', nominalHz: 50.0);
  final SensorRateTracker mag =
      SensorRateTracker(name: 'MAGNETOMETER', nominalHz: 25.0);

  final List<SensorAnomaly> _anomalies = [];
  final Set<String> _faults = {};

  /// Consumer accelerometers clip around +/-40 m/s^2. A reading at the rail is the sensor
  /// saturating, not the car doing 4 g, and integrating it produces confident nonsense.
  static const double accelSaturation = 39.0;
  static const double gyroSaturation = 16.0;

  static const int _maxAnomalies = 40;

  void recordAccelEvent(double x, double y, double z, int tMs) {
    accel.recordEvent(timestampMs: tMs);
    if (x.isNaN || y.isNaN || z.isNaN) {
      _flag('ACCEL', 'ACCEL_NAN', 'non-finite accelerometer sample', tMs);
    } else if (x.abs() > accelSaturation ||
        y.abs() > accelSaturation ||
        z.abs() > accelSaturation) {
      _flag(
          'ACCEL',
          'ACCEL_SATURATED',
          'axis at the sensor rail (${accelSaturation.toStringAsFixed(0)} m/s²)',
          tMs);
    }
  }

  void recordGyroEvent(double x, double y, double z, int tMs) {
    gyro.recordEvent(timestampMs: tMs);
    if (x.isNaN || y.isNaN || z.isNaN) {
      _flag('GYRO', 'GYRO_NAN', 'non-finite gyroscope sample', tMs);
    } else if (x.abs() > gyroSaturation ||
        y.abs() > gyroSaturation ||
        z.abs() > gyroSaturation) {
      _flag('GYRO', 'GYRO_SATURATED', 'angular rate at the sensor rail', tMs);
    }
  }

  void recordMagEvent(double x, double y, double z, int tMs) {
    mag.recordEvent(timestampMs: tMs);
    final norm = math.sqrt(x * x + y * y + z * z);
    // Earth's field is 25-65 uT. Well outside that means a magnet, a speaker, or the car
    // body - the heading it implies is not worth having.
    if (norm > 120.0 || (norm < 10.0 && norm > 0)) {
      _flag(
          'MAG',
          'MAG_DISTURBED',
          'field ${norm.toStringAsFixed(0)} µT is outside the geomagnetic range',
          tMs);
    }
  }

  void _flag(String sensor, String code, String detail, int tMs) {
    _anomalies.add(SensorAnomaly(sensor, code, detail, tMs));
    while (_anomalies.length > _maxAnomalies) {
      _anomalies.removeAt(0);
    }
    _faults.add(code);
  }

  SensorQualityReport generateReport(int nowMs) {
    accel.computeRateMetrics(nowMs);
    gyro.computeRateMetrics(nowMs);
    mag.computeRateMetrics(nowMs);

    final a = accel.getDiagnostic(nowMs);
    final g = gyro.getDiagnostic(nowMs);
    final m = mag.getDiagnostic(nowMs);

    var score = 1.0;
    score -= 0.15 * _faults.length;
    for (final d in [a, g]) {
      if (d.freshnessStatus == SensorFreshnessStatus.unavailable) {
        score -= 0.4;
      } else if (d.freshnessStatus == SensorFreshnessStatus.stale) {
        score -= 0.15;
      }
      if (d.nominalHz > 0 && d.actualHz < 0.7 * d.nominalHz) score -= 0.1;
    }

    return SensorQualityReport(
      accel: a,
      gyro: g,
      mag: m,
      recentAnomalies: List.unmodifiable(_anomalies),
      activeFaults: Set.unmodifiable(_faults),
      overallScore: score.clamp(0.0, 1.0),
    );
  }

  void clearFaults() => _faults.clear();

  void reset() {
    accel.reset();
    gyro.reset();
    mag.reset();
    _anomalies.clear();
    _faults.clear();
  }
}
