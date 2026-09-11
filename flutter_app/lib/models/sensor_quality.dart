/// How recently a sensor produced a sample.
enum SensorFreshnessStatus { live, stale, unavailable }

/// Overall health of a sensor stream, from rate and jitter against its nominal rate.
enum SensorHealthGrade { excellent, good, degraded, poor, dead }

class SensorDiagnostic {
  final String sensorName;
  final double actualHz;
  final double nominalHz;
  final double medianDeltaTMs;
  final double jitterMs;
  final int ageMs;
  final int sampleCount;
  final SensorFreshnessStatus freshnessStatus;
  final SensorHealthGrade healthGrade;

  const SensorDiagnostic({
    required this.sensorName,
    required this.actualHz,
    required this.nominalHz,
    required this.medianDeltaTMs,
    required this.jitterMs,
    required this.ageMs,
    required this.sampleCount,
    required this.freshnessStatus,
    required this.healthGrade,
  });

  /// Rate as a fraction of nominal, clamped for display.
  double get rateRatio =>
      nominalHz <= 0 ? 0.0 : (actualHz / nominalHz).clamp(0.0, 1.5);
}

class SensorAnomaly {
  final String sensor;
  final String code;
  final String detail;
  final int timestampMs;
  const SensorAnomaly(this.sensor, this.code, this.detail, this.timestampMs);
}

class SensorQualityReport {
  final SensorDiagnostic accel;
  final SensorDiagnostic gyro;
  final SensorDiagnostic mag;
  final List<SensorAnomaly> recentAnomalies;
  final Set<String> activeFaults;

  /// 1.0 = every stream healthy. Drops with faults and with rate shortfall, so a single
  /// number can drive the UI badge without hiding which stream is at fault.
  final double overallScore;

  const SensorQualityReport({
    required this.accel,
    required this.gyro,
    required this.mag,
    required this.recentAnomalies,
    required this.activeFaults,
    required this.overallScore,
  });
}
