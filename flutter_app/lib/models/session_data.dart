/// One logged telemetry row. Deliberately flat and explicit: this is the record that
/// gets exported for offline analysis, and a schema you have to decode later is a schema
/// nobody uses.
class SessionSample {
  final double timestampS;
  final double gnssLat, gnssLon, gnssSpeed, gnssAccuracy;
  final String gnssMode;
  final double ax, ay, az, gx, gy, gz;
  final double mlSpeed, mlUncertainty;
  final double ekfEast, ekfNorth, ekfSpeed, ekfHeading, ekfGyroBias;
  final double qualityScore;

  const SessionSample({
    required this.timestampS,
    required this.gnssLat,
    required this.gnssLon,
    required this.gnssSpeed,
    required this.gnssAccuracy,
    required this.gnssMode,
    required this.ax,
    required this.ay,
    required this.az,
    required this.gx,
    required this.gy,
    required this.gz,
    required this.mlSpeed,
    required this.mlUncertainty,
    required this.ekfEast,
    required this.ekfNorth,
    required this.ekfSpeed,
    required this.ekfHeading,
    required this.ekfGyroBias,
    required this.qualityScore,
  });

  List<Object> toCsvRow() => [
        timestampS,
        gnssLat,
        gnssLon,
        gnssSpeed,
        gnssAccuracy,
        gnssMode,
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        mlSpeed,
        mlUncertainty,
        ekfEast,
        ekfNorth,
        ekfSpeed,
        ekfHeading,
        ekfGyroBias,
        qualityScore,
      ];

  static const String csvHeader =
      'timestamp_s,gnss_lat,gnss_lon,gnss_speed,gnss_accuracy,gnss_mode,'
      'ax,ay,az,gx,gy,gz,ml_speed,ml_uncertainty,'
      'ekf_east,ekf_north,ekf_speed,ekf_heading,ekf_gyro_bias,quality_score';
}

class SessionEvent {
  final double timestampS;
  final String label;
  const SessionEvent(this.timestampS, this.label);
}

/// Post-drive summary. Every field is measured from the samples, never assumed.
class SessionSummary {
  final String sessionName;
  final DateTime startedAt;
  final double durationS;
  final int totalSamples;
  final double gnssAvailablePct;
  final double gnssDeniedPct;
  final double meanMlSpeedMps;
  final double meanGnssSpeedMps;

  /// Mean |ML speed - GNSS speed| while GNSS was available. This is the model's honest
  /// error against the only reference the phone has, and it is what the EKF's assumed
  /// sigma should be set from.
  final double meanSpeedDiffMps;

  final double distanceM;
  final double accelHz;
  final double gyroHz;
  final double meanQualityScore;
  final List<SessionEvent> events;

  const SessionSummary({
    required this.sessionName,
    required this.startedAt,
    required this.durationS,
    required this.totalSamples,
    required this.gnssAvailablePct,
    required this.gnssDeniedPct,
    required this.meanMlSpeedMps,
    required this.meanGnssSpeedMps,
    required this.meanSpeedDiffMps,
    required this.distanceM,
    required this.accelHz,
    required this.gyroHz,
    required this.meanQualityScore,
    required this.events,
  });
}
