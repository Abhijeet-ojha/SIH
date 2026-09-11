import 'dart:math' as math;

enum CalibrationStatus { uncalibrated, calibrating, completed, failed }

class CalibrationResult {
  final double accelBiasX, accelBiasY, accelBiasZ;
  final double gyroBiasX, gyroBiasY, gyroBiasZ;
  final double gravityMagnitude;
  final int totalSamplesCollected;
  final DateTime completedAt;

  const CalibrationResult({
    required this.accelBiasX,
    required this.accelBiasY,
    required this.accelBiasZ,
    required this.gyroBiasX,
    required this.gyroBiasY,
    required this.gyroBiasZ,
    required this.gravityMagnitude,
    required this.totalSamplesCollected,
    required this.completedAt,
  });

  double get gyroBiasMagnitude => math.sqrt(
      gyroBiasX * gyroBiasX + gyroBiasY * gyroBiasY + gyroBiasZ * gyroBiasZ);
}

/// Stationary bias calibration.
///
/// A MEMS gyro sitting perfectly still still reports a small non-zero rate. That offset is
/// integrated directly into heading: 0.03 rad/s unremoved is 155 degrees of heading error
/// over a 90-second blackout, which is the difference between arriving on the right road
/// and arriving on a different one. Measuring it takes a few seconds of stillness.
///
/// The accelerometer's gravity component is deliberately NOT removed - only the horizontal
/// bias. Gravity is a signal, not an error; the frame alignment needs it.
class SensorCalibrator {
  CalibrationStatus status = CalibrationStatus.uncalibrated;
  CalibrationResult? currentCalibration;

  bool get isCalibrated =>
      status == CalibrationStatus.completed && currentCalibration != null;

  final List<double> _ax = [], _ay = [], _az = [];
  final List<double> _gx = [], _gy = [], _gz = [];
  double _durationSec = 4.0;
  double _elapsed = 0.0;

  /// Motion rejection. If the phone moved during the wizard the "bias" we would measure is
  /// actually motion, and baking it in is worse than not calibrating at all.
  static const double maxAccelStd = 0.35; // m/s^2
  static const double maxGyroStd = 0.08; // rad/s
  static const int minSamples = 20;

  String message = 'Not calibrated. Park the vehicle and hold still.';

  void startCalibration({double durationSec = 4.0}) {
    status = CalibrationStatus.calibrating;
    _durationSec = durationSec;
    _elapsed = 0.0;
    currentCalibration = null;
    for (final b in [_ax, _ay, _az, _gx, _gy, _gz]) {
      b.clear();
    }
    message = 'Hold still…';
  }

  double get progress {
    if (status != CalibrationStatus.calibrating) {
      return status == CalibrationStatus.completed ? 1.0 : 0.0;
    }
    if (_durationSec <= 0) {
      return (_ax.length / minSamples).clamp(0.0, 1.0);
    }
    return (_elapsed / _durationSec).clamp(0.0, 1.0);
  }

  void feedRawSample({
    required double ax,
    required double ay,
    required double az,
    required double gx,
    required double gy,
    required double gz,
    double dt = 0.02,
  }) {
    if (status != CalibrationStatus.calibrating) return;
    _ax.add(ax);
    _ay.add(ay);
    _az.add(az);
    _gx.add(gx);
    _gy.add(gy);
    _gz.add(gz);
    _elapsed += dt;

    final enoughTime = _durationSec <= 0 || _elapsed >= _durationSec;
    if (enoughTime && _ax.length >= minSamples) _finish();
  }

  void _finish() {
    final moved = _std(_ax) > maxAccelStd ||
        _std(_ay) > maxAccelStd ||
        _std(_az) > maxAccelStd ||
        _std(_gx) > maxGyroStd ||
        _std(_gy) > maxGyroStd ||
        _std(_gz) > maxGyroStd;

    if (moved) {
      status = CalibrationStatus.failed;
      currentCalibration = null;
      message = 'Motion detected during calibration. Park, then try again.';
      return;
    }

    final gMag = math.sqrt(math.pow(_mean(_ax), 2) +
        math.pow(_mean(_ay), 2) +
        math.pow(_mean(_az), 2));

    currentCalibration = CalibrationResult(
      // Recorded for display only - NOT subtracted. See applyCorrection.
      accelBiasX: _mean(_ax),
      accelBiasY: _mean(_ay),
      accelBiasZ: _mean(_az),
      gyroBiasX: _mean(_gx),
      gyroBiasY: _mean(_gy),
      gyroBiasZ: _mean(_gz),
      gravityMagnitude: gMag,
      totalSamplesCollected: _ax.length,
      completedAt: DateTime.now(),
    );
    status = CalibrationStatus.completed;
    message = 'Calibrated on ${_ax.length} stationary samples.';
  }

  /// Returns the corrected sample: gyro bias removed, accelerometer untouched.
  ///
  /// The accelerometer is deliberately NOT corrected. A single static pose cannot separate
  /// sensor bias from gravity - at rest the whole reading IS gravity - so whatever is
  /// measured as "horizontal bias" in a tilted cradle is the gravity component along those
  /// axes. Subtracting it was measured to drop the resting magnitude to 7.51 m/s^2, which
  /// falls outside FrameAlignment.accelTrustBand (|a| - g must be within 1.5) and so
  /// silently switched off gravity correction for the entire session - turning a
  /// calibration step into a fault. Separating accel bias from gravity needs several
  /// distinct orientations, which is not what this wizard asks the user to do.
  (double, double, double, double, double, double) applyCorrection({
    required double rawAx,
    required double rawAy,
    required double rawAz,
    required double rawGx,
    required double rawGy,
    required double rawGz,
  }) {
    final c = currentCalibration;
    if (c == null) return (rawAx, rawAy, rawAz, rawGx, rawGy, rawGz);
    return (
      rawAx,
      rawAy,
      rawAz,
      rawGx - c.gyroBiasX,
      rawGy - c.gyroBiasY,
      rawGz - c.gyroBiasZ,
    );
  }

  void reset() {
    status = CalibrationStatus.uncalibrated;
    currentCalibration = null;
    message = 'Not calibrated. Park the vehicle and hold still.';
  }

  static double _mean(List<double> v) {
    if (v.isEmpty) return 0.0;
    var s = 0.0;
    for (final x in v) {
      s += x;
    }
    return s / v.length;
  }

  static double _std(List<double> v) {
    if (v.length < 2) return 0.0;
    final m = _mean(v);
    var s = 0.0;
    for (final x in v) {
      final d = x - m;
      s += d * d;
    }
    return math.sqrt(s / v.length);
  }
}
