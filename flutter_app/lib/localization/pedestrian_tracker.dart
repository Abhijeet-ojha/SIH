import 'dart:math' as math;

/// Relative pedestrian dead reckoning. Each detected gait cycle advances one
/// user-configured step; no GNSS, vehicle model, or invented playback is used.
class PedestrianTracker {
  double stepLengthM = 0.65;
  double threshold = 0.6;
  double east = 0, north = 0, heading = 0, distanceM = 0;
  int steps = 0;
  double signal = 0;
  int? _firstMs, _lastMs, _lastStepMs, _peakMs;
  double _gravity = 9.80665;
  bool _valley = false, _peak = false;
  double? _headingOrigin;
  double _speed = 0;
  bool ready = false;

  void reset() {
    east = north = heading = distanceM = signal = _speed = 0;
    steps = 0;
    _firstMs = _lastMs = _lastStepMs = _peakMs = null;
    _headingOrigin = null;
    _gravity = 9.80665;
    _valley = _peak = ready = false;
  }

  static double wrap(double v) => (v + math.pi) % (2 * math.pi) - math.pi;

  void observeHeading(double radians) {
    if (!radians.isFinite) return;
    _headingOrigin ??= radians;
    heading = wrap(radians - _headingOrigin!);
  }

  void integrateYaw(double clockwiseRate, double dt) {
    if (clockwiseRate.isFinite && dt > 0 && dt < 0.3) {
      heading = wrap(heading + clockwiseRate * dt);
    }
  }

  double speedAt(int nowMs) =>
      _lastStepMs != null && nowMs - _lastStepMs! < 1400 ? _speed : 0;

  /// Magnitude is independent of phone tilt. Hysteresis, a complete valley/peak
  /// cycle and a refractory period reject stationary noise and rapid duplicates.
  bool addAcceleration(double ax, double ay, double az, int nowMs) {
    final magnitude = math.sqrt(ax * ax + ay * ay + az * az);
    if (!magnitude.isFinite || magnitude < 2 || magnitude > 25) return false;
    _firstMs ??= nowMs;
    final dt = _lastMs == null ? 0.02 : (nowMs - _lastMs!) / 1000;
    if (dt <= 0) return false;
    _lastMs = nowMs;
    if (dt > 0.5) {
      _peak = _valley = false;
      signal = 0;
      return false;
    }
    _gravity += (1 - math.exp(-dt / 0.8)) * (magnitude - _gravity);
    signal += (1 - math.exp(-dt / 0.055)) * (magnitude - _gravity - signal);
    ready = nowMs - _firstMs! >= 1500;
    if (!ready) {
      _peak = _valley = false;
      return false;
    }
    if (signal < -threshold * 0.4) _valley = true;
    if (_valley && signal > threshold && !_peak) {
      _peak = true;
      _peakMs = nowMs;
    }
    if (_peak && nowMs - _peakMs! > 650) {
      _peak = _valley = false;
    }
    if (!_peak || signal > 0) return false;
    _peak = _valley = false;
    final interval = _lastStepMs == null ? 600 : nowMs - _lastStepMs!;
    if (interval < 300) return false;
    _lastStepMs = nowMs;
    _speed = stepLengthM / (interval.clamp(350, 1400) / 1000);
    steps++;
    distanceM += stepLengthM;
    east += stepLengthM * math.sin(heading);
    north += stepLengthM * math.cos(heading);
    return true;
  }
}
