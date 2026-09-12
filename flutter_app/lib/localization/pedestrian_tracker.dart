import 'dart:math' as math;

/// Relative pedestrian dead reckoning. Each detected gait cycle advances one
/// user-configured step; no GNSS, vehicle model, or invented playback is used.
///
/// ── Why this is more than a peak detector ────────────────────────────────────
///
/// Detecting steps from |a| alone is tilt-invariant, which is why it was chosen, but it is
/// also direction-blind: a hand waving the phone sideways at 2.5 Hz writes the same
/// magnitude signature as a 2 Hz gait, and every shake became metres on the map. The app
/// used to handle that by asking the presenter not to shake the phone.
///
/// Walking differs from shaking in three ways that a phone can actually measure:
///
///   1. DIRECTION. Walking drives the body up and down, so the step impulse lies along
///      gravity. Waving a phone drives it sideways. Tracking gravity as a VECTOR rather
///      than a magnitude makes that separable.
///   2. THE MOUNTING HOLDS STILL. A walker carrying a phone keeps it roughly fixed relative
///      to their body; a hand shaking one rotates it, so the gravity direction swings
///      through the body frame.
///   3. RHYTHM, AND PERSISTENCE OF RHYTHM. Human cadence sits in a narrow band and repeats.
///      A burst of shaking is erratic and short.
///
/// KNOWN LIMIT, stated rather than hidden: a sustained, rhythmic, purely vertical
/// oscillation at true walking cadence and amplitude, with the phone held perfectly steady,
/// is indistinguishable from walking by an accelerometer - because to an accelerometer it
/// IS walking. Rejecting that would need an independent observation of translation, which
/// is the thing dead reckoning is trying to produce. What is rejected here is every shake a
/// person actually performs.
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

  // ── Human gait bounds ──────────────────────────────────────────────────────
  /// Fastest plausible walking cadence (~2.5 steps/s). Below this is shaking, and the old
  /// 300 ms limit let a 3.3 Hz wave through.
  static const int minStepIntervalMs = 400;
  static const int maxStepIntervalMs = 1400;

  /// The step impulse must lie mostly along gravity. A sideways wave scores ~0.
  static const double minVerticalRatio = 0.7;

  /// Dispersion of the gravity unit vector over the recent window. A carried phone sits
  /// near 0.02; a shaken one swings well past this.
  static const double maxGravitySwing = 0.15;

  /// Vertical RMS ceiling for a footfall. Normal gait measures ~0.7 (gentle) to ~1.3
  /// (brisk) in this window; 4.5 leaves more than 3x headroom while rejecting a 9 m/s^2
  /// hand oscillation, which comes in at ~6.4. A person does not accelerate their own
  /// centre of mass that hard by walking.
  static const double maxStepAmplitude = 4.5;

  /// Consecutive consistent intervals required before distance is committed - about 2.5 s
  /// of steady cadence. This is what stops a brief burst: walking sustains a rhythm, a
  /// reflexive shake lasts a second or two. Everything held is credited retroactively the
  /// instant the gait locks, so a real walker loses nothing.
  static const int gaitLockSteps = 5;
  static const double gaitIntervalTolerance = 0.3; // +/-30% of the running median

  // Gravity as a vector, so direction is available and not just magnitude.
  final List<double> _gv = [0, 0, 9.80665];
  final List<List<double>> _gHist = [];
  double gravitySwing = 0;

  // Short-window RMS of the vertical and horizontal components of linear acceleration.
  double _vertSq = 0, _horizSq = 0;
  double verticalRatio = 0;

  // Steps detected but not yet credited, held until the gait locks.
  final List<double> _pendingHeadings = [];
  final List<int> _recentIntervals = [];
  bool gaitLocked = false;

  void reset() {
    east = north = heading = distanceM = signal = _speed = 0;
    steps = 0;
    _firstMs = _lastMs = _lastStepMs = _peakMs = null;
    _headingOrigin = null;
    _gravity = 9.80665;
    _valley = _peak = ready = false;
    _gv[0] = 0;
    _gv[1] = 0;
    _gv[2] = 9.80665;
    _gHist.clear();
    gravitySwing = 0;
    _vertSq = _horizSq = 0;
    verticalRatio = 0;
    _pendingHeadings.clear();
    _recentIntervals.clear();
    gaitLocked = false;
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

  /// Why the last candidate was refused, for the UI. A gate that rejects silently is
  /// indistinguishable from a broken sensor.
  String rejection = '';

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
      _breakGait();
      return false;
    }

    _updateDirection(ax, ay, az, dt);

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

    // ── A magnitude cycle completed. Now decide whether it was a STEP ────────
    final interval = _lastStepMs == null ? 600 : nowMs - _lastStepMs!;

    if (interval < minStepIntervalMs) {
      rejection = 'too fast for a human stride';
      return false;
    }
    if (verticalRatio < minVerticalRatio) {
      rejection = 'motion is sideways, not up and down';
      _breakGait();
      return false;
    }
    if (gravitySwing > maxGravitySwing) {
      rejection = 'the phone is being turned, not carried';
      _breakGait();
      return false;
    }
    if (math.sqrt(_vertSq) > maxStepAmplitude) {
      rejection = 'too violent to be a footfall';
      _breakGait();
      return false;
    }

    rejection = '';
    _lastStepMs = nowMs;
    _speed = stepLengthM / (interval.clamp(350, maxStepIntervalMs) / 1000);
    _registerCandidate(interval);
    return gaitLocked;
  }

  /// Gravity as a vector, plus how much the step impulse points along it.
  void _updateDirection(double ax, double ay, double az, double dt) {
    final a = 1 - math.exp(-dt / 0.8);
    _gv[0] += a * (ax - _gv[0]);
    _gv[1] += a * (ay - _gv[1]);
    _gv[2] += a * (az - _gv[2]);

    final gn = math.max(
        math.sqrt(_gv[0] * _gv[0] + _gv[1] * _gv[1] + _gv[2] * _gv[2]), 1e-9);
    final ux = _gv[0] / gn, uy = _gv[1] / gn, uz = _gv[2] / gn;

    _gHist.add([ux, uy, uz]);
    while (_gHist.length > 50) {
      _gHist.removeAt(0);
    }
    gravitySwing = _swing();

    // Linear acceleration, split along and across gravity.
    final lx = ax - _gv[0], ly = ay - _gv[1], lz = az - _gv[2];
    final vert = lx * ux + ly * uy + lz * uz;
    final hx = lx - vert * ux, hy = ly - vert * uy, hz = lz - vert * uz;
    final horiz = math.sqrt(hx * hx + hy * hy + hz * hz);

    // ~0.4 s energy window, long enough to span a footfall.
    final w = 1 - math.exp(-dt / 0.4);
    _vertSq += w * (vert * vert - _vertSq);
    _horizSq += w * (horiz * horiz - _horizSq);
    final h = math.sqrt(_horizSq);
    verticalRatio = h < 1e-6 ? 999.0 : math.sqrt(_vertSq) / h;
  }

  double _swing() {
    if (_gHist.length < 2) return 0;
    var acc = 0.0;
    for (var axis = 0; axis < 3; axis++) {
      var m = 0.0;
      for (final v in _gHist) {
        m += v[axis];
      }
      m /= _gHist.length;
      var s = 0.0;
      for (final v in _gHist) {
        final d = v[axis] - m;
        s += d * d;
      }
      acc += s / _gHist.length;
    }
    return math.sqrt(acc);
  }

  /// Hold a candidate step until the cadence proves itself, then credit the whole run.
  ///
  /// Retroactive crediting matters: without it the first few real steps would be dropped,
  /// step count and distance would disagree, and the map would permanently lag the walker.
  void _registerCandidate(int interval) {
    _recentIntervals.add(interval);
    while (_recentIntervals.length > gaitLockSteps) {
      _recentIntervals.removeAt(0);
    }

    if (!gaitLocked) {
      _pendingHeadings.add(heading);
      if (_recentIntervals.length >= gaitLockSteps && _rhythmic()) {
        gaitLocked = true;
        for (final h in _pendingHeadings) {
          _commit(h);
        }
        _pendingHeadings.clear();
      }
      return;
    }
    _commit(heading);
  }

  bool _rhythmic() {
    final sorted = List<int>.from(_recentIntervals)..sort();
    final median = sorted[sorted.length ~/ 2].toDouble();
    if (median <= 0) return false;
    for (final i in _recentIntervals) {
      if ((i - median).abs() / median > gaitIntervalTolerance) return false;
    }
    return true;
  }

  void _commit(double atHeading) {
    steps++;
    distanceM += stepLengthM;
    east += stepLengthM * math.sin(atHeading);
    north += stepLengthM * math.cos(atHeading);
  }

  /// Something non-gait happened, so the rhythm evidence is void.
  void _breakGait() {
    gaitLocked = false;
    _recentIntervals.clear();
    _pendingHeadings.clear();
  }
}
