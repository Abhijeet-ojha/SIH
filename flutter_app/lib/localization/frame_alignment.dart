import 'dart:math' as math;

/// Phone-frame -> vehicle-frame alignment, ported from src/frame_alignment.py.
///
/// Nothing downstream may consume raw ax/ay/az or gyro z. Those encode how the phone
/// happens to be sitting in the cradle, so a model trained on them learns one mounting and
/// fails on the next. Everything here is built from quantities that do not change when the
/// whole phone is rotated:
///
///   aVert    = a . gHat                  (scalar)
///   aHoriz   = a - (a . gHat) gHat       (magnitude is invariant)
///   yawRate  = omega . gHat              (scalar)
///
/// plus a forward axis estimated from data, which rotates with the phone so that
/// a . fHat is invariant too.
///
/// The offline pipeline verifies this to 1e-14 across 12 random SO(3) rotations of a real
/// drive (tests/test_rotation_invariance.py).
class FrameAlignment {
  static const double g0 = 9.80665;

  /// Mahony complementary filter constants. A plain low-pass cannot do this job: one time
  /// constant has to both reject linear acceleration (wants slow) and track real tilt
  /// (wants fast), and at 0.2 Hz it resolved that by absorbing any sustained acceleration
  /// into "gravity". Measured on 32 real brake events, the low-pass retained 19% of the
  /// vehicle's actual deceleration; this filter retains 61%.
  static const double kp = 0.10;
  static const double ki = 0.002;
  static const double accelTrustBand = 1.5;

  final List<double> _g = [0.0, 0.0, g0];
  final List<double> _bias = [0.0, 0.0, 0.0];
  bool _initialised = false;

  /// Forward axis in the phone body frame, estimated from acceleration events.
  List<double> _fHat = [0.0, 1.0, 0.0];
  final List<List<double>> _fwdCov =
      List.generate(3, (_) => List.filled(3, 0.0));
  int _fwdEvents = 0;
  double forwardConfidence = 0.0;

  double aVert = 0.0;
  double aHorizMag = 0.0;
  double aFwd = 0.0;
  double aLat = 0.0;
  double yawRate = 0.0;
  double gyroMag = 0.0;
  double tiltRate = 0.0;
  double gravStability = 0.0;

  final List<List<double>> _gHist = [];
  int _stabilityWindow = 20;

  /// Feed one IMU sample. [gravity] is Sensor.TYPE_GRAVITY when the platform provides it;
  /// the sensor hub computes it more accurately than we can, for free.
  void update(
    double ax,
    double ay,
    double az,
    double gx,
    double gy,
    double gz,
    double dt, {
    List<double>? gravity,
  }) {
    if (!_initialised) {
      _g[0] = gravity?[0] ?? ax;
      _g[1] = gravity?[1] ?? ay;
      _g[2] = gravity?[2] ?? az;
      _stabilityWindow = math.max(2, (2.0 / (dt > 0 ? dt : 0.02)).round());
      _initialised = true;
    } else if (gravity != null) {
      _g[0] = gravity[0];
      _g[1] = gravity[1];
      _g[2] = gravity[2];
    } else {
      _propagateGravity(ax, ay, az, gx, gy, gz, dt);
    }

    final gn = math.max(
        math.sqrt(_g[0] * _g[0] + _g[1] * _g[1] + _g[2] * _g[2]), 1e-9);
    final ux = _g[0] / gn, uy = _g[1] / gn, uz = _g[2] / gn;

    final lx = ax - _g[0], ly = ay - _g[1], lz = az - _g[2];
    aVert = lx * ux + ly * uy + lz * uz;
    final hx = lx - aVert * ux, hy = ly - aVert * uy, hz = lz - aVert * uz;
    aHorizMag = math.sqrt(hx * hx + hy * hy + hz * hz);

    // Yaw is the gyro projected onto gravity. The z axis is only yaw if the phone happens
    // to be lying perfectly flat, which it never is in a cradle.
    yawRate = gx * ux + gy * uy + gz * uz;
    gyroMag = math.sqrt(gx * gx + gy * gy + gz * gz);
    tiltRate = math.sqrt(math.max(0.0, gyroMag * gyroMag - yawRate * yawRate));

    _pushGravityHistory([ux, uy, uz]);
    gravStability = _gravityStability();
    _updateMountTrust(dt);

    _lastHoriz = [hx, hy, hz];
    aFwd = hx * _fHat[0] + hy * _fHat[1] + hz * _fHat[2];
    final l = _cross([ux, uy, uz], _fHat);
    aLat = hx * l[0] + hy * l[1] + hz * l[2];
  }

  // ── Mount disturbance: is the PHONE turning, or is the VEHICLE turning? ────
  //
  // yawRate = omega . gHat is the rotation rate about the vertical. It is frame-invariant,
  // which makes it the right quantity - but it cannot tell whose rotation it is. Rotating
  // the phone in its cradle and driving round a bend look identical to a gyroscope.
  //
  // Physics does separate them. A vehicle turning on a road rotates about the vertical and
  // essentially nothing else, so its rotation has almost no component perpendicular to
  // gravity. A hand slanting a phone rotates it about a horizontal axis, which shows up
  // directly in tiltRate. Measured on the synthetic rig in test/tilt_heading_test.dart:
  //
  //   real 90 deg/6 s vehicle turn, phone still   tiltRate ~ 0.00 rad/s
  //   hand slant, 25 deg/s                        tiltRate ~ 0.44 rad/s
  //
  // So tiltRate is the discriminator, and yaw is believed in proportion to how still the
  // mounting is. A soft ramp rather than a hard gate, because a hard gate would throw away
  // a genuine turn the instant the phone was nudged.
  //
  // ponytail: this suppresses real vehicle yaw during a disturbance rather than recovering
  // it - during those ~1 s the two are genuinely not separable from the gyro alone.
  // Upgrade path is to estimate yaw from lateral specific force (a_lat = v * omega) while
  // the mounting is moving, which needs a speed estimate we do not yet trust enough.
  static const double tiltQuiet = 0.08;      // rad/s: road camber and pitch live below this
  static const double tiltDisturbed = 0.30;  // rad/s: unambiguously a hand
  static const double settleSeconds = 0.6;   // hold-off after the phone stops moving

  /// 0 = the measured yaw is not the vehicle's, 1 = fully trusted.
  double yawTrust = 1.0;

  /// The vehicle's yaw rate about the vertical, ANTICLOCKWISE-positive (the right-hand
  /// rule about gHat, which points up). Correct for physics, wrong for a compass.
  double vehicleYawRate = 0.0;

  /// The same rate as a COMPASS bearing rate: clockwise-positive, so that adding it to a
  /// heading where 0 = North and +90 = East turns the right way.
  ///
  /// This exists because the two halves of the codebase disagreed about the sign and the
  /// disagreement was invisible until someone drove round a corner. Heading is integrated
  /// as a bearing (east += v*sin(psi), north += v*cos(psi)) which is clockwise-positive,
  /// while omega . gHat is anticlockwise-positive. Feeding one into the other inverts
  /// every turn. The pedestrian path had been patched with a local minus sign; the EKF
  /// path had not. Both now take this, and the conversion is stated once, here.
  double get compassYawRate => -vehicleYawRate;

  /// True while the phone is moving relative to its mounting.
  bool mountDisturbed = false;

  /// Seconds of disturbance accumulated in the current session - surfaced in the UI so a
  /// held heading is explained rather than looking like a freeze.
  double disturbanceSeconds = 0.0;

  double _settleTimer = 0.0;

  void _updateMountTrust(double dt) {
    double t;
    if (tiltRate <= tiltQuiet) {
      t = 1.0;
    } else if (tiltRate >= tiltDisturbed) {
      t = 0.0;
    } else {
      t = 1.0 - (tiltRate - tiltQuiet) / (tiltDisturbed - tiltQuiet);
    }

    // After the phone stops moving the gravity estimate is still re-settling and the
    // mounting rotation has changed, so trust is ramped back rather than restored at once.
    if (t < 0.5) {
      _settleTimer = settleSeconds;
    } else if (_settleTimer > 0) {
      _settleTimer = math.max(0.0, _settleTimer - dt);
      t = math.min(t, 1.0 - _settleTimer / settleSeconds);
    }

    yawTrust = t.clamp(0.0, 1.0);
    mountDisturbed = yawTrust < 0.5;
    if (mountDisturbed) disturbanceSeconds += dt;
    vehicleYawRate = yawRate * yawTrust;
  }

  void _propagateGravity(double ax, double ay, double az, double gx, double gy,
      double gz, double dt) {
    final aNorm = math.sqrt(ax * ax + ay * ay + az * az);
    var wx = gx - _bias[0], wy = gy - _bias[1], wz = gz - _bias[2];

    // Only trust the accelerometer when it is plausibly reading gravity alone - i.e. NOT
    // while the vehicle is braking hard, which is exactly when we need the estimate to
    // hold still.
    if ((aNorm - g0).abs() < accelTrustBand && aNorm > 1e-6) {
      final gn = math.max(
          math.sqrt(_g[0] * _g[0] + _g[1] * _g[1] + _g[2] * _g[2]), 1e-9);
      final v = [_g[0] / gn, _g[1] / gn, _g[2] / gn];
      final aHat = [ax / aNorm, ay / aNorm, az / aNorm];
      final err = _cross(v, aHat);
      wx -= kp * err[0];
      wy -= kp * err[1];
      wz -= kp * err[2];
      _bias[0] += ki * err[0] * dt;
      _bias[1] += ki * err[1] * dt;
      _bias[2] += ki * err[2] * dt;
    }

    final gn = math.max(
        math.sqrt(_g[0] * _g[0] + _g[1] * _g[1] + _g[2] * _g[2]), 1e-9);
    final v = [_g[0] / gn, _g[1] / gn, _g[2] / gn];
    final rot = _cross([wx, wy, wz], v);
    for (var i = 0; i < 3; i++) {
      v[i] -= rot[i] * dt;
    }
    final n =
        math.max(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1e-9);
    for (var i = 0; i < 3; i++) {
      _g[i] = v[i] / n * g0;
    }
  }

  void _pushGravityHistory(List<double> u) {
    _gHist.add(u);
    while (_gHist.length > _stabilityWindow) {
      _gHist.removeAt(0);
    }
  }

  /// Dispersion of the gravity direction. ~0.01 for a phone in a cradle, past 0.3 for one
  /// being held. This single number is the most useful signal in the system for telling
  /// "the vehicle moved" from "a human moved the phone".
  double _gravityStability() {
    if (_gHist.length < 2) return 0.0;
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

  /// Accumulate the forward-axis estimate during genuine acceleration events while GNSS is
  /// healthy, then take the principal axis of horizontal acceleration.
  void observeForwardAxis(double dv, double speed) {
    if (dv.abs() < 0.35 || speed < 1.5) return;
    final v = _lastHoriz;
    if (v == null) return;
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        _fwdCov[i][j] += v[i] * v[j];
      }
    }
    _fwdEvents++;
    if (_fwdEvents < 25 || _fwdEvents % 10 != 0) return;

    var e = [_fHat[0], _fHat[1], _fHat[2]];
    for (var it = 0; it < 24; it++) {
      final n = [
        _fwdCov[0][0] * e[0] + _fwdCov[0][1] * e[1] + _fwdCov[0][2] * e[2],
        _fwdCov[1][0] * e[0] + _fwdCov[1][1] * e[1] + _fwdCov[1][2] * e[2],
        _fwdCov[2][0] * e[0] + _fwdCov[2][1] * e[1] + _fwdCov[2][2] * e[2],
      ];
      final nn = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (nn > 1e-12) e = [n[0] / nn, n[1] / nn, n[2] / nn];
    }
    final proj = v[0] * e[0] + v[1] * e[1] + v[2] * e[2];
    if (proj * dv < 0) e = [-e[0], -e[1], -e[2]];
    _fHat = e;

    final trace = _fwdCov[0][0] + _fwdCov[1][1] + _fwdCov[2][2];
    final along = e[0] *
            (_fwdCov[0][0] * e[0] +
                _fwdCov[0][1] * e[1] +
                _fwdCov[0][2] * e[2]) +
        e[1] *
            (_fwdCov[1][0] * e[0] +
                _fwdCov[1][1] * e[1] +
                _fwdCov[1][2] * e[2]) +
        e[2] *
            (_fwdCov[2][0] * e[0] +
                _fwdCov[2][1] * e[1] +
                _fwdCov[2][2] * e[2]);
    forwardConfidence = trace > 1e-12 ? along / trace : 0.0;
  }

  /// Horizontal specific force from the most recent update, in body frame.
  List<double>? _lastHoriz;

  List<double> _cross(List<double> a, List<double> b) => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ];

  void reset() {
    _initialised = false;
    yawTrust = 1.0;
    vehicleYawRate = 0.0;
    mountDisturbed = false;
    disturbanceSeconds = 0.0;
    _settleTimer = 0.0;
    _gHist.clear();
    _fwdEvents = 0;
    forwardConfidence = 0.0;
    _fHat = [0.0, 1.0, 0.0];
    for (final r in _fwdCov) {
      r.fillRange(0, 3, 0.0);
    }
    _bias.fillRange(0, 3, 0.0);
  }
}
