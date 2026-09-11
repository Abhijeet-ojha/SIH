import 'dart:math' as math;

import 'frame_alignment.dart';

enum MotionState { inVehicleMoving, stationary, phoneHandled }

/// Decides whether the phone is in a moving vehicle, parked, or being handled - and can
/// veto the speed estimate entirely.
///
/// This exists because the speed estimate is, at bottom, driven by vibration. A model
/// trained on a phone bolted into one car has never seen vibration WITHOUT motion, so it
/// cannot tell shaking from driving; the earlier on-device formula would happily integrate
/// a shaken phone into hundreds of metres of imaginary travel. The gate supplies that
/// distinction from physics rather than from training data.
///
/// Primary discriminator is gravity-direction stability: a cradled phone holds it to about
/// 0.01 over two seconds, a handled one swings past 0.3.
class MotionGate {
  // GNSS-available thresholds.
  static const double gravStabilityMax = 0.08;
  static const double tiltRateMax = 0.25;
  static const double stillAccRms = 0.12;
  static const double stillYawRate = 0.02;
  static const int debounceFrames = 5;

  // Blackout thresholds. The two error directions are NOT symmetric:
  //
  //   false "stopped" during a blackout - position freezes while the car keeps going.
  //     Every metre becomes along-track error, nothing observes it until GNSS returns, and
  //     map matching cannot recover it because map matching only fixes cross-track.
  //   false "stopped" with GNSS up - corrected on the very next fix. Nearly free.
  //
  // So while blind, demand much stronger evidence before declaring anything but MOVING.
  static const double blackoutStillAccRms = 0.06;
  static const double blackoutStillYawRate = 0.01;
  static const double blackoutGravStabilityMax = 0.16;
  static const double blackoutTiltRateMax = 0.50;
  static const int blackoutDebounceFrames = 12;

  // Corroboration. IMU quietness alone cannot separate "parked" from "gliding smoothly on
  // a good road" - both are silent. Measured: a 12 m/s cruise was frozen for an entire
  // 30 s outage, losing 360 m, before this precondition existed. Afterwards, 4.8 m.
  static const double blackoutStillMaxSpeed = 2.0;

  MotionState state = MotionState.stationary;
  MotionState _pending = MotionState.stationary;
  int _run = 0;

  final List<double> _tilt = [], _acc = [], _yaw = [];
  int _window = 20;
  bool _sized = false;

  /// Human-readable reason for the current verdict, shown under the state chip. A gate
  /// that vetoes without saying why is indistinguishable from a bug.
  String reason = 'Waiting for sensor data.';

  MotionState update(
    FrameAlignment f, {
    required double dt,
    required bool gnssAvailable,
    required double speedHint,
    bool stepDetected = false,
  }) {
    if (!_sized && dt > 0) {
      _window = math.max(4, (2.0 / dt).round());
      _sized = true;
    }
    _push(_tilt, f.tiltRate.abs());
    _push(_acc, f.aHorizMag);
    _push(_yaw, f.yawRate.abs());

    final blackout = !gnssAvailable;
    final gravTh = blackout ? blackoutGravStabilityMax : gravStabilityMax;
    final tiltTh = blackout ? blackoutTiltRateMax : tiltRateMax;
    final accTh = blackout ? blackoutStillAccRms : stillAccRms;
    final yawTh = blackout ? blackoutStillYawRate : stillYawRate;

    final tiltRms = _rms(_tilt);
    final accRms = _rms(_acc);
    final yawMean = _mean(_yaw);

    var handled = f.gravStability > gravTh || tiltRms > tiltTh || stepDetected;
    var still = accRms < accTh && yawMean < yawTh;
    if (blackout && speedHint > blackoutStillMaxSpeed) still = false;

    final MotionState raw;
    if (handled) {
      raw = MotionState.phoneHandled;
      reason = stepDetected
          ? 'Step detector fired — you are walking, not driving.'
          : 'Gravity direction is moving — speed estimate vetoed.';
    } else if (still) {
      raw = MotionState.stationary;
      reason = 'No horizontal force, no yaw. Position held.';
    } else {
      raw = MotionState.inVehicleMoving;
      reason = blackout
          ? 'Dead reckoning — GNSS withheld, inertial only.'
          : 'Gravity steady, phone tracking the vehicle.';
    }

    if (raw == _pending) {
      _run++;
    } else {
      _pending = raw;
      _run = 1;
    }
    // Leaving MOVING during a blackout needs sustained evidence; everything else uses the
    // normal debounce.
    final need = (blackout && raw != MotionState.inVehicleMoving)
        ? blackoutDebounceFrames
        : debounceFrames;
    if (_run >= need) state = _pending;
    return state;
  }

  void _push(List<double> b, double v) {
    b.add(v);
    while (b.length > _window) {
      b.removeAt(0);
    }
  }

  static double _mean(List<double> v) {
    if (v.isEmpty) return 0.0;
    var s = 0.0;
    for (final x in v) {
      s += x;
    }
    return s / v.length;
  }

  static double _rms(List<double> v) {
    if (v.isEmpty) return 0.0;
    var s = 0.0;
    for (final x in v) {
      s += x * x;
    }
    return math.sqrt(s / v.length);
  }

  void reset() {
    state = MotionState.stationary;
    _pending = MotionState.stationary;
    _run = 0;
    _tilt.clear();
    _acc.clear();
    _yaw.clear();
    reason = 'Waiting for sensor data.';
  }
}
