import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/localization/frame_alignment.dart';

/// Reproduces the reported defect:
///
///   "I tilt my phone a little slant while moving and the system detected me
///    changing my direction."
///
/// The rig below synthesises a physically consistent IMU stream. Given a rotation
/// R (vehicle -> phone body) and an angular velocity, it produces exactly the
/// accelerometer and gyroscope readings a phone would report, so the test exercises the
/// same arithmetic the live pipeline does rather than a hand-waved approximation.
///
/// Convention: vehicle frame has +Z up, so specific force at rest is +g on vehicle Z.
/// A phone lying flat, screen up, reads gravity on body +Z.

const double g0 = 9.80665;

/// Rotation about an arbitrary unit axis by [angle], Rodrigues form.
List<List<double>> rotAxis(List<double> axis, double angle) {
  final n = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  final x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
  final c = math.cos(angle), s = math.sin(angle), t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

List<List<double>> matMul(List<List<double>> a, List<List<double>> b) =>
    List.generate(3, (i) => List.generate(3, (j) {
          var s = 0.0;
          for (var k = 0; k < 3; k++) {
            s += a[i][k] * b[k][j];
          }
          return s;
        }));

/// v expressed in the body frame, given R maps body -> vehicle.
List<double> toBody(List<List<double>> r, List<double> v) =>
    List.generate(3, (i) {
      var s = 0.0;
      for (var k = 0; k < 3; k++) {
        s += r[k][i] * v[k]; // R^T * v
      }
      return s;
    });

/// Integrates a phone-motion scenario through FrameAlignment exactly as
/// NavigationStateProvider does, and returns the heading it would have accumulated.
///
/// [phoneRate] returns the phone's angular velocity relative to the vehicle, in BODY
/// axes, at time t. [vehicleYawRate] is the vehicle's own turn rate about vertical.
double integratedHeadingDeg({
  required double seconds,
  required List<double> Function(double t) phoneRate,
  double vehicleYawRate = 0.0,
  double dt = 0.01,
  double forwardAccel = 0.0,
  FrameAlignment? sharedFrame,
}) {
  final frame = sharedFrame ?? FrameAlignment();
  var r = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ];
  var heading = 0.0;

  // Settle the gravity estimate first, the way a real session starts stationary.
  for (var i = 0; i < 200; i++) {
    frame.update(0, 0, g0, 0, 0, 0, dt);
  }

  final steps = (seconds / dt).round();
  for (var i = 0; i < steps; i++) {
    final t = i * dt;
    final pr = phoneRate(t);

    // Total body angular velocity = vehicle rotation (about vehicle +Z, expressed in
    // body axes) plus the phone's rotation relative to the vehicle.
    final vehOmegaBody = toBody(r, [0.0, 0.0, vehicleYawRate]);
    final wx = pr[0] + vehOmegaBody[0];
    final wy = pr[1] + vehOmegaBody[1];
    final wz = pr[2] + vehOmegaBody[2];

    // Specific force in the vehicle frame: gravity plus forward acceleration along +Y.
    final fBody = toBody(r, [0.0, forwardAccel, g0]);

    frame.update(fBody[0], fBody[1], fBody[2], wx, wy, wz, dt);
    heading += frame.compassYawRate * dt;

    // Advance the body orientation by the phone-relative rotation only; the vehicle
    // frame itself is what heading is measured against.
    final mag = math.sqrt(pr[0] * pr[0] + pr[1] * pr[1] + pr[2] * pr[2]);
    if (mag > 1e-12) {
      r = matMul(r, rotAxis(pr, mag * dt));
    }
  }
  return heading * 180.0 / math.pi;
}

void main() {
  group('Phone tilt must not be read as a change of vehicle direction', () {
    test('A pure single-axis tilt produces no heading change', () {
      // Tipping the phone back 30 degrees over 1.2 s, about the body X axis only.
      // omega is perpendicular to gravity throughout, so the projection onto gravity is
      // zero and this case should already be clean. It is the control.
      final deg = integratedHeadingDeg(
        seconds: 3.0,
        phoneRate: (t) => (t > 0.6 && t < 1.8)
            ? [30 * math.pi / 180 / 1.2, 0.0, 0.0]
            : [0.0, 0.0, 0.0],
      );
      expect(deg.abs(), lessThan(2.0),
          reason: 'pure tilt about a horizontal axis leaked $deg° into heading');
    });

    test('A realistic hand slant does not swing the heading', () {
      // Real hands do not rotate about one clean axis. Slanting a phone couples a tilt
      // with an incidental twist about the vertical - and that twist is genuine phone
      // yaw, indistinguishable from a vehicle turn by gyroscope alone. This is the case
      // the user actually hit.
      const tiltRate = 25 * math.pi / 180 / 1.0; // 25 deg of tilt over 1 s
      const twistRate = 8 * math.pi / 180 / 1.0; // 8 deg of incidental twist
      final deg = integratedHeadingDeg(
        seconds: 4.0,
        forwardAccel: 0.0,
        phoneRate: (t) =>
            (t > 1.0 && t < 2.0) ? [tiltRate, 0.0, twistRate] : [0.0, 0.0, 0.0],
      );
      expect(deg.abs(), lessThan(3.0),
          reason: 'a hand slant moved the reported heading by $deg°, so the system '
              'believes the vehicle turned when only the phone did');
    });

    test('Re-seating the phone in the cradle does not rotate the map', () {
      // Picking the phone up, turning it 40 degrees, and putting it back.
      const rate = 40 * math.pi / 180 / 0.8;
      final deg = integratedHeadingDeg(
        seconds: 4.0,
        phoneRate: (t) => (t > 1.0 && t < 1.8)
            ? [rate * 0.5, rate * 0.3, rate * 0.8]
            : [0.0, 0.0, 0.0],
      );
      expect(deg.abs(), lessThan(5.0),
          reason: 're-seating the phone moved heading by $deg°');
    });

    test('A genuine vehicle turn IS still tracked', () {
      // The guard must not be so aggressive that it suppresses real turns. The vehicle
      // holds 15 deg/s for the whole 8 s window, so the truth is 120 deg, not 90.
      // Measured before the guard existed: 120.0000000000004 - yaw integration is exact
      // when the phone is still, which is what makes the tilt case a pure rejection
      // problem rather than an accuracy problem.
      const yaw = 90 * math.pi / 180 / 6.0;
      final deg = integratedHeadingDeg(
        seconds: 8.0,
        vehicleYawRate: yaw,
        phoneRate: (_) => [0.0, 0.0, 0.0],
      );
      expect(deg.abs(), greaterThan(114.0),
          reason: 'a real turn was only tracked as $deg° of 120° - the guard is too strict');
      expect(deg.abs(), lessThan(126.0));
    });

    test('Turning RIGHT increases the compass bearing', () {
      // Sign convention, pinned because the two halves of this codebase disagreed.
      //
      //   EKF heading is a compass bearing: east += v*sin(psi), north += v*cos(psi),
      //   so psi = 0 is North and psi = +90 is East - CLOCKWISE positive.
      //   yawRate = omega . gHat, and gHat points UP (an accelerometer at rest reads +g
      //   along the upward axis), so by the right-hand rule it is ANTICLOCKWISE positive.
      //
      // Feeding one straight into the other inverts every turn: the vehicle goes right and
      // the map swings left. The pedestrian path had already been patched with a minus
      // sign - pedestrian.integrateYaw(-frame.vehicleYawRate) into a parameter literally
      // named clockwiseRate - while the EKF path had not.
      //
      // Here the rig's vehicleYawRate is about vehicle +Z (up), so a RIGHT turn is
      // negative, and a right turn from North must read as a bearing heading toward +90.
      const rightTurn = -(90 * math.pi / 180 / 6.0);
      final deg = integratedHeadingDeg(
        seconds: 6.0,
        vehicleYawRate: rightTurn,
        phoneRate: (_) => [0.0, 0.0, 0.0],
      );
      expect(deg, greaterThan(80.0),
          reason: 'a right turn produced a bearing of $deg°; it should approach +90° '
              '(East). A negative value means the map turns the wrong way.');
      expect(deg, lessThan(100.0));
    });

    test('A vehicle turn is still tracked while the phone is gently tilted', () {
      // The hard case: both happen at once. The real turn must survive.
      const yaw = 90 * math.pi / 180 / 6.0;
      final deg = integratedHeadingDeg(
        seconds: 8.0,
        vehicleYawRate: yaw,
        phoneRate: (t) => (t > 2.0 && t < 2.8)
            ? [15 * math.pi / 180 / 0.8, 0.0, 0.0]
            : [0.0, 0.0, 0.0],
      );
      expect(deg.abs(), greaterThan(70.0),
          reason: 'a real turn was lost ($deg°) because the phone was tilted during it');
    });
  });
}
