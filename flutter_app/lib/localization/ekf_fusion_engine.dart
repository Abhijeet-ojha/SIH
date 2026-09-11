import 'dart:math' as math;

/// GNSS availability state machine, surfaced to the UI so the user is never guessing
/// whether the position they are looking at came from satellites or from inertia.
enum GnssNavMode { gnssNormal, gnssDenied, gnssReacquired }

/// 6-state Joseph-form EKF, ported from src/fusion_ekf.py.
///
/// State: [posEast, posNorth, vFwd, vLat, heading, gyroBias]
///
/// Joseph form is used for every covariance update because the naive (I-KH)P form loses
/// symmetry under floating point and can drive P indefinite over a long blackout, which is
/// exactly the regime this filter exists for.
class EkfFusionEngine6State {
  final List<double> x;
  List<List<double>> _p;

  // Process noise
  final double qPos = 0.05;
  double qVelBase = 0.12;
  final double qVelLat = 0.05;
  final double qHeading = 0.003;
  final double qBias = 1e-6;

  // Measurement noise
  final double rGpsPos = 2.0;
  final double rGpsVel = 0.4;
  final double rGpsHeadingBase = 0.12;
  double nhcLateralVariance = 0.05 * 0.05;
  final double rZupt = 0.04 * 0.04;

  // Which subsystems have actually contributed. The UI shows these as live pipeline
  // indicators, so a stage that silently never runs is visible rather than assumed.
  bool isImuFused = false;
  bool isMlFused = false;
  bool isNhcFused = false;
  bool isZuptFused = false;
  bool isGnssFused = false;

  GnssNavMode gnssState = GnssNavMode.gnssNormal;
  double blackoutDurationS = 0.0;
  double _reacquireHoldS = 0.0;

  EkfFusionEngine6State({
    double initX = 0.0,
    double initY = 0.0,
    double initV = 0.0,
    double initVLat = 0.0,
    double initHeading = 0.0,
    double initGyroBias = 0.0,
    String driverStyle = 'normal',
  })  : x = [initX, initY, initV, initVLat, initHeading, initGyroBias],
        _p = _diag([4.0, 4.0, 1.0, 0.25, 0.05, 1e-4]) {
    if (driverStyle.toLowerCase() == 'aggressive') {
      nhcLateralVariance = 0.25 * 0.25;
      qVelBase = 0.20;
    }
  }

  double get posEast => x[0];
  double get posNorth => x[1];
  double get speed => x[2];
  double get lateralSpeed => x[3];
  double get heading => x[4];
  double get gyroBias => x[5];

  /// 1-sigma horizontal position uncertainty, in metres. This is what the UI shows as the
  /// headline number - it is the filter's own estimate of how wrong it might be, and it
  /// grows on its own during a blackout without anyone having to fake it.
  double get positionSigmaM => math.sqrt(math.max(0.0, _p[0][0] + _p[1][1]));

  void setGnssBlackout(bool denied) {
    if (denied) {
      if (gnssState != GnssNavMode.gnssDenied) blackoutDurationS = 0.0;
      gnssState = GnssNavMode.gnssDenied;
    } else if (gnssState == GnssNavMode.gnssDenied) {
      gnssState = GnssNavMode.gnssReacquired;
      _reacquireHoldS = 0.0;
    }
  }

  void predict({
    required double dt,
    required double vAi,
    required double vAiStd,
    required double gyroZ,
    bool isStationary = false,
  }) {
    final px = x[0], py = x[1], vF = x[2], vL = x[3], th = x[4], bg = x[5];

    if (gnssState == GnssNavMode.gnssDenied) blackoutDurationS += dt;
    if (gnssState == GnssNavMode.gnssReacquired) {
      _reacquireHoldS += dt;
      if (_reacquireHoldS > 5.0) gnssState = GnssNavMode.gnssNormal;
    }

    double alphaV, alphaLat, vFwdEff, vLatEff, thNew;
    if (isStationary) {
      alphaV = 0.0;
      alphaLat = 0.0;
      vFwdEff = 0.0;
      vLatEff = 0.0;
      thNew = th;
    } else {
      // Blend weight falls as the speed source gets less trustworthy. With the measured
      // blackout RMSE of 5.7 m/s this keeps the filter from chasing the model.
      alphaV = 0.25 / (1.0 + 1.5 * math.max(0.0, vAiStd));
      alphaLat = 0.10;
      vFwdEff = (1.0 - alphaV) * vF + alphaV * vAi;
      vLatEff = (1.0 - alphaLat) * vL;
      thNew = _wrap(th + (gyroZ - bg) * dt);
      isMlFused = true;
    }
    isImuFused = true;

    x[0] = px + (vFwdEff * math.sin(thNew) + vLatEff * math.cos(thNew)) * dt;
    x[1] = py + (vFwdEff * math.cos(thNew) - vLatEff * math.sin(thNew)) * dt;
    x[2] = vFwdEff;
    x[3] = vLatEff;
    x[4] = thNew;

    final qVelDyn = qVelBase + 1.5 * vAiStd * vAiStd;
    final q = _diag([
      qPos * qPos,
      qPos * qPos,
      qVelDyn,
      qVelLat * qVelLat,
      qHeading * qHeading,
      qBias * qBias,
    ]);

    final f = _eye();
    if (!isStationary) {
      final s = math.sin(thNew), c = math.cos(thNew);
      f[0][2] = (1 - alphaV) * s * dt;
      f[0][3] = (1 - alphaLat) * c * dt;
      f[0][4] = (vFwdEff * c - vLatEff * s) * dt;
      f[1][2] = (1 - alphaV) * c * dt;
      f[1][3] = -(1 - alphaLat) * s * dt;
      f[1][4] = -(vFwdEff * s + vLatEff * c) * dt;
      f[2][2] = 1 - alphaV;
      f[3][3] = 1 - alphaLat;
      f[4][5] = -dt;
    } else {
      f[2][2] = 0.0;
      f[3][3] = 0.0;
    }
    _p = _add(_mul(_mul(f, _p), _transpose(f)), q);
  }

  /// Non-holonomic constraint: a ground vehicle does not travel sideways.
  void updateNhc() {
    _scalarUpdate(3, 0.0, nhcLateralVariance, const [5]);
    isNhcFused = true;
  }

  /// Zero-velocity update, applied when the motion gate says we are genuinely stopped.
  void updateZupt() {
    _scalarUpdate(2, 0.0, rZupt, const [0, 1, 4, 5]);
    _scalarUpdate(3, 0.0, rZupt, const [0, 1, 4, 5]);
    isZuptFused = true;
  }

  void updateGps({
    required double east,
    required double north,
    required double gpsSpeed,
    double? gpsHeading,
  }) {
    if (gnssState == GnssNavMode.gnssDenied) return;
    _scalarUpdate(0, east, rGpsPos * rGpsPos, const [4, 5]);
    _scalarUpdate(1, north, rGpsPos * rGpsPos, const [4, 5]);
    _scalarUpdate(2, gpsSpeed, rGpsVel * rGpsVel, const [4, 5]);
    if (gpsHeading != null) {
      // GNSS course over ground is undefined at a standstill - it swings through the full
      // circle. Scale its variance up as speed falls instead of believing it.
      final vRef = math.max(gpsSpeed, 0.2);
      final rH =
          rGpsHeadingBase * rGpsHeadingBase * (1.0 + math.pow(1.5 / vRef, 2));
      _scalarUpdate(4, _wrap(gpsHeading), rH.toDouble(), const [0, 1, 2, 3],
          angular: true);
    }
    isGnssFused = true;
  }

  /// Single-measurement Joseph-form update on state [idx].
  /// [decoupled] lists states this measurement carries no observability of; their gain
  /// rows are zeroed, matching the partitioned update in the Python filter.
  void _scalarUpdate(int idx, double z, double r, List<int> decoupled,
      {bool angular = false}) {
    final innovation = angular ? _wrap(z - x[idx]) : z - x[idx];
    final s = _p[idx][idx] + r;
    if (s <= 0) return;
    final k = List<double>.generate(6, (i) => _p[i][idx] / s);
    for (final d in decoupled) {
      k[d] = 0.0;
    }
    for (var i = 0; i < 6; i++) {
      x[i] += k[i] * innovation;
    }
    if (angular) x[4] = _wrap(x[4]);

    // P = (I - KH) P (I - KH)^T + K R K^T
    final ikh = _eye();
    for (var i = 0; i < 6; i++) {
      ikh[i][idx] -= k[i];
    }
    final krk =
        List.generate(6, (i) => List.generate(6, (j) => k[i] * r * k[j]));
    _p = _add(_mul(_mul(ikh, _p), _transpose(ikh)), krk);
  }

  void reset({double initHeading = 0.0}) {
    for (var i = 0; i < 6; i++) {
      x[i] = 0.0;
    }
    x[4] = initHeading;
    _p = _diag([4.0, 4.0, 1.0, 0.25, 0.05, 1e-4]);
    isImuFused = isMlFused = isNhcFused = isZuptFused = isGnssFused = false;
    gnssState = GnssNavMode.gnssNormal;
    blackoutDurationS = 0.0;
  }

  static double _wrap(double a) {
    var v = (a + math.pi) % (2 * math.pi);
    if (v < 0) v += 2 * math.pi;
    return v - math.pi;
  }

  static List<List<double>> _eye() =>
      List.generate(6, (i) => List.generate(6, (j) => i == j ? 1.0 : 0.0));

  static List<List<double>> _diag(List<double> d) =>
      List.generate(6, (i) => List.generate(6, (j) => i == j ? d[i] : 0.0));

  static List<List<double>> _transpose(List<List<double>> m) =>
      List.generate(6, (i) => List.generate(6, (j) => m[j][i]));

  static List<List<double>> _add(List<List<double>> a, List<List<double>> b) =>
      List.generate(6, (i) => List.generate(6, (j) => a[i][j] + b[i][j]));

  static List<List<double>> _mul(List<List<double>> a, List<List<double>> b) =>
      List.generate(
          6,
          (i) => List.generate(6, (j) {
                var s = 0.0;
                for (var k = 0; k < 6; k++) {
                  s += a[i][k] * b[k][j];
                }
                return s;
              }));
}
