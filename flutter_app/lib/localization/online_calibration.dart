import 'dart:math' as math;

/// Learns this phone and this vehicle while satellites are visible, and applies what it
/// learned the moment they are not.
///
/// The idea this rests on: a GNSS-denied navigator has a free supervisor for most of its
/// life. Whenever GPS is healthy we know the true speed and the true course, so the errors
/// of the inertial pipeline are directly observable. Nothing has to be labelled by hand and
/// nothing has to be uploaded — the tunnel is simply where we spend what we banked.
///
/// Four things are learned, chosen because each is (a) roughly constant for one user in one
/// car, and (b) badly wrong in a model fitted across many:
///
///   yawBias    rad/s   a MEMS gyro at rest does not read zero, and the accelerometer
///                      cannot observe rotation about the vertical, so nothing else in the
///                      pipeline can see this. Unremoved, 0.03 rad/s is ~155 deg of heading
///                      over a 90 s blackout.
///   yawScale   -       gyros carry a 1-3% scale error that is fixed per device. Two
///                      percent of a 90 s blackout at speed is metres of cross-track.
///   speedGain,
///   speedOffset        the speed model reads road vibration, and the vibration-to-speed
///                      relationship depends on suspension, tyres and how rigid the mount
///                      is. That is precisely why held-out-driver R2 went negative: it is
///                      not learnable ACROSS vehicles, and it is easy to learn within one.
///   speedSigma m/s     how wrong the speed source actually is. The filter was previously
///                      told 0.2 while the measured error was 5.7 - 28x overconfident, and
///                      that miscalibration mattered more than which source was chosen.
///
/// THREE RULES, because naive "learn from usage" degrades quietly:
///
///   1. Never learn from our own output. Every observation here comes from GNSS or from a
///      stationarity detector. If the filter's estimate fed back into training it would
///      drift into confident nonsense with nothing to arrest it.
///   2. Gate the inputs hard. One session with the phone loose in a bag would otherwise
///      poison every future drive.
///   3. Never end up worse than the shipped model. Each parameter is blended toward its
///      neutral value by a confidence that starts at zero, so a cold start behaves exactly
///      like the uncalibrated system rather than like something unpredictable.
///
/// Everything is in the COMPASS convention (clockwise-positive, 0 = North) to match
/// EkfFusionEngine6State. Mixing conventions is what inverted heading in this codebase
/// once already.
class OnlineCalibration {
  // ── Neutral values: what we behave as before anything is learned ──────────
  static const double neutralYawScale = 1.0;
  static const double neutralSpeedGain = 1.0;
  static const double defaultSpeedSigma = 5.7; // measured on real IO-VNBD blackouts

  // ── Acceptance gates ──────────────────────────────────────────────────────
  static const double maxGnssAccuracyM = 15.0;
  static const double minSpeedForSpeedLearningMps = 2.0;
  /// GNSS course over ground is undefined at low speed; below this it is noise.
  static const double minSpeedForHeadingLearningMps = 5.0;
  /// A heading window must contain a real turn, or we are dividing noise by noise.
  static const double minTurnForScaleRad = 0.25; // ~14 deg
  static const double headingWindowSec = 8.0;

  // ── Learned state ─────────────────────────────────────────────────────────
  double _yawBias = 0.0;
  double _yawScale = neutralYawScale;
  double _speedGain = neutralSpeedGain;
  double _speedOffset = 0.0;
  double _speedSigma = defaultSpeedSigma;

  double stationarySeconds = 0.0;
  int _biasSamples = 0;
  int headingWindows = 0;
  int speedSamples = 0;

  // RLS state. P is the inverse-correlation matrix; lambda forgets slowly so the fit can
  // follow a real change (different cradle, tyre pressure) without chasing noise.
  static const double _lambda = 0.999;
  double _pScale = 100.0;
  List<List<double>> _pSpeed = [
    [100.0, 0.0],
    [0.0, 100.0],
  ];

  // Heading window accumulators
  double _winGyro = 0.0;
  double _winSec = 0.0;
  double? _winCourseStart;

  /// 0 while nothing is known, approaching 1 as evidence accumulates. Used to blend each
  /// parameter toward neutral so a cold start cannot be worse than no calibration at all.
  double get yawBiasConfidence => _sat(stationarySeconds / 20.0);
  double get yawScaleConfidence => _sat(headingWindows / 8.0);
  double get speedConfidence => _sat(speedSamples / 300.0);

  static double _sat(double x) => x <= 0 ? 0.0 : (x >= 1 ? 1.0 : x);

  /// Effective (confidence-blended) values. These are what the pipeline should use.
  double get yawBias => _yawBias * yawBiasConfidence;
  double get yawScale =>
      neutralYawScale + (_yawScale - neutralYawScale) * yawScaleConfidence;
  double get speedGain =>
      neutralSpeedGain + (_speedGain - neutralSpeedGain) * speedConfidence;
  double get speedOffset => _speedOffset * speedConfidence;

  /// What to tell the EKF about the speed source. Falls back to the measured global figure
  /// until this device has said otherwise.
  double get speedSigma =>
      defaultSpeedSigma + (_speedSigma - defaultSpeedSigma) * speedConfidence;

  bool get hasLearnedAnything =>
      yawBiasConfidence > 0.05 ||
      yawScaleConfidence > 0.05 ||
      speedConfidence > 0.05;

  // ── Applying what was learned ─────────────────────────────────────────────

  /// Correct a compass-convention yaw rate. This is the payload: during a blackout it is
  /// the only thing standing between the gyro's raw error and the heading.
  double correctYawRate(double compassYawRate) =>
      (compassYawRate - yawBias) * yawScale;

  /// Correct the speed model's output for this vehicle.
  double correctSpeed(double modelSpeed) =>
      math.max(0.0, speedGain * modelSpeed + speedOffset);

  // ── Learning ──────────────────────────────────────────────────────────────

  /// Call on every IMU sample while stationarity is asserted by the motion gate.
  ///
  /// A gyro at rest reveals its own bias directly, and this needs no GPS at all — which
  /// matters, because it keeps working at a red light inside a city canyon.
  void observeStationary(double compassYawRate, double dt) {
    if (!compassYawRate.isFinite || dt <= 0 || dt > 0.5) return;
    stationarySeconds += dt;
    // Adaptive gain: a true running mean while evidence is thin, decaying into an
    // exponential filter with a ~30 s memory once it is not.
    //
    // A plain exponential filter is wrong at the start - after two time constants it has
    // only reached 1 - e^-2 = 86% of the true bias, so a minute of standing still leaves
    // 14% of the error in place and a 90 s tunnel still accrues ~14 deg of heading. A plain
    // running mean fixes that but then cannot follow thermal drift later. Taking the larger
    // of the two gains gives fast initial convergence AND long-run tracking.
    _biasSamples++;
    final alpha = math.max(1.0 - math.exp(-dt / 30.0), 1.0 / _biasSamples);
    _yawBias += alpha * (compassYawRate - _yawBias);
  }

  /// Call on every IMU sample while GNSS is healthy and the vehicle is moving.
  /// [courseRad] is GNSS course over ground, compass convention.
  void observeMovingWithGnss({
    required double compassYawRate,
    required double dt,
    required double gnssSpeed,
    required double gnssAccuracy,
    required double? courseRad,
    required bool mountDisturbed,
  }) {
    if (dt <= 0 || dt > 0.5) return;
    if (gnssAccuracy <= 0 || gnssAccuracy > maxGnssAccuracyM) return;
    if (mountDisturbed) {
      // The mounting moved, so gyro and course no longer describe the same rigid body.
      _resetHeadingWindow();
      return;
    }
    if (gnssSpeed < minSpeedForHeadingLearningMps || courseRad == null) {
      _resetHeadingWindow();
      return;
    }

    _winCourseStart ??= courseRad;
    _winGyro += (compassYawRate - yawBias) * dt;
    _winSec += dt;

    if (_winSec < headingWindowSec) return;

    final measured = _wrap(courseRad - _winCourseStart!);
    final predicted = _winGyro;
    _resetHeadingWindow();

    // Only a window containing a real turn says anything about scale. On a straight road
    // both numbers are ~0 and the ratio is pure noise.
    if (predicted.abs() < minTurnForScaleRad) return;
    // A wildly inconsistent window is a GNSS course glitch or a missed disturbance, not
    // evidence about the gyro.
    if ((measured - predicted).abs() > 0.6 * predicted.abs() + 0.35) return;

    // RLS, one parameter: minimise sum (measured - s * predicted)^2.
    final x = predicted;
    final denom = _lambda + x * _pScale * x;
    if (denom.abs() < 1e-12) return;
    final k = _pScale * x / denom;
    _yawScale += k * (measured - _yawScale * x);
    _pScale = (_pScale - k * x * _pScale) / _lambda;
    _yawScale = _yawScale.clamp(0.85, 1.15); // beyond this it is not a scale error
    headingWindows++;
  }

  /// Call whenever the speed model produced an estimate and GNSS can score it.
  void observeSpeed({
    required double modelSpeed,
    required double gnssSpeed,
    required double gnssAccuracy,
    required bool mountDisturbed,
  }) {
    if (!modelSpeed.isFinite || !gnssSpeed.isFinite) return;
    if (gnssAccuracy <= 0 || gnssAccuracy > maxGnssAccuracyM) return;
    if (mountDisturbed) return;
    if (gnssSpeed < minSpeedForSpeedLearningMps) return;

    // RLS, two parameters: gnssSpeed ~ gain * modelSpeed + offset.
    final x = [modelSpeed, 1.0];
    final px = [
      _pSpeed[0][0] * x[0] + _pSpeed[0][1] * x[1],
      _pSpeed[1][0] * x[0] + _pSpeed[1][1] * x[1],
    ];
    final denom = _lambda + x[0] * px[0] + x[1] * px[1];
    if (denom.abs() < 1e-12) return;
    final k = [px[0] / denom, px[1] / denom];

    final predicted = _speedGain * x[0] + _speedOffset * x[1];
    final err = gnssSpeed - predicted;
    _speedGain += k[0] * err;
    _speedOffset += k[1] * err;

    final xp = [
      x[0] * _pSpeed[0][0] + x[1] * _pSpeed[1][0],
      x[0] * _pSpeed[0][1] + x[1] * _pSpeed[1][1],
    ];
    _pSpeed = [
      [
        (_pSpeed[0][0] - k[0] * xp[0]) / _lambda,
        (_pSpeed[0][1] - k[0] * xp[1]) / _lambda,
      ],
      [
        (_pSpeed[1][0] - k[1] * xp[0]) / _lambda,
        (_pSpeed[1][1] - k[1] * xp[1]) / _lambda,
      ],
    ];

    // Keep the fit physically sane. A gain outside this range means the model is not
    // tracking at all, and stretching it further would only hide that.
    _speedGain = _speedGain.clamp(0.2, 3.0);
    _speedOffset = _speedOffset.clamp(-5.0, 5.0);

    // Residual RMSE of the CORRECTED estimate - this is what the EKF should be told.
    final residual = gnssSpeed - (_speedGain * modelSpeed + _speedOffset);
    final alpha = 1.0 / math.min(speedSamples + 1, 200);
    _speedSigma = math.sqrt(
        (1 - alpha) * _speedSigma * _speedSigma + alpha * residual * residual);
    speedSamples++;
  }

  void _resetHeadingWindow() {
    _winGyro = 0.0;
    _winSec = 0.0;
    _winCourseStart = null;
  }

  static double _wrap(double a) {
    var v = (a + math.pi) % (2 * math.pi);
    if (v < 0) v += 2 * math.pi;
    return v - math.pi;
  }

  /// A one-line human summary for the UI. Says what was learned and on how much evidence,
  /// because a number with no provenance is not worth showing.
  String get summary {
    if (!hasLearnedAnything) return 'Not yet calibrated — drive with GPS to teach it.';
    final bits = <String>[];
    if (yawBiasConfidence > 0.05) {
      bits.add('gyro bias ${(yawBias * 180 / math.pi).toStringAsFixed(3)}°/s');
    }
    if (yawScaleConfidence > 0.05) {
      bits.add('gyro scale ${yawScale.toStringAsFixed(3)}×');
    }
    if (speedConfidence > 0.05) {
      bits.add('speed ×${speedGain.toStringAsFixed(2)}'
          '${speedOffset >= 0 ? '+' : ''}${speedOffset.toStringAsFixed(2)}');
    }
    return bits.join(' · ');
  }

  Map<String, dynamic> toJson() => {
        'yawBias': _yawBias,
        'yawScale': _yawScale,
        'speedGain': _speedGain,
        'speedOffset': _speedOffset,
        'speedSigma': _speedSigma,
        'stationarySeconds': stationarySeconds,
        'headingWindows': headingWindows,
        'speedSamples': speedSamples,
      };

  void loadJson(Map<String, dynamic> j) {
    double d(String k, double fallback) {
      final v = j[k];
      return v is num && v.isFinite ? v.toDouble() : fallback;
    }

    _yawBias = d('yawBias', 0.0);
    _yawScale = d('yawScale', neutralYawScale).clamp(0.85, 1.15);
    _speedGain = d('speedGain', neutralSpeedGain).clamp(0.2, 3.0);
    _speedOffset = d('speedOffset', 0.0).clamp(-5.0, 5.0);
    _speedSigma = d('speedSigma', defaultSpeedSigma);
    stationarySeconds = d('stationarySeconds', 0.0);
    headingWindows = (j['headingWindows'] as num?)?.toInt() ?? 0;
    speedSamples = (j['speedSamples'] as num?)?.toInt() ?? 0;
  }

  void reset() {
    _yawBias = 0.0;
    _yawScale = neutralYawScale;
    _speedGain = neutralSpeedGain;
    _speedOffset = 0.0;
    _speedSigma = defaultSpeedSigma;
    stationarySeconds = 0.0;
    _biasSamples = 0;
    headingWindows = 0;
    speedSamples = 0;
    _pScale = 100.0;
    _pSpeed = [
      [100.0, 0.0],
      [0.0, 100.0],
    ];
    _resetHeadingWindow();
  }
}

/// Where learned parameters live between sessions.
///
/// In-memory by default so the app and its tests need no new dependency. Cross-session
/// persistence is a drop-in: implement this against shared_preferences or a file in the
/// app documents directory and hand it to the provider. Within-session learning - drive,
/// then enter the tunnel - works either way, and that is the case that matters most.
abstract class CalibrationStore {
  Map<String, dynamic>? load();
  void save(Map<String, dynamic> data);
}

class InMemoryCalibrationStore implements CalibrationStore {
  Map<String, dynamic>? _data;
  @override
  Map<String, dynamic>? load() => _data;
  @override
  void save(Map<String, dynamic> data) => _data = Map.of(data);
}
