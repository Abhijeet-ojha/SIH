import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;

import 'frame_alignment.dart';

/// The trained speed model, running on the phone.
///
/// This is the same gradient-boosted ensemble exported by
/// scripts/export_ondevice_model.py from REAL IO-VNBD drives with CAN-bus speed labels -
/// 98,900 training windows, 60 shallow trees, 43 KB of JSON. Not a heuristic, not a
/// re-implementation: the identical tree structure, evaluated by identical traversal.
///
/// Tree traversal deliberately mirrors eval_trees() in the exporter so parity is exact
/// rather than approximate: same feature order, same `<=` comparison, same accumulation.
/// outputs/models/golden_vectors.json holds 486 windows of reference input/output for
/// checking that claim on device.
class SpeedModel {
  static const String assetPath = 'assets/models/ondevice_model.json';

  /// Must match ONDEVICE_FEATURES in scripts/export_ondevice_model.py, in order.
  static const List<String> featureNames = [
    'a_fwd_mean',
    'a_fwd_std',
    'a_fwd_rms',
    'a_lat_std',
    'a_lat_rms',
    'a_vert_std',
    'a_vert_rms',
    'a_horiz_mag_mean',
    'a_horiz_mag_std',
    'a_horiz_mag_rms',
    'yaw_rate_absmean',
    'yaw_rate_std',
    'gyro_mag_mean',
    'tilt_rate_rms',
    'grav_stab_mean',
    'road_vibration',
  ];

  double _init = 0.0;
  double _learningRate = 0.1;
  List<_Tree> _trees = const [];
  String provenance = 'not loaded';
  bool get isLoaded => _trees.isNotEmpty;

  /// Measured RMSE of this model during real GNSS blackouts. The EKF was previously told
  /// to expect 0.2 m/s, which is 28x more confident than the model deserves; a filter that
  /// trusts a bad source follows it off the road. Reported honestly so the filter can
  /// weight it correctly.
  static const double measuredBlackoutRmse = 5.7;

  Future<void> load() async {
    final raw = await rootBundle.loadString(assetPath);
    final m = json.decode(raw) as Map<String, dynamic>;

    final feats = (m['features'] as List).cast<String>();
    if (feats.length != featureNames.length) {
      throw StateError('model has ${feats.length} features, app expects '
          '${featureNames.length}');
    }
    for (var i = 0; i < feats.length; i++) {
      if (feats[i] != featureNames[i]) {
        throw StateError(
            'feature order mismatch at $i: asset says "${feats[i]}", '
            'app computes "${featureNames[i]}". Re-export or update the app - a silent '
            'reorder would produce plausible nonsense.');
      }
    }

    _init = (m['init'] as num).toDouble();
    _learningRate = (m['learning_rate'] as num).toDouble();
    provenance = (m['provenance'] ?? 'unknown').toString();
    _trees = (m['trees'] as List)
        .map((t) => _Tree.fromJson(t as Map<String, dynamic>))
        .toList(growable: false);
  }

  double predict(List<double> x) {
    var out = _init;
    for (final t in _trees) {
      var node = 0;
      while (t.feature[node] != -2) {
        node = x[t.feature[node]] <= t.threshold[node]
            ? t.left[node]
            : t.right[node];
      }
      out += _learningRate * t.value[node];
    }
    return out;
  }
}

class _Tree {
  final List<int> feature, left, right;
  final List<double> threshold, value;
  _Tree(this.feature, this.threshold, this.left, this.right, this.value);

  factory _Tree.fromJson(Map<String, dynamic> t) => _Tree(
        (t['feature'] as List).map((e) => (e as num).toInt()).toList(),
        (t['threshold'] as List).map((e) => (e as num).toDouble()).toList(),
        (t['left'] as List).map((e) => (e as num).toInt()).toList(),
        (t['right'] as List).map((e) => (e as num).toInt()).toList(),
        (t['value'] as List).map((e) => (e as num).toDouble()).toList(),
      );
}

/// Rolling 1.5 s window of frame-invariant channels, producing the 16 features the model
/// expects. Same window length and same statistics as the Python extractor.
class SpeedFeatureWindow {
  static const double windowSec = 1.5;

  final List<double> _aFwd = [], _aLat = [], _aVert = [], _aHoriz = [];
  final List<double> _yaw = [], _gyroMag = [], _tilt = [], _gravStab = [];
  int _capacity = 75; // 1.5 s at 50 Hz until the real rate is known
  final List<double> _dtHistory = [];

  bool get isReady => _aFwd.length >= _capacity;

  /// Window length in seconds, for display. The model was trained on [windowSec]; if this
  /// reads materially different, the features are not the features it learned.
  double get windowSeconds => _capacity * _medianDt;
  double _medianDt = 0.02;

  void push(FrameAlignment f, double dt) {
    // Size from the MEDIAN observed interval, not the first one. The provider cannot know
    // dt on the very first sample and passes a hardcoded 0.02, so sizing on that gave a
    // 75-sample window on every handset: correct at 50 Hz, but only 0.76 s of data on a
    // 100 Hz device - half the window the model was trained on, silently.
    if (dt > 0 && dt < 0.5 && _dtHistory.length < 64) {
      _dtHistory.add(dt);
      if (_dtHistory.length >= 12) {
        final sorted = List<double>.from(_dtHistory)..sort();
        _medianDt = sorted[sorted.length ~/ 2];
        _capacity = math.max(8, (windowSec / _medianDt).round());
      }
    }
    _add(_aFwd, f.aFwd);
    _add(_aLat, f.aLat);
    _add(_aVert, f.aVert);
    _add(_aHoriz, f.aHorizMag);
    _add(_yaw, f.yawRate);
    _add(_gyroMag, f.gyroMag);
    _add(_tilt, f.tiltRate);
    _add(_gravStab, f.gravStability);
  }

  void _add(List<double> buf, double v) {
    buf.add(v);
    while (buf.length > _capacity) {
      buf.removeAt(0);
    }
  }

  List<double> features() => [
        _mean(_aFwd), _std(_aFwd), _rms(_aFwd),
        _std(_aLat), _rms(_aLat),
        _std(_aVert), _rms(_aVert),
        _mean(_aHoriz), _std(_aHoriz), _rms(_aHoriz),
        _absMean(_yaw), _std(_yaw),
        _mean(_gyroMag), _rms(_tilt), _mean(_gravStab),
        _std(_aVert) * _rms(_aHoriz), // road_vibration
      ];

  void reset() {
    for (final b in [
      _aFwd,
      _aLat,
      _aVert,
      _aHoriz,
      _yaw,
      _gyroMag,
      _tilt,
      _gravStab
    ]) {
      b.clear();
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

  static double _absMean(List<double> v) {
    if (v.isEmpty) return 0.0;
    var s = 0.0;
    for (final x in v) {
      s += x.abs();
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

  static double _std(List<double> v) {
    if (v.isEmpty) return 0.0;
    final m = _mean(v);
    var s = 0.0;
    for (final x in v) {
      final d = x - m;
      s += d * d;
    }
    return math.sqrt(s / v.length);
  }
}
