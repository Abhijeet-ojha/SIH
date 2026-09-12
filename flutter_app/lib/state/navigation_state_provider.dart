import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import '../localization/pedestrian_tracker.dart';
import 'package:geolocator/geolocator.dart';
import 'package:sensors_plus/sensors_plus.dart';

import '../analytics/session_recorder.dart';
import '../localization/ekf_fusion_engine.dart';
import '../localization/frame_alignment.dart';
import '../localization/motion_gate.dart';
import '../localization/online_calibration.dart';
import '../localization/speed_model.dart';
import '../models/sensor_quality.dart';
import '../safety/position_report.dart';
import '../safety/safety_service.dart';
import '../models/tile_basemap.dart';
import '../models/session_data.dart';
import '../sensors/sensor_calibrator.dart';
import '../sensors/sensor_quality_engine.dart';

class TrackPoint {
  final double east, north;
  final bool denied;
  const TrackPoint(this.east, this.north, this.denied);
}

/// Owns the whole pipeline and is the single source of truth for the UI.
///
///   sensors -> calibration -> frame alignment -> motion gate -> speed model -> EKF
///
/// The gate sits BEFORE the model deliberately: when the phone is being handled, the speed
/// estimate is not merely inaccurate, it is meaningless, and the correct action is to stop
/// integrating rather than to integrate something smaller.
class NavigationStateProvider extends ChangeNotifier {
  final FrameAlignment frame = FrameAlignment();
  final MotionGate gate = MotionGate();
  final SpeedModel model = SpeedModel();
  final SpeedFeatureWindow window = SpeedFeatureWindow();
  final SensorCalibrator calibrator = SensorCalibrator();
  final SensorQualityEngine quality = SensorQualityEngine();
  final SessionRecorder recorder = SessionRecorder();

  /// Learns this phone and this car while GPS is visible; spends it in the tunnel.
  final OnlineCalibration calibration = OnlineCalibration();
  final CalibrationStore calibrationStore = InMemoryCalibrationStore();

  /// Emergency sharing. Lives here so the panic button works from any screen.
  final SafetyService safety = SafetyService();

  // ── GNSS anchor ───────────────────────────────────────────────────────────
  // The pipeline works in local metres from wherever the session began, which cannot be
  // shared with anyone. These pin that frame to the Earth using the last real fix, so a
  // position can be reported as a latitude and longitude with an honest error bound.
  double? _anchorLat, _anchorLon;
  double _anchorAccuracyM = 0;
  double _anchorEast = 0, _anchorNorth = 0;
  DateTime? _anchorAt;
  double _distanceAtAnchor = 0;

  double? get anchorLat => _anchorLat;
  double? get anchorLon => _anchorLon;
  double get anchorEast => _anchorEast;
  double get anchorNorth => _anchorNorth;

  /// Offline basemap for the demo area, if one was bundled. Null is normal and harmless -
  /// the map draws a metre grid instead.
  TileBasemap? basemap;

  // ── Destination ───────────────────────────────────────────────────────────
  // Held in local metres rather than lat/lon so setting one works before any GNSS fix -
  // which is the indoor case, and the one where guidance is worth the most.
  double? _destE, _destN;
  String destinationLabel = '';

  bool get hasDestination => _destE != null;
  (double, double)? get destinationLocal =>
      _destE == null ? null : (_destE!, _destN!);

  void setDestinationLocal(double east, double north,
      {String label = 'Destination'}) {
    _destE = east;
    _destN = north;
    destinationLabel = label;
    notifyListeners();
  }

  /// Set a destination from real coordinates. Returns false when the local frame has not
  /// been pinned to the Earth yet, because there is then no way to place it.
  bool setDestinationLatLon(double lat, double lon,
      {String label = 'Destination'}) {
    if (_anchorLat == null || _anchorLon == null) return false;
    final (e, n) =
        PositionReport.enuFromLatLon(_anchorLat!, _anchorLon!, lat, lon);
    setDestinationLocal(_anchorEast + e, _anchorNorth + n, label: label);
    return true;
  }

  void clearDestination() {
    _destE = _destN = null;
    destinationLabel = '';
    notifyListeners();
  }

  double? get destinationDistanceM {
    if (_destE == null) return null;
    final e = (indoorMode ? pedestrian.east : ekf.posEast) - _destE!;
    final n = (indoorMode ? pedestrian.north : ekf.posNorth) - _destN!;
    return math.sqrt(e * e + n * n);
  }

  /// Where the destination lies relative to the way the user is currently facing:
  /// 0 straight ahead, +90 to the right, -90 to the left. This is the only guidance the
  /// device can honestly give offline - there is no road graph on it, so a turn-by-turn
  /// route would be invention.
  double? get destinationRelativeBearingDeg {
    if (_destE == null) return null;
    final dE = _destE! - (indoorMode ? pedestrian.east : ekf.posEast);
    final dN = _destN! - (indoorMode ? pedestrian.north : ekf.posNorth);
    if (dE.abs() < 1e-9 && dN.abs() < 1e-9) return 0;
    final bearing = math.atan2(dE, dN) * 180.0 / math.pi;
    final facing =
        (indoorMode ? pedestrian.heading : ekf.heading) * 180.0 / math.pi;
    var rel = (bearing - facing) % 360.0;
    if (rel > 180) rel -= 360;
    if (rel < -180) rel += 360;
    return rel;
  }

  EkfFusionEngine6State ekf = EkfFusionEngine6State();

  StreamSubscription? _accelSub, _gyroSub, _magSub, _gpsSub;

  final PedestrianTracker pedestrian = PedestrianTracker();
  bool indoorMode = true;
  StreamSubscription? _headingSub;
  Timer? _indoorTimer;
  bool _rotationLive = false;
  int _rotationMs = 0;
  bool _starting = false;
  final Stopwatch _clock = Stopwatch()..start();
  int get _now => _clock.elapsedMilliseconds;
  double get positionSigmaM =>
      indoorMode ? 0.3 + pedestrian.distanceM * 0.2 : ekf.positionSigmaM;
  String get headingSource =>
      _rotationLive ? 'Android rotation sensor' : 'Gyroscope fallback';
  void setIndoorMode(bool value) {
    if (isRunning || _starting) return;
    indoorMode = value;
    blackoutSimulated = false;
    track.clear();
    pedestrian.reset();
    ekf = EkfFusionEngine6State();
    notifyListeners();
  }

  void setStepLength(double value) {
    if (isRunning) return;
    pedestrian.stepLengthM = value;
    notifyListeners();
  }

  void setStepThreshold(double value) {
    if (isRunning) return;
    pedestrian.threshold = value;
    notifyListeners();
  }

  bool isRunning = false;
  bool blackoutSimulated = false;
  bool modelLoaded = false;
  String modelProvenance = '';
  String? lastError;

  // Latest raw sample
  double _ax = 0, _ay = 0, _az = 9.80665;
  double _gx = 0, _gy = 0, _gz = 0;
  int _lastImuMs = 0;

  // GNSS
  double? gnssLat, gnssLon;
  double gnssSpeed = 0.0;
  double gnssAccuracy = 0.0;
  double? gnssHeading;
  bool hasFix = false;
  double _originLat = 0, _originLon = 0;
  bool _originSet = false;

  double mlSpeed = 0.0;
  double mlUncertainty = SpeedModel.measuredBlackoutRmse;
  double _prevGnssSpeed = 0.0;
  double _blackoutSpeed = 0.0;

  final List<TrackPoint> track = [];
  static const int _trackCapacity = 5000;

  SensorQualityReport? qualityReport;
  double _blackoutStartEast = 0, _blackoutStartNorth = 0;
  double blackoutDistanceM = 0.0;

  MotionState get motionState => gate.state;
  String get motionReason => !indoorMode
      ? gate.reason
      : !isRunning
          ? 'Tap Play. Hold still for two seconds, then walk with phone screen-up.'
          : !pedestrian.ready
              ? 'Hold still: preparing step detector...'
              : '${pedestrian.steps} steps / ${pedestrian.distanceM.toStringAsFixed(1)} m. GPS off. Top of phone points forward.';
  bool get gnssAvailable => !indoorMode && hasFix && !blackoutSimulated;

  /// Metres travelled since the session began, whichever mode is running.
  double get _travelledM =>
      indoorMode ? pedestrian.distanceM : _vehicleOdometerM;
  double _vehicleOdometerM = 0;

  /// Current position as something that can be given to another person.
  ///
  /// Returns an unanchored report (isShareable == false) until a GNSS fix has been seen.
  /// It deliberately does NOT invent a coordinate in that case: sending the session origin
  /// or (0,0) would send help to the wrong place, which is worse than sending nothing.
  PositionReport get report {
    final now = DateTime.now();
    final east = indoorMode ? pedestrian.east : ekf.posEast;
    final north = indoorMode ? pedestrian.north : ekf.posNorth;
    final sinceFix = (_travelledM - _distanceAtAnchor).clamp(0.0, double.infinity);

    if (_anchorLat == null || _anchorLon == null) {
      return PositionReport(
        lat: double.nan,
        lon: double.nan,
        uncertaintyM: double.infinity,
        fixAgeSeconds: double.infinity,
        distanceSinceFixM: sinceFix,
        gpsAnchored: false,
        mode: indoorMode ? 'walking' : 'vehicle',
        at: now,
      );
    }

    return PositionReport.fromAnchor(
      anchorLat: _anchorLat!,
      anchorLon: _anchorLon!,
      anchorAccuracyM: _anchorAccuracyM,
      eastM: east - _anchorEast,
      northM: north - _anchorNorth,
      fixAgeSeconds: _anchorAt == null
          ? double.infinity
          : now.difference(_anchorAt!).inMilliseconds / 1000.0,
      distanceSinceFixM: sinceFix,
      drUncertaintyM: indoorMode
          ? PositionReport.pedestrianDrift(sinceFix)
          : ekf.positionSigmaM,
      gpsAnchored: true,
      mode: indoorMode ? 'walking' : 'vehicle',
      at: now,
    );
  }

  // Contacts are edited from the UI, so the mutations go through here rather than the
  // service directly - the sheet has to rebuild when the list changes.
  void addSafetyContact(String number) {
    safety.addContact(number);
    notifyListeners();
  }

  void removeSafetyContact(String number) {
    safety.removeContact(number);
    notifyListeners();
  }

  void setSafetyOwnerName(String? name) {
    safety.ownerName = (name == null || name.trim().isEmpty) ? null : name.trim();
    notifyListeners();
  }

  /// One tap: capture the current position and get it to the emergency contacts.
  Future<ShareResult> panic({bool emergency = true}) async {
    final r = report;
    final res = await safety.share(r, emergency: emergency);
    notifyListeners();
    return res;
  }

  double get speedKmh => ekf.speed * 3.6;
  double get headingDeg {
    var d = ekf.heading * 180.0 / math.pi;
    if (d < 0) d += 360.0;
    return d;
  }

  Future<void> initialise() async {
    try {
      await model.load();
      modelLoaded = model.isLoaded;
      modelProvenance = model.provenance;
    } catch (e) {
      lastError = 'Speed model failed to load: $e';
      modelLoaded = false;
    }
    // Optional: absent tiles are the normal case and must not surface as an error.
    basemap = await TileBasemap.load();
    notifyListeners();
  }

  Future<void> start() async {
    if (isRunning || _starting) return;
    _starting = true;
    lastError = null;

    if (!indoorMode && !await _ensureLocationPermission()) {
      _starting = false;
      lastError = 'Location permission denied — GNSS fusion unavailable.';
      notifyListeners();
      return;
    }

    ekf = EkfFusionEngine6State();
    frame.reset();
    gate.reset();
    window.reset();
    quality.reset();
    track.clear();
    _originSet = false;
    _anchorLat = _anchorLon = null;
    _anchorAt = null;
    _anchorAccuracyM = 0;
    _anchorEast = _anchorNorth = 0;
    _distanceAtAnchor = 0;
    _vehicleOdometerM = 0;
    // A destination is stored in local metres, so it stops meaning anything once the
    // local frame is re-originated.
    _destE = _destN = null;
    destinationLabel = '';
    hasFix = false;
    blackoutSimulated = false;
    gnssSpeed = 0.0;
    _prevGnssSpeed = 0.0;
    _blackoutSpeed = 0.0;
    mlSpeed = 0.0;
    qualityReport = null;
    _lastImuMs = 0;
    isRunning = true;
    _starting = false;
    pedestrian.reset();
    _rotationLive = false;
    _rotationMs = 0;

    final saved = calibrationStore.load();
    if (saved != null) calibration.loadJson(saved);

    recorder.startRecording(
        sessionName:
            'DRIVE_${DateTime.now().toIso8601String().substring(0, 19)}');

    void sensorError(Object error) {
      lastError = 'Sensor unavailable: $error';
      notifyListeners();
    }

    _accelSub = accelerometerEventStream(
            samplingPeriod: const Duration(milliseconds: 20))
        .listen(_onAccel, onError: sensorError);
    _gyroSub =
        gyroscopeEventStream(samplingPeriod: const Duration(milliseconds: 20))
            .listen(_onGyro, onError: sensorError);
    if (indoorMode) {
      track.add(const TrackPoint(0, 0, true));
      _headingSub = const EventChannel('navpulse/relative_heading')
          .receiveBroadcastStream()
          .listen((value) {
        if (!isRunning) return;
        _rotationLive = true;
        _rotationMs = _now;
        pedestrian.observeHeading((value as num).toDouble());
      }, onError: (Object error) {
        _rotationLive = false;
      });
      _indoorTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
        if (!isRunning) return;
        ekf.x[2] = pedestrian.speedAt(_now);
        ekf.x[4] = pedestrian.heading;
        qualityReport = quality.generateReport(_now);
        if (recorder.isRecording) _record(_now);
        notifyListeners();
      });
    } else {
      _magSub = magnetometerEventStream(
              samplingPeriod: const Duration(milliseconds: 100))
          .listen(_onMag, onError: sensorError);
      _gpsSub = Geolocator.getPositionStream(
        locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.bestForNavigation, distanceFilter: 0),
      ).listen(_onGnss, onError: (Object e) {
        lastError = 'GNSS: $e';
        notifyListeners();
      });
    }

    notifyListeners();
  }

  Future<void> stop() async {
    if (!isRunning) return;
    isRunning = false;
    _starting = false;
    _indoorTimer?.cancel();
    await _headingSub?.cancel();
    ekf.x[2] = 0;
    await _accelSub?.cancel();
    await _gyroSub?.cancel();
    await _magSub?.cancel();
    await _gpsSub?.cancel();
    calibrationStore.save(calibration.toJson());
    final q = qualityReport;
    recorder.stopRecording(
        accelHz: q?.accel.actualHz ?? 0.0, gyroHz: q?.gyro.actualHz ?? 0.0);
    notifyListeners();
  }

  Future<bool> _ensureLocationPermission() async {
    var p = await Geolocator.checkPermission();
    if (p == LocationPermission.denied)
      p = await Geolocator.requestPermission();
    return p == LocationPermission.always || p == LocationPermission.whileInUse;
  }

  void _onAccel(AccelerometerEvent e) {
    final now = _now;
    _ax = e.x;
    _ay = e.y;
    _az = e.z;
    quality.recordAccelEvent(e.x, e.y, e.z, now);
    if (isRunning &&
        indoorMode &&
        calibrator.status != CalibrationStatus.calibrating) {
      if (pedestrian.addAcceleration(e.x, e.y, e.z, now)) {
        ekf.x[0] = pedestrian.east;
        ekf.x[1] = pedestrian.north;
        ekf.x[2] = pedestrian.speedAt(now);
        ekf.x[4] = pedestrian.heading;
        track.add(TrackPoint(pedestrian.east, pedestrian.north, true));
        if (track.length > _trackCapacity) track.removeAt(0);
        notifyListeners();
      }
    }
    if (calibrator.status == CalibrationStatus.calibrating) {
      calibrator.feedRawSample(
          ax: _ax, ay: _ay, az: _az, gx: _gx, gy: _gy, gz: _gz);
    }
  }

  void _onMag(MagnetometerEvent e) {
    quality.recordMagEvent(e.x, e.y, e.z, _now);
  }

  /// The gyro drives the pipeline: it is the highest-rate stream and the one whose
  /// integration the position depends on, so one step per gyro sample keeps dt honest.
  void _onGyro(GyroscopeEvent e) {
    final nowMs = _now;
    _gx = e.x;
    _gy = e.y;
    _gz = e.z;
    quality.recordGyroEvent(e.x, e.y, e.z, nowMs);
    if (!isRunning) return;

    final dt = _lastImuMs == 0 ? 0.02 : (nowMs - _lastImuMs) / 1000.0;
    _lastImuMs = nowMs;
    if (dt <= 0 || dt > 0.5) return;

    final (cax, cay, caz, cgx, cgy, cgz) = calibrator.applyCorrection(
      rawAx: _ax,
      rawAy: _ay,
      rawAz: _az,
      rawGx: _gx,
      rawGy: _gy,
      rawGz: _gz,
    );

    frame.update(cax, cay, caz, cgx, cgy, cgz, dt);
    if (indoorMode) {
      if (_now - _rotationMs > 500) _rotationLive = false;
      if (!_rotationLive) pedestrian.integrateYaw(frame.compassYawRate, dt);
      return;
    }
    window.push(frame, dt);

    gate.update(frame,
        dt: dt, gnssAvailable: gnssAvailable, speedHint: ekf.speed);

    // ── LEARN, only while satellites can mark our work ────────────────────
    // Bias needs no GPS - a gyro at rest reveals it directly - so this keeps working at a
    // red light in a city canyon where no fix is available.
    if (gate.state == MotionState.stationary) {
      calibration.observeStationary(frame.compassYawRate, dt);
    } else if (gnssAvailable && hasFix) {
      calibration.observeMovingWithGnss(
        compassYawRate: frame.compassYawRate,
        dt: dt,
        gnssSpeed: gnssSpeed,
        gnssAccuracy: gnssAccuracy,
        courseRad: gnssHeading,
        mountDisturbed: frame.mountDisturbed,
      );
    }

    // Model runs only when the gate agrees we are in a moving vehicle.
    if (modelLoaded &&
        window.isReady &&
        gate.state == MotionState.inVehicleMoving) {
      mlSpeed = math.max(0.0, model.predict(window.features()));
    } else if (gate.state != MotionState.inVehicleMoving) {
      mlSpeed = 0.0;
    }

    // ── LEARN the speed correction, then APPLY it ────────────────────────
    // Score the RAW model output against GPS, before correcting it - fitting a correction
    // against its own corrected output would chase its own tail.
    final rawModelSpeed = mlSpeed;
    if (gnssAvailable && hasFix && gate.state == MotionState.inVehicleMoving) {
      calibration.observeSpeed(
        modelSpeed: rawModelSpeed,
        gnssSpeed: gnssSpeed,
        gnssAccuracy: gnssAccuracy,
        mountDisturbed: frame.mountDisturbed,
      );
    }
    // Per-vehicle gain and offset. Blended by confidence inside OnlineCalibration, so
    // before anything is learned this is the identity and behaviour is unchanged.
    mlSpeed = calibration.correctSpeed(rawModelSpeed);

    // Sigma is the model's MEASURED error, not a hopeful constant. Telling the filter a
    // source is better than it is makes the filter follow it off the road. Once this
    // device has enough evidence, its own measured residual replaces the global figure.
    mlUncertainty = switch (gate.state) {
      MotionState.stationary => 0.04,
      MotionState.phoneHandled => 20.0,
      MotionState.inVehicleMoving => calibration.speedSigma,
    };

    final stationary = gate.state != MotionState.inVehicleMoving;

    // During a blackout the ablation says holding the last known GNSS speed beats the
    // model. Outside one, GNSS corrects the filter directly so it hardly matters.
    final vAi = blackoutSimulated ? _blackoutSpeed : mlSpeed;

    ekf.predict(
      dt: dt,
      vAi: vAi,
      vAiStd: mlUncertainty,
      // The payload of everything learned above: bias removed, scale applied. During a
      // blackout this is the only thing between the gyro's own error and the heading.
      gyroZ: calibration.correctYawRate(frame.compassYawRate),
      isStationary: stationary,
    );
    ekf.updateNhc();
    if (stationary) ekf.updateZupt();

    _vehicleOdometerM += ekf.speed * dt;
    _appendTrack();
    if (recorder.isRecording) _record(nowMs);

    qualityReport = quality.generateReport(nowMs);
    notifyListeners();
  }

  void _onGnss(Position p) {
    if (indoorMode || !isRunning) return;
    hasFix = true;
    gnssLat = p.latitude;
    gnssLon = p.longitude;
    gnssAccuracy = p.accuracy;
    _prevGnssSpeed = gnssSpeed;
    gnssSpeed = p.speed.isFinite ? math.max(0.0, p.speed) : 0.0;
    gnssHeading = p.heading.isFinite && gnssSpeed > 2.0
        ? p.heading * math.pi / 180.0
        : null;

    // Re-anchor on every fix good enough to trust. Each one resets the error budget: the
    // uncertainty circle collapses back to the GNSS accuracy and starts growing again only
    // as dead reckoning takes over.
    if (p.accuracy > 0 && p.accuracy <= 30) {
      _anchorLat = p.latitude;
      _anchorLon = p.longitude;
      _anchorAccuracyM = p.accuracy;
      _anchorEast = indoorMode ? pedestrian.east : ekf.posEast;
      _anchorNorth = indoorMode ? pedestrian.north : ekf.posNorth;
      _anchorAt = DateTime.now();
      _distanceAtAnchor = _travelledM;
      // Anything held while there was no position to report can go now.
      if (safety.pendingCount > 0) unawaited(safety.flush());
    }

    if (!_originSet) {
      _originLat = p.latitude;
      _originLon = p.longitude;
      _originSet = true;
    }

    // Learn the forward axis from real acceleration events while GNSS can confirm them.
    if (!blackoutSimulated) {
      frame.observeForwardAxis(gnssSpeed - _prevGnssSpeed, gnssSpeed);
    }

    if (!blackoutSimulated && isRunning) {
      final (e, n) = _toEnu(p.latitude, p.longitude);
      ekf.updateGps(
          east: e, north: n, gpsSpeed: gnssSpeed, gpsHeading: gnssHeading);
    }
    notifyListeners();
  }

  (double, double) _toEnu(double lat, double lon) {
    const r = 6371000.0;
    final x = r *
        (lon - _originLon) *
        math.pi /
        180.0 *
        math.cos(_originLat * math.pi / 180.0);
    final y = r * (lat - _originLat) * math.pi / 180.0;
    return (x, y);
  }

  void toggleBlackout() {
    blackoutSimulated = !blackoutSimulated;
    ekf.setGnssBlackout(blackoutSimulated);
    if (blackoutSimulated) {
      _blackoutSpeed = gnssSpeed;
      _blackoutStartEast = ekf.posEast;
      _blackoutStartNorth = ekf.posNorth;
      blackoutDistanceM = 0.0;
      recorder.logEvent('GNSS BLACKOUT SIMULATED');
    } else {
      blackoutDistanceM = math.sqrt(
          math.pow(ekf.posEast - _blackoutStartEast, 2) +
              math.pow(ekf.posNorth - _blackoutStartNorth, 2));
      recorder.logEvent(
          'GNSS RESTORED after ${ekf.blackoutDurationS.toStringAsFixed(0)} s');
    }
    notifyListeners();
  }

  void startCalibration() {
    calibrator.startCalibration(durationSec: 4.0);
    notifyListeners();
  }

  void _appendTrack() {
    track.add(TrackPoint(ekf.posEast, ekf.posNorth, blackoutSimulated));
    while (track.length > _trackCapacity) {
      track.removeAt(0);
    }
  }

  void _record(int nowMs) {
    recorder.recordSample(SessionSample(
      timestampS: nowMs / 1000.0,
      gnssLat: gnssLat ?? 0.0,
      gnssLon: gnssLon ?? 0.0,
      gnssSpeed: gnssSpeed,
      gnssAccuracy: gnssAccuracy,
      gnssMode: indoorMode
          ? 'INDOOR_PDR'
          : blackoutSimulated
              ? 'DENIED'
              : (hasFix ? 'NORMAL' : 'NOFIX'),
      ax: _ax,
      ay: _ay,
      az: _az,
      gx: _gx,
      gy: _gy,
      gz: _gz,
      mlSpeed: mlSpeed,
      mlUncertainty: mlUncertainty,
      ekfEast: ekf.posEast,
      ekfNorth: ekf.posNorth,
      ekfSpeed: ekf.speed,
      ekfHeading: headingDeg,
      ekfGyroBias: ekf.gyroBias,
      qualityScore: qualityReport?.overallScore ?? 1.0,
    ));
  }

  @override
  void dispose() {
    _indoorTimer?.cancel();
    _headingSub?.cancel();
    _accelSub?.cancel();
    _gyroSub?.cancel();
    _magSub?.cancel();
    _gpsSub?.cancel();
    super.dispose();
  }
}
