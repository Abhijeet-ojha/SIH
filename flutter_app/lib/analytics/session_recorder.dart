import 'dart:math' as math;

import '../models/session_data.dart';

/// Records a drive and produces an honest post-drive summary.
///
/// Everything is held in memory and written on demand - no network, no background upload.
/// A system whose selling point is working without infrastructure should not quietly
/// require infrastructure to record its own results.
class SessionRecorder {
  bool isRecording = false;
  String _sessionName = '';
  DateTime? _startedAt;
  double _startTimestampS = 0.0;

  final List<SessionSample> _samples = [];
  final List<SessionEvent> eventLog = [];
  final List<SessionSummary> completedSessions = [];

  List<SessionSample> get samples => List.unmodifiable(_samples);
  int get sampleCount => _samples.length;

  void startRecording({required String sessionName}) {
    isRecording = true;
    _sessionName = sessionName;
    _startedAt = DateTime.now();
    _startTimestampS = 0.0;
    _samples.clear();
    eventLog.clear();
    eventLog.add(SessionEvent(0.0, 'RECORDING STARTED — $sessionName'));
  }

  void recordSample(SessionSample s) {
    if (!isRecording) return;
    if (_samples.isEmpty) _startTimestampS = s.timestampS;

    // Log GNSS mode transitions rather than every sample - the transitions are what a
    // reviewer wants to see, and one line per sample would bury them.
    if (_samples.isNotEmpty && _samples.last.gnssMode != s.gnssMode) {
      eventLog.add(SessionEvent(s.timestampS - _startTimestampS,
          'GNSS ${_samples.last.gnssMode} → ${s.gnssMode}'));
    }
    _samples.add(s);
  }

  void logEvent(String label) {
    if (!isRecording) return;
    final t =
        _samples.isEmpty ? 0.0 : _samples.last.timestampS - _startTimestampS;
    eventLog.add(SessionEvent(t, label));
  }

  SessionSummary? stopRecording({double accelHz = 0.0, double gyroHz = 0.0}) {
    if (!isRecording) return null;
    isRecording = false;
    if (_samples.isEmpty) return null;

    final n = _samples.length;
    final duration = _samples.last.timestampS - _samples.first.timestampS;

    var denied = 0;
    var mlSum = 0.0, gnssSum = 0.0, qualSum = 0.0;
    var diffSum = 0.0;
    var diffCount = 0;
    var distance = 0.0;

    for (var i = 0; i < n; i++) {
      final s = _samples[i];
      if (s.gnssMode == 'DENIED') denied++;
      mlSum += s.mlSpeed;
      gnssSum += s.gnssSpeed;
      qualSum += s.qualityScore;

      // Only compare against GNSS where GNSS was actually trusted.
      if (s.gnssMode != 'DENIED') {
        diffSum += (s.mlSpeed - s.gnssSpeed).abs();
        diffCount++;
      }
      if (i > 0) {
        final p = _samples[i - 1];
        distance += math.sqrt(math.pow(s.ekfEast - p.ekfEast, 2) +
            math.pow(s.ekfNorth - p.ekfNorth, 2));
      }
    }

    final summary = SessionSummary(
      sessionName: _sessionName,
      startedAt: _startedAt ?? DateTime.now(),
      durationS: duration,
      totalSamples: n,
      gnssAvailablePct: 100.0 * (n - denied) / n,
      gnssDeniedPct: 100.0 * denied / n,
      meanMlSpeedMps: mlSum / n,
      meanGnssSpeedMps: gnssSum / n,
      meanSpeedDiffMps: diffCount > 0 ? diffSum / diffCount : 0.0,
      distanceM: distance,
      accelHz: accelHz,
      gyroHz: gyroHz,
      meanQualityScore: qualSum / n,
      events: List.unmodifiable(eventLog),
    );

    completedSessions.add(summary);
    return summary;
  }

  String exportCsv() {
    final b = StringBuffer()..writeln(SessionSample.csvHeader);
    for (final s in _samples) {
      b.writeln(s.toCsvRow().join(','));
    }
    return b.toString();
  }
}
