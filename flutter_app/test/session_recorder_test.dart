import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/analytics/session_recorder.dart';
import 'package:navpulse_localizer/models/session_data.dart';

void main() {
  group('SessionRecorder Tests', () {
    test('Logs telemetry samples and computes post-drive engineering summary', () {
      final recorder = SessionRecorder();
      expect(recorder.isRecording, isFalse);

      recorder.startRecording(sessionName: 'TestDrive_01');
      expect(recorder.isRecording, isTrue);
      expect(recorder.eventLog.length, greaterThanOrEqualTo(1));

      // Feed 20 samples (10 Normal, 10 Denied)
      for (int i = 0; i < 20; i++) {
        final isDenied = i >= 10;
        recorder.recordSample(SessionSample(
          timestampS: 1000.0 + i * 0.1,
          gnssLat: 12.9716,
          gnssLon: 77.5946,
          gnssSpeed: 10.0,
          gnssAccuracy: 4.0,
          gnssMode: isDenied ? 'DENIED' : 'NORMAL',
          ax: 0.1, ay: 0.2, az: 9.81,
          gx: 0.0, gy: 0.0, gz: 0.01,
          mlSpeed: 9.8,
          mlUncertainty: 0.45,
          ekfEast: i * 1.0,
          ekfNorth: i * 0.5,
          ekfSpeed: 9.9,
          ekfHeading: 45.0,
          ekfGyroBias: 0.001,
          qualityScore: 1.0,
        ));
      }

      final summary = recorder.stopRecording(accelHz: 50.0, gyroHz: 50.0);
      expect(summary, isNotNull);
      expect(summary!.totalSamples, equals(20));
      expect(summary.gnssAvailablePct, closeTo(50.0, 1.0));
      expect(summary.gnssDeniedPct, closeTo(50.0, 1.0));
      expect(summary.meanMlSpeedMps, closeTo(9.8, 0.1));
      expect(summary.meanSpeedDiffMps, closeTo(0.2, 0.05));
      expect(recorder.completedSessions.length, equals(1));
    });
  });
}
