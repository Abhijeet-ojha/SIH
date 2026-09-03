import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/sensors/sensor_calibrator.dart';

void main() {
  group('SensorCalibrator Tests', () {
    test('Calculates stationary biases and applies software corrections', () {
      final calibrator = SensorCalibrator();
      expect(calibrator.status, equals(CalibrationStatus.uncalibrated));

      calibrator.startCalibration(durationSec: 0.0);
      for (int i = 0; i < 30; i++) {
        calibrator.feedRawSample(
          ax: 0.05,
          ay: 0.02,
          az: 9.85,
          gx: 0.01,
          gy: -0.01,
          gz: 0.03,
        );
      }

      expect(calibrator.status, equals(CalibrationStatus.completed));
      expect(calibrator.isCalibrated, isTrue);

      final res = calibrator.currentCalibration!;
      expect(res.gyroBiasZ, closeTo(0.03, 0.005));
      expect(res.totalSamplesCollected, greaterThanOrEqualTo(20));

      // Test applied corrections
      final (cAx, cAy, cAz, cGx, cGy, cGz) = calibrator.applyCorrection(
        rawAx: 0.05,
        rawAy: 0.02,
        rawAz: 9.85,
        rawGx: 0.01,
        rawGy: -0.01,
        rawGz: 0.03,
      );

      expect(cGz, closeTo(0.0, 0.005));
      expect(cGx, closeTo(0.0, 0.005));
    });

    test('Rejects calibration if motion is detected during stationary wizard', () {
      final calibrator = SensorCalibrator();
      calibrator.startCalibration(durationSec: 0.0);

      // Feed high dynamic motion
      for (int i = 0; i < 30; i++) {
        calibrator.feedRawSample(
          ax: (i % 2 == 0) ? 5.0 : -5.0,
          ay: 1.0,
          az: 9.8,
          gx: 0.5,
          gy: 0.2,
          gz: 0.8,
        );
      }

      expect(calibrator.status, equals(CalibrationStatus.failed));
      expect(calibrator.isCalibrated, isFalse);
    });
  });
}
