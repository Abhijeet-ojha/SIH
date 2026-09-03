import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/localization/ekf_fusion_engine.dart';

void main() {
  group('EkfFusionEngine6State Offline Tests', () {
    test('Propagates kinematic state with NHC and ZUPT constraints', () {
      final ekf = EkfFusionEngine6State(initX: 0.0, initY: 0.0, initV: 10.0, initHeading: 0.0);

      // 1. Predict 1 second of forward motion at 10 m/s heading North (0 rad)
      for (int i = 0; i < 10; i++) {
        ekf.predict(
          dt: 0.1,
          vAi: 10.0,
          vAiStd: 0.45,
          gyroZ: 0.0,
          isStationary: false,
        );
        ekf.updateNhc();
      }

      expect(ekf.x[1], greaterThan(5.0)); // North position increased
      expect(ekf.x[3].abs(), lessThan(0.1)); // Lateral velocity constrained near 0
      expect(ekf.isImuFused, isTrue);
      expect(ekf.isMlFused, isTrue);
      expect(ekf.isNhcFused, isTrue);

      // 2. Standstill ZUPT test
      ekf.predict(
        dt: 0.1,
        vAi: 0.0,
        vAiStd: 0.10,
        gyroZ: 0.0,
        isStationary: true,
      );
      ekf.updateZupt();
      expect(ekf.isZuptFused, isTrue);
      expect(ekf.x[2], lessThan(5.0)); // Forward velocity clamped down
    });

    test('Transitions GNSS state machine without position divergence', () {
      final ekf = EkfFusionEngine6State(initX: 0.0, initY: 0.0, initV: 5.0, initHeading: 0.0);
      expect(ekf.gnssState, equals(GnssNavMode.gnssNormal));

      // Enter blackout
      ekf.setGnssBlackout(true);
      expect(ekf.gnssState, equals(GnssNavMode.gnssDenied));

      // Propagate in blackout
      for (int i = 0; i < 20; i++) {
        ekf.predict(
          dt: 0.1,
          vAi: 5.0,
          vAiStd: 0.50,
          gyroZ: 0.0,
        );
      }
      expect(ekf.blackoutDurationS, greaterThan(1.8));

      // Restore GNSS
      ekf.setGnssBlackout(false);
      expect(ekf.gnssState, equals(GnssNavMode.gnssReacquired));
    });
  });
}
