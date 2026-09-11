import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:navpulse_localizer/localization/pedestrian_tracker.dart';

void warmup(PedestrianTracker p) {
  for (var t = 0; t < 2000; t += 20) { p.addAcceleration(0, 0, 9.80665, t); }
}
void walk(PedestrianTracker p, int start, int cycles) {
  for (var t = start; t < start + cycles * 500; t += 20) {
    p.addAcceleration(0, 0, 9.80665 + 1.8 * math.sin(2 * math.pi * (t - start) / 500), t);
  }
}
void main() {
  test('Stationary noise and pure phone tilt do not create travel', () {
    final p = PedestrianTracker();
    for (var t = 0; t < 30000; t += 20) {
      final a = t / 900.0;
      final g = 9.80665 + 0.03 * math.sin(t.toDouble());
      p.addAcceleration(g * math.sin(a), 0, g * math.cos(a), t);
    }
    expect(p.steps, 0);
    expect(p.distanceM, 0);
  });
  test('Walking advances measured cycles and stops without continued drift', () {
    final p = PedestrianTracker();
    warmup(p);
    walk(p, 2000, 20);
    expect(p.steps, inInclusiveRange(18, 20));
    expect(p.north, closeTo(p.steps * p.stepLengthM, 1e-9));
    expect(p.east, closeTo(0, 1e-9));
    expect(p.speedAt(12000), greaterThan(0));
    for (var t = 12000; t < 15000; t += 20) { p.addAcceleration(0, 0, 9.80665, t); }
    final stopped = p.distanceM;
    for (var t = 15000; t < 20000; t += 20) { p.addAcceleration(0, 0, 9.80665, t); }
    expect(p.speedAt(20000), 0);
    expect(p.distanceM, stopped);
  });
  test('Relative 90-degree turn changes trajectory, without GPS', () {
    final p = PedestrianTracker();
    p.observeHeading(1.1);
    warmup(p);
    walk(p, 2000, 10);
    final firstNorth = p.north;
    p.observeHeading(1.1 + math.pi / 2);
    walk(p, 7000, 10);
    expect(p.north, closeTo(firstNorth, 1e-8));
    expect(p.east, greaterThan(5));
    expect(p.heading, closeTo(math.pi / 2, 1e-8));
  });
  test('Gentle handheld gait is detected at the default sensitivity', () {
    final p = PedestrianTracker(); warmup(p);
    for (var t = 2000; t < 12000; t += 20) {
      p.addAcceleration(0, 0, 9.80665 + 0.95 * math.sin(2 * math.pi * (t - 2000) / 500), t);
    }
    expect(p.steps, inInclusiveRange(18, 20));
  });
  test('Reset clears step history and uses a new relative heading origin', () {
    final p = PedestrianTracker()..stepLengthM = 0.75;
    warmup(p); walk(p, 2000, 10); p.observeHeading(2);
    p.reset(); p.observeHeading(-2);
    expect(p.steps, 0); expect(p.east, 0); expect(p.north, 0);
    expect(p.heading, 0); expect(p.ready, false); expect(p.stepLengthM, 0.75);
  });
  test('Non-finite inputs and repeated timestamps do not create steps', () {
    final p = PedestrianTracker(); warmup(p);
    for (var i = 0; i < 100; i++) {
      p.addAcceleration(double.nan, 0, 10, 2000);
      p.addAcceleration(0, 0, 50, 2000);
      p.addAcceleration(0, 0, 10, 2000);
    }
    expect(p.steps, 0);
  });
}
