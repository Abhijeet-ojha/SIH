import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:sensors_plus_platform_interface/sensors_plus_platform_interface.dart';
import 'package:navpulse_localizer/main.dart';
import 'package:navpulse_localizer/state/navigation_state_provider.dart';

class FakeSensors extends SensorsPlatform {
  final accel = StreamController<AccelerometerEvent>.broadcast();
  final gyro = StreamController<GyroscopeEvent>.broadcast();
  @override
  Stream<AccelerometerEvent> accelerometerEventStream({Duration samplingPeriod = const Duration(milliseconds: 20)}) => accel.stream;
  @override
  Stream<GyroscopeEvent> gyroscopeEventStream({Duration samplingPeriod = const Duration(milliseconds: 20)}) => gyro.stream;
}
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Indoor start needs neither GNSS permission nor a GPS fix; stop releases sensors', () async {
    final old = SensorsPlatform.instance;
    final fake = FakeSensors(); SensorsPlatform.instance = fake;
    var gpsCalls = 0;
    final messenger = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(const MethodChannel('flutter.baseflow.com/geolocator'), (call) async {
      gpsCalls++; throw PlatformException(code: 'LOCATION_DISABLED');
    });
    messenger.setMockMethodCallHandler(const MethodChannel('navpulse/relative_heading'), (_) async => null);
    final nav = NavigationStateProvider();
    await nav.start(); await Future<void>.delayed(Duration.zero);
    expect(nav.isRunning, true); expect(nav.hasFix, false);
    expect(gpsCalls, 0); expect(nav.track.length, 1);
    expect(fake.accel.hasListener, true);
    await nav.stop();
    expect(fake.accel.hasListener, false);
    expect(fake.gyro.hasListener, false);
    expect(nav.isRunning, false);
    nav.dispose(); SensorsPlatform.instance = old;
    await fake.accel.close(); await fake.gyro.close();
    messenger.setMockMethodCallHandler(const MethodChannel('navpulse/relative_heading'), null);
    messenger.setMockMethodCallHandler(const MethodChannel('flutter.baseflow.com/geolocator'), null);
  });
  testWidgets('Phone portrait shows indoor status and opens walking controls without overflow', (tester) async {
    tester.view.physicalSize = const Size(390, 844);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final nav = NavigationStateProvider();
    await tester.pumpWidget(ChangeNotifierProvider.value(value: nav, child: const NavPulseApp()));
    await tester.pump();
    expect(find.text('INDOOR WALK / READY'), findsOneWidget);
    expect(find.text('SEARCHING'), findsNothing);
    expect(find.text('0 steps / 0.0 m'), findsOneWidget);
    await tester.tap(find.text('Navigate')); await tester.pumpAndSettle();
    expect(find.text('WALKING SETUP'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.ensureVisible(find.text('Pipeline')); await tester.pumpAndSettle();
    await tester.tap(find.text('Pipeline')); await tester.pumpAndSettle();
    expect(find.text('PEDESTRIAN PIPELINE'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox()); nav.dispose();
  });
}
