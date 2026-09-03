import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:navpulse_localizer/main.dart';
import 'package:navpulse_localizer/state/navigation_state_provider.dart';

void main() {
  testWidgets('Renders 5-Tab Technical Navigation Shell without network', (WidgetTester tester) async {
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider(create: (_) => NavigationStateProvider()),
        ],
        child: const NavPulseApp(),
      ),
    );

    // Initial render: Navigation Screen HUD
    expect(find.text('NAVPULSE INSTRUMENT'), findsOneWidget);
    expect(find.text('CURRENT SPEED'), findsOneWidget);
    expect(find.text('LOCAL ENU VECTOR TRAJECTORY (OFFLINE)'), findsOneWidget);

    // Switch to Diagnostics tab
    await tester.tap(find.byIcon(Icons.monitor_heart));
    await tester.pumpAndSettle();
    expect(find.text('SENSOR DIAGNOSTICS HUD'), findsOneWidget);
    expect(find.text('TRI-AXIAL ACCELEROMETER'), findsOneWidget);
    expect(find.text('ON-DEVICE SENSOR CALIBRATION WIZARD'), findsOneWidget);

    // Switch to Analytics tab
    await tester.tap(find.byIcon(Icons.analytics));
    await tester.pumpAndSettle();
    expect(find.text('NAVIGATION ANALYTICS & PIPELINE'), findsOneWidget);
    expect(find.text('REAL-TIME SPEED SUBSYSTEM COMPARISON'), findsOneWidget);
    expect(find.text('END-TO-END PIPELINE MONITOR'), findsOneWidget);

    // Switch to Sessions tab
    await tester.tap(find.byIcon(Icons.folder_special));
    await tester.pumpAndSettle();
    expect(find.text('LOCAL SESSIONS & ANALYTICS'), findsOneWidget);
    expect(find.text('LOCAL STANDALONE RECORDER'), findsOneWidget);

    // Switch to Settings tab
    await tester.tap(find.byIcon(Icons.settings));
    await tester.pumpAndSettle();
    expect(find.text('SYSTEM CONFIGURATION'), findsOneWidget);
    expect(find.text('100% OFFLINE-FIRST ARCHITECTURE'), findsOneWidget);
  });
}
