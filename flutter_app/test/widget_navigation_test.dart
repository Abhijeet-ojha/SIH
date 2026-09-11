import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:navpulse_localizer/main.dart';
import 'package:navpulse_localizer/state/navigation_state_provider.dart';

/// The shell was restructured from a five-tab technical HUD into an Apple Maps layout:
/// a full-bleed map owns the screen, and everything else lives in a draggable sheet.
///
/// This test was rewritten to match. The five panels still exist and still carry the same
/// information - they moved into the sheet rather than being deleted - so the assertions
/// below walk the same five destinations through the new selector.
void main() {
  Widget harness() => MultiProvider(
        providers: [
          ChangeNotifierProvider(
              create: (_) => NavigationStateProvider()..setIndoorMode(false)),
        ],
        child: const NavPulseApp(),
      );

  testWidgets('Renders the map shell and sheet without network',
      (tester) async {
    await tester.pumpWidget(harness());
    await tester.pump();

    // Map is the primary surface; speed and uncertainty are the headline readouts.
    expect(find.text('km/h'), findsOneWidget);
    expect(find.text('UNCERTAINTY'), findsOneWidget);

    // Collapsed sheet shows the panel selector.
    expect(find.text('Navigate'), findsOneWidget);
    expect(find.text('Sensors'), findsOneWidget);
    expect(find.text('Pipeline'), findsOneWidget);
    expect(find.text('Sessions'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
  });

  testWidgets('Every sheet panel opens and renders its content',
      (tester) async {
    await tester.pumpWidget(harness());
    await tester.pump();

    await tester.tap(find.text('Navigate'));
    await tester.pumpAndSettle();
    expect(find.text('POSITION ESTIMATE'), findsOneWidget);
    expect(find.text('GNSS'), findsOneWidget);

    await tester.tap(find.text('Sensors'));
    await tester.pumpAndSettle();
    expect(find.text('CALIBRATION'), findsOneWidget);
    expect(find.text('Calibrate sensors'), findsOneWidget);

    await tester.tap(find.text('Pipeline'));
    await tester.pumpAndSettle();
    expect(find.text('SPEED SOURCES'), findsOneWidget);
    expect(find.text('PIPELINE'), findsOneWidget);
    expect(find.text('MODEL'), findsOneWidget);

    await tester.tap(find.text('Sessions'));
    await tester.pumpAndSettle();
    expect(find.text('CURRENT SESSION'), findsOneWidget);

    await tester.tap(find.text('Settings'));
    await tester.pumpAndSettle();
    expect(find.text('ARCHITECTURE'), findsOneWidget);
    // The app states its own measured limits rather than only its strengths.
    expect(find.text('HONEST LIMITS'), findsOneWidget);
  });

  testWidgets('Blackout banner appears only while GNSS is withheld',
      (tester) async {
    final nav = NavigationStateProvider()..setIndoorMode(false);
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider<NavigationStateProvider>.value(value: nav)
        ],
        child: const NavPulseApp(),
      ),
    );
    await tester.pump();
    expect(find.textContaining('Dead reckoning'), findsNothing);

    nav.toggleBlackout();
    await tester.pump();
    expect(find.textContaining('Dead reckoning'), findsOneWidget);
  });
}
