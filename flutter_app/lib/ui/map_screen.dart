import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../localization/ekf_fusion_engine.dart';
import '../localization/motion_gate.dart';
import '../state/navigation_state_provider.dart';
import 'hud_controls.dart';
import 'map_canvas.dart';
import 'safety_panel.dart';
import 'sheet_panels.dart';
import 'theme.dart';

/// Apple Maps shell: a full-bleed map with floating controls and a draggable sheet.
///
/// The layout follows Maps because the interaction model is the right one for a phone in a
/// cradle - the map owns the screen, one thumb-reachable sheet holds everything else, and
/// the sheet can be pushed away entirely when the driver only wants the road.
class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  final _sheetController = DraggableScrollableController();
  int _panel = 0;

  /// Start/stop, with any failure surfaced to the user. A demo that silently does nothing
  /// when a permission is refused is worse than one that says why.
  Future<void> _startStop(
      BuildContext context, NavigationStateProvider nav) async {
    try {
      if (nav.isRunning) {
        await nav.stop();
      } else {
        await nav.start();
      }
      if (context.mounted && nav.lastError != null) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text(nav.lastError!)));
      }
    } catch (error) {
      await nav.stop();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Unable to start sensors: $error')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final nav = context.watch<NavigationStateProvider>();
    final media = MediaQuery.of(context);

    return Scaffold(
      backgroundColor: NavTheme.mapLand,
      body: Stack(
        children: [
          Positioned.fill(
            child: MapCanvas(
              track: List<TrackPoint>.of(nav.track),
              indoor: nav.indoorMode,
              followHeading: false,
              headingRad: nav.ekf.heading,
              basemap: nav.basemap,
              anchor: nav.anchorLat == null
                  ? null
                  : GeoAnchor(nav.anchorLat!, nav.anchorLon!, nav.anchorEast,
                      nav.anchorNorth),
              destination: nav.destinationLocal,
              onPickDestination: (e, n) {
                nav.setDestinationLocal(e, n);
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                    content: Text('Destination set. Long-press elsewhere to '
                        'move it.'),
                    duration: Duration(seconds: 2)));
              },
            ),
          ),

          // Everything a judge needs to operate the demo lives in one column at the top,
          // in reading order: which mode, what the system currently believes, and any
          // condition worth shouting about.
          Positioned(
            top: media.padding.top + 8,
            left: 12,
            right: 12,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ModeSwitcher(nav: nav),
                const SizedBox(height: 8),
                _StatusCapsule(nav: nav),
                if (nav.blackoutSimulated) const SizedBox(height: 8),
                if (nav.blackoutSimulated) _BlackoutBanner(nav: nav),
                if (nav.frame.mountDisturbed) const SizedBox(height: 8),
                if (nav.frame.mountDisturbed) HeadingHeldBanner(nav: nav),
              ],
            ),
          ),

          if (nav.hasDestination)
            Positioned(
              left: 12,
              right: 12,
              top: media.padding.top + 118,
              child: _GuidanceCard(nav: nav),
            ),

          // First-run guidance, only while idle - it disappears the moment the demo
          // starts so it never competes with the live track.
          if (!nav.isRunning)
            Positioned(
              left: 12,
              right: 12,
              top: media.size.height * 0.30,
              child: CoachCard(nav: nav),
            ),

          // SOS sits on the map, not buried in the sheet. The whole point is that it can
          // be found without reading anything, by someone who is not calm.
          Positioned(
            left: 12,
            bottom: media.size.height * 0.42 + 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                SosButton(nav: nav),
                const SizedBox(height: 10),
                MapLegend(indoor: nav.indoorMode),
              ],
            ),
          ),

          Positioned(
            right: 12,
            bottom: media.size.height * 0.42 + 16,
            child: PrimaryActions(
              nav: nav,
              onStartStop: () => _startStop(context, nav),
            ),
          ),

          DraggableScrollableSheet(
            controller: _sheetController,
            initialChildSize: 0.40,
            minChildSize: 0.14,
            maxChildSize: 0.92,
            snap: true,
            snapSizes: const [0.14, 0.40, 0.92],
            builder: (context, scrollController) => _Sheet(
              scrollController: scrollController,
              panel: _panel,
              onPanelChanged: (i) {
                setState(() => _panel = i);
                _sheetController.animateTo(0.92,
                    duration: const Duration(milliseconds: 260),
                    curve: Curves.easeOutCubic);
              },
            ),
          ),
        ],
      ),
    );
  }
}

/// Distance and relative bearing to the destination.
///
/// Deliberately not turn-by-turn. There is no road graph on the device, so instructions
/// like "turn left in 200 m" would be fabricated. What the system genuinely knows is how
/// far away the destination is and which way it lies relative to the direction of travel,
/// and that is what is shown.
class _GuidanceCard extends StatelessWidget {
  final NavigationStateProvider nav;
  const _GuidanceCard({required this.nav});

  @override
  Widget build(BuildContext context) {
    final d = nav.destinationDistanceM ?? 0;
    final rel = nav.destinationRelativeBearingDeg ?? 0;

    final String turn;
    if (d < 15) {
      turn = 'You have arrived';
    } else if (rel.abs() < 20) {
      turn = 'Straight ahead';
    } else if (rel.abs() > 150) {
      turn = 'Turn around';
    } else {
      turn = '${rel.abs().round()}° to the ${rel > 0 ? "right" : "left"}';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
      decoration: BoxDecoration(
        color: NavTheme.sheet.withOpacity(0.94),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: NavTheme.good.withOpacity(0.5)),
      ),
      child: Row(
        children: [
          Transform.rotate(
            angle: rel * math.pi / 180.0,
            child: const Icon(Icons.navigation, color: NavTheme.good, size: 30),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  d >= 1000
                      ? '${(d / 1000).toStringAsFixed(2)} km'
                      : '${d.round()} m',
                  style: const TextStyle(
                      color: NavTheme.label,
                      fontSize: 19,
                      fontWeight: FontWeight.w700),
                ),
                Text('$turn · straight-line, no road data offline',
                    style: const TextStyle(
                        color: NavTheme.secondaryLabel, fontSize: 11)),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close, color: NavTheme.tertiaryLabel),
            onPressed: nav.clearDestination,
          ),
        ],
      ),
    );
  }
}

class _StatusCapsule extends StatelessWidget {
  final NavigationStateProvider nav;
  const _StatusCapsule({required this.nav});

  @override
  Widget build(BuildContext context) {
    final (label, colour) = nav.indoorMode
        ? (
            nav.isRunning ? 'INDOOR WALK / GPS OFF' : 'INDOOR WALK / READY',
            NavTheme.good
          )
        : switch (nav.ekf.gnssState) {
            GnssNavMode.gnssDenied => ('GNSS DENIED', NavTheme.denied),
            GnssNavMode.gnssReacquired => ('REACQUIRED', NavTheme.good),
            GnssNavMode.gnssNormal => nav.hasFix
                ? ('GNSS LOCKED', NavTheme.good)
                : ('SEARCHING', NavTheme.bad),
          };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: NavTheme.sheet.withOpacity(0.92),
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.4),
              blurRadius: 18,
              offset: const Offset(0, 6)),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 9,
            height: 9,
            decoration: BoxDecoration(color: colour, shape: BoxShape.circle),
          ),
          const SizedBox(width: 10),
          Flexible(
              child: Text(label,
                  style: TextStyle(
                      color: colour,
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.6))),
          const Spacer(),
          Text(
            nav.gnssAccuracy > 0
                ? '±${nav.gnssAccuracy.toStringAsFixed(0)} m'
                : '—',
            style:
                const TextStyle(color: NavTheme.secondaryLabel, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _BlackoutBanner extends StatelessWidget {
  final NavigationStateProvider nav;
  const _BlackoutBanner({required this.nav});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: NavTheme.denied.withOpacity(0.16),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: NavTheme.denied.withOpacity(0.5)),
      ),
      child: Row(
        children: [
          const Icon(Icons.satellite_alt, color: NavTheme.denied, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Dead reckoning · ${nav.ekf.blackoutDurationS.toStringAsFixed(0)} s '
              'without satellites',
              style: const TextStyle(
                  color: NavTheme.denied,
                  fontSize: 13,
                  fontWeight: FontWeight.w600),
            ),
          ),
          Text('±${nav.positionSigmaM.toStringAsFixed(0)} m',
              style: const TextStyle(
                  color: NavTheme.denied,
                  fontSize: 15,
                  fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}

/// The sheet. Collapsed it shows speed and status; expanded it becomes the technical
/// panels, so nothing was removed from the old five-tab build - it was moved somewhere a
/// driver is not forced to look at it.
class _Sheet extends StatelessWidget {
  final ScrollController scrollController;
  final int panel;
  final ValueChanged<int> onPanelChanged;

  const _Sheet({
    required this.scrollController,
    required this.panel,
    required this.onPanelChanged,
  });

  @override
  Widget build(BuildContext context) {
    final nav = context.watch<NavigationStateProvider>();

    return Container(
      decoration: const BoxDecoration(
        color: NavTheme.sheet,
        borderRadius: BorderRadius.vertical(top: Radius.circular(14)),
        boxShadow: [BoxShadow(color: Colors.black54, blurRadius: 24)],
      ),
      child: ListView(
        controller: scrollController,
        padding: EdgeInsets.zero,
        children: [
          const SizedBox(height: 8),
          Center(
            child: Container(
              width: 36,
              height: 5,
              decoration: BoxDecoration(
                color: NavTheme.tertiaryLabel,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),
          const SizedBox(height: 14),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 18),
            child: _SpeedHeader(nav: nav),
          ),
          const SizedBox(height: 16),
          _PanelSelector(selected: panel, onChanged: onPanelChanged),
          const SizedBox(height: 6),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
            child: switch (panel) {
              1 => DiagnosticsPanel(nav: nav),
              2 => AnalyticsPanel(nav: nav),
              3 => SafetyPanel(nav: nav),
              4 => SessionsPanel(nav: nav),
              5 => SettingsPanel(nav: nav),
              _ => NavigationPanel(nav: nav),
            },
          ),
        ],
      ),
    );
  }
}

class _SpeedHeader extends StatelessWidget {
  final NavigationStateProvider nav;
  const _SpeedHeader({required this.nav});

  @override
  Widget build(BuildContext context) {
    final (stateLabel, stateColour) = switch (nav.motionState) {
      MotionState.inVehicleMoving => ('In vehicle', NavTheme.good),
      MotionState.stationary => ('Stopped', NavTheme.secondaryLabel),
      MotionState.phoneHandled => ('Phone handled', NavTheme.denied),
    };

    return Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        FittedBox(fit: BoxFit.scaleDown, alignment: Alignment.centerLeft,
          child: Row(crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic, children: [
            Text(nav.speedKmh.round().toString(), style: const TextStyle(
              fontSize: 56, fontWeight: FontWeight.w300, height: 1, color: NavTheme.label)),
            const SizedBox(width: 6),
            const Text('km/h', style: TextStyle(fontSize: 15, color: NavTheme.secondaryLabel)),
          ])),
        Text(nav.indoorMode
            ? '${nav.pedestrian.steps} steps / ${nav.pedestrian.distanceM.toStringAsFixed(1)} m'
            : stateLabel, style: TextStyle(color: stateColour, fontSize: 13)),
      ])),
      const SizedBox(width: 12),
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
        Text(nav.indoorMode ? 'EST. DRIFT' : 'UNCERTAINTY',
          style: const TextStyle(fontSize: 10, letterSpacing: 1, color: NavTheme.tertiaryLabel)),
        Text('${nav.positionSigmaM.toStringAsFixed(1)} m',
          style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: NavTheme.good)),
        Text(nav.indoorMode ? '${nav.headingDeg.round()} deg from start'
            : '${nav.headingDeg.round()} deg ${_cardinal(nav.headingDeg)}',
          textAlign: TextAlign.right,
          style: const TextStyle(fontSize: 12, color: NavTheme.secondaryLabel)),
      ])),
    ]);
  }

  static String _cardinal(double deg) {
    const d = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    return d[(((deg + 22.5) % 360) / 45).floor().clamp(0, 7)];
  }
}

class _PanelSelector extends StatelessWidget {
  final int selected;
  final ValueChanged<int> onChanged;
  const _PanelSelector({required this.selected, required this.onChanged});

  static const _items = [
    (Icons.navigation, 'Navigate'),
    (Icons.monitor_heart, 'Sensors'),
    (Icons.analytics, 'Pipeline'),
    (Icons.shield, 'Safety'),
    (Icons.folder_special, 'Sessions'),
    (Icons.settings, 'Settings'),
  ];

  /// Wrapped rather than horizontally scrolled. Six labelled destinations do not fit one
  /// phone-width row, and a scrolling strip hides the last of them behind a gesture nobody
  /// discovers - the panel would simply not exist for a judge holding the phone for the
  /// first time. Two visible rows cost 40 px of sheet and nothing else.
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          for (var i = 0; i < _items.length; i++)
            GestureDetector(
              onTap: () => onChanged(i),
              child: Container(
                height: 38,
                padding: const EdgeInsets.symmetric(horizontal: 13),
                decoration: BoxDecoration(
                  color:
                      i == selected ? NavTheme.accent : NavTheme.sheetElevated,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(_items[i].$1,
                        size: 16,
                        color: i == selected
                            ? Colors.white
                            : NavTheme.secondaryLabel),
                    const SizedBox(width: 6),
                    Text(_items[i].$2,
                        style: TextStyle(
                            fontSize: 13,
                            fontWeight: i == selected
                                ? FontWeight.w600
                                : FontWeight.w400,
                            color: i == selected
                                ? Colors.white
                                : NavTheme.secondaryLabel)),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
