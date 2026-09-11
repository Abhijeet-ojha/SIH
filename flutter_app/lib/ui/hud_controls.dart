import 'package:flutter/material.dart';

import '../state/navigation_state_provider.dart';
import 'theme.dart';

/// Foreground controls for the map.
///
/// Design rule throughout this file: nothing important is icon-only. Tooltips do not
/// appear on a touch device without a long-press, so an unlabelled round button is
/// invisible to anyone who has not been told what it does — which in a demo is everyone.
/// Every control here carries a word.

/// Mode selector. The single most important control in the app and previously buried in a
/// sheet panel as a SwitchListTile, where a judge would never find it. Now it is the first
/// thing on screen.
class ModeSwitcher extends StatelessWidget {
  final NavigationStateProvider nav;
  const ModeSwitcher({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final locked = nav.isRunning;
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: NavTheme.sheet.withOpacity(0.94),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.45),
              blurRadius: 18,
              offset: const Offset(0, 6)),
        ],
      ),
      child: Row(
        children: [
          _segment(
            label: 'VEHICLE',
            sub: 'GPS + inertial',
            icon: Icons.directions_car_filled,
            selected: !nav.indoorMode,
            locked: locked,
            onTap: () => nav.setIndoorMode(false),
          ),
          _segment(
            label: 'INDOOR WALK',
            sub: 'no GPS at all',
            icon: Icons.directions_walk,
            selected: nav.indoorMode,
            locked: locked,
            onTap: () => nav.setIndoorMode(true),
          ),
        ],
      ),
    );
  }

  Widget _segment({
    required String label,
    required String sub,
    required IconData icon,
    required bool selected,
    required bool locked,
    required VoidCallback onTap,
  }) {
    final fg = selected
        ? Colors.white
        : (locked ? NavTheme.tertiaryLabel : NavTheme.secondaryLabel);
    return Expanded(
      child: GestureDetector(
        onTap: locked ? null : onTap,
        behavior: HitTestBehavior.opaque,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(vertical: 9),
          decoration: BoxDecoration(
            color: selected ? NavTheme.accent : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(icon, size: 16, color: fg),
                  const SizedBox(width: 6),
                  Flexible(
                    child: Text(label,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                            color: fg,
                            fontSize: 12.5,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.4)),
                  ),
                ],
              ),
              const SizedBox(height: 1),
              Text(locked ? 'stop to switch' : sub,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                      color: selected
                          ? Colors.white.withOpacity(0.85)
                          : NavTheme.tertiaryLabel,
                      fontSize: 9.5)),
            ],
          ),
        ),
      ),
    );
  }
}

/// The two actions that drive the whole demo, as labelled pills rather than icons.
class PrimaryActions extends StatelessWidget {
  final NavigationStateProvider nav;
  final VoidCallback onStartStop;
  const PrimaryActions(
      {super.key, required this.nav, required this.onStartStop});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: [
        // Withholding GNSS is the entire pitch, so it gets a full-width labelled control
        // that states what it will do and, while active, what is happening.
        if (!nav.indoorMode)
          _pill(
            label: nav.blackoutSimulated ? 'RESTORE GPS' : 'CUT GPS SIGNAL',
            sub: nav.blackoutSimulated
                ? 'inertial only'
                : (nav.isRunning ? 'simulate a tunnel' : 'start first'),
            icon: nav.blackoutSimulated
                ? Icons.satellite_alt
                : Icons.signal_cellular_nodata,
            colour: nav.blackoutSimulated ? NavTheme.good : NavTheme.denied,
            filled: nav.blackoutSimulated,
            onTap: nav.isRunning ? nav.toggleBlackout : null,
          ),
        if (!nav.indoorMode) const SizedBox(height: 10),
        _pill(
          label: nav.isRunning ? 'STOP' : 'START',
          sub: nav.isRunning
              ? 'recording'
              : (nav.indoorMode ? 'walk with the phone' : 'drive'),
          icon: nav.isRunning ? Icons.stop_rounded : Icons.play_arrow_rounded,
          colour: nav.isRunning ? NavTheme.bad : NavTheme.good,
          filled: true,
          big: true,
          onTap: onStartStop,
        ),
      ],
    );
  }

  Widget _pill({
    required String label,
    required String sub,
    required IconData icon,
    required Color colour,
    required bool filled,
    bool big = false,
    VoidCallback? onTap,
  }) {
    final enabled = onTap != null;
    final bg = !enabled
        ? NavTheme.sheetElevated.withOpacity(0.9)
        : (filled ? colour : NavTheme.sheet.withOpacity(0.94));
    final fg = !enabled
        ? NavTheme.tertiaryLabel
        : (filled ? Colors.white : colour);

    return Material(
      color: bg,
      borderRadius: BorderRadius.circular(16),
      elevation: 8,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Container(
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: big ? 13 : 11),
          constraints: const BoxConstraints(minWidth: 168),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: filled || !enabled
                ? null
                : Border.all(color: colour.withOpacity(0.65), width: 1.5),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: fg, size: big ? 24 : 20),
              const SizedBox(width: 10),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(label,
                      style: TextStyle(
                          color: fg,
                          fontSize: big ? 16 : 14,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.5)),
                  Text(sub,
                      style: TextStyle(
                          color: fg.withOpacity(0.8), fontSize: 10.5)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Explains the two track colours without anyone having to narrate it. A demo where the
/// key visual needs a spoken caption is a demo that fails when the room is noisy.
class MapLegend extends StatelessWidget {
  final bool indoor;
  const MapLegend({super.key, required this.indoor});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: NavTheme.sheet.withOpacity(0.9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _key(NavTheme.accent, indoor ? 'Walked path' : 'GPS available'),
          const SizedBox(height: 5),
          _key(NavTheme.denied,
              indoor ? 'Sensors only' : 'GPS denied — inertial only'),
        ],
      ),
    );
  }

  Widget _key(Color c, String label) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
              width: 14,
              height: 4,
              decoration: BoxDecoration(
                  color: c, borderRadius: BorderRadius.circular(2))),
          const SizedBox(width: 7),
          Text(label,
              style: const TextStyle(
                  color: NavTheme.secondaryLabel, fontSize: 11)),
        ],
      );
}

/// Shown before a session starts. A judge picks the phone up cold; this tells them what to
/// press without anyone standing over their shoulder.
class CoachCard extends StatelessWidget {
  final NavigationStateProvider nav;
  const CoachCard({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final steps = nav.indoorMode
        ? const [
            'Hold the phone flat, top edge forward.',
            'Press START, then walk normally.',
            'Turn your body and phone together.',
          ]
        : const [
            'Mount the phone, then press START.',
            'Drive. The blue trail follows GPS.',
            'Press CUT GPS SIGNAL — the trail turns amber and continues on sensors alone.',
          ];

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: BoxDecoration(
        color: NavTheme.sheet.withOpacity(0.95),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: NavTheme.accent.withOpacity(0.35)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.lightbulb_outline,
                  size: 16, color: NavTheme.accent),
              const SizedBox(width: 8),
              Text(nav.indoorMode ? 'INDOOR DEMO' : 'VEHICLE DEMO',
                  style: const TextStyle(
                      color: NavTheme.accent,
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.0)),
            ],
          ),
          const SizedBox(height: 10),
          for (var i = 0; i < steps.length; i++)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 18,
                    height: 18,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                        color: NavTheme.accent.withOpacity(0.18),
                        borderRadius: BorderRadius.circular(9)),
                    child: Text('${i + 1}',
                        style: const TextStyle(
                            color: NavTheme.accent,
                            fontSize: 10,
                            fontWeight: FontWeight.w700)),
                  ),
                  const SizedBox(width: 9),
                  Expanded(
                    child: Text(steps[i],
                        style: const TextStyle(
                            color: NavTheme.secondaryLabel,
                            fontSize: 12.5,
                            height: 1.35)),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// Surfaces the mount-disturbance guard.
///
/// Before the guard existed, slanting the phone rotated the whole map — the system read
/// phone rotation as vehicle rotation. Now that rotation is rejected, and saying so turns
/// an invisible correction into the most demonstrable feature in the app: a judge can pick
/// the phone up, twist it, and watch the heading refuse to move.
class HeadingHeldBanner extends StatelessWidget {
  final NavigationStateProvider nav;
  const HeadingHeldBanner({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: NavTheme.accent.withOpacity(0.18),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: NavTheme.accent.withOpacity(0.55)),
      ),
      child: Row(
        children: [
          const Icon(Icons.screen_rotation_alt,
              color: NavTheme.accent, size: 18),
          const SizedBox(width: 10),
          const Expanded(
            child: Text(
              'HEADING HELD — phone moved, not the vehicle',
              style: TextStyle(
                  color: NavTheme.accent,
                  fontSize: 12,
                  fontWeight: FontWeight.w700),
            ),
          ),
          Text('${(nav.frame.yawTrust * 100).round()}%',
              style: const TextStyle(
                  color: NavTheme.accent,
                  fontSize: 13,
                  fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}
