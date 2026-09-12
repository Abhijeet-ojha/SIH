import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../state/navigation_state_provider.dart';
import 'theme.dart';

/// The safety surface: an always-visible SOS control and the panel behind it.
///
/// Two things are deliberate here.
///
/// A confirmation step sits between the button and the radio. An SOS that fires on a
/// pocket-tap trains people to ignore it, and the message goes to real phones. The dialog
/// shows the EXACT text that will be sent, so the sender knows what the recipient will see -
/// including the accuracy circle and how old the fix is.
///
/// When there is no GNSS anchor the button says so and still works, but what it sends says
/// "position unknown" rather than a coordinate. A responder given a fabricated location is
/// worse off than one given none.
class SosButton extends StatelessWidget {
  final NavigationStateProvider nav;
  const SosButton({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final r = nav.report;
    final ready = r.isShareable;

    return Material(
      color: NavTheme.bad,
      shape: const CircleBorder(),
      elevation: 10,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: () => showSosConfirm(context, nav),
        child: SizedBox(
          width: 76,
          height: 76,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.sos_rounded, color: Colors.white, size: 26),
              Text(
                ready ? 'SHARE' : 'NO FIX',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.9),
                  fontSize: 9.5,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.6,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Preview-then-send. Returns after the attempt so the caller can show the outcome.
Future<void> showSosConfirm(
    BuildContext context, NavigationStateProvider nav) async {
  final report = nav.report;
  final body = report.smsBody(name: nav.safety.ownerName);

  if (!nav.safety.hasContacts) {
    await showSafetyContactsSheet(context, nav);
    return;
  }

  final go = await showDialog<bool>(
    context: context,
    builder: (context) => AlertDialog(
      backgroundColor: NavTheme.sheetElevated,
      title: const Text('Send this to your contacts?',
          style: TextStyle(color: NavTheme.label, fontSize: 17)),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: NavTheme.sheet,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(body,
                style: const TextStyle(
                    color: NavTheme.label,
                    fontSize: 12.5,
                    height: 1.35,
                    fontFamily: 'monospace')),
          ),
          const SizedBox(height: 10),
          Text('To: ${nav.safety.contacts.join(", ")}',
              style: const TextStyle(
                  color: NavTheme.secondaryLabel, fontSize: 11.5)),
        ],
      ),
      actions: [
        TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel')),
        FilledButton(
          style: FilledButton.styleFrom(backgroundColor: NavTheme.bad),
          onPressed: () => Navigator.pop(context, true),
          child: const Text('SEND'),
        ),
      ],
    ),
  );

  if (go != true || !context.mounted) return;
  final res = await nav.panic();
  if (!context.mounted) return;
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
    content: Text(res.detail),
    backgroundColor:
        res.reachedSomeone ? NavTheme.good : NavTheme.sheetElevated,
  ));
}

Future<void> showSafetyContactsSheet(
        BuildContext context, NavigationStateProvider nav) =>
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: NavTheme.sheet,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16))),
      builder: (context) => Padding(
        padding: EdgeInsets.only(
            left: 18,
            right: 18,
            top: 18,
            bottom: MediaQuery.of(context).viewInsets.bottom + 24),
        child: SafetyContactsEditor(nav: nav),
      ),
    );

class SafetyContactsEditor extends StatefulWidget {
  final NavigationStateProvider nav;
  const SafetyContactsEditor({super.key, required this.nav});

  @override
  State<SafetyContactsEditor> createState() => _SafetyContactsEditorState();
}

class _SafetyContactsEditorState extends State<SafetyContactsEditor> {
  final _number = TextEditingController();
  late final _name =
      TextEditingController(text: widget.nav.safety.ownerName ?? '');

  @override
  void dispose() {
    _number.dispose();
    _name.dispose();
    super.dispose();
  }

  void _add() {
    final n = _number.text.trim();
    if (n.isEmpty) return;
    widget.nav.addSafetyContact(n);
    _number.clear();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final safety = widget.nav.safety;
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Emergency contacts',
            style: TextStyle(
                color: NavTheme.label,
                fontSize: 18,
                fontWeight: FontWeight.w700)),
        const SizedBox(height: 4),
        const Text(
            'Sent by SMS, so it works with no internet — in a basement, a tunnel or '
            'on one bar of signal.',
            style: TextStyle(color: NavTheme.secondaryLabel, fontSize: 12)),
        const SizedBox(height: 16),
        TextField(
          controller: _name,
          style: const TextStyle(color: NavTheme.label),
          decoration: _dec('Your name (appears in the message)'),
          onChanged: widget.nav.setSafetyOwnerName,
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _number,
                keyboardType: TextInputType.phone,
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'[0-9+ \-]')),
                ],
                style: const TextStyle(color: NavTheme.label),
                decoration: _dec('Phone number'),
                onSubmitted: (_) => _add(),
              ),
            ),
            const SizedBox(width: 10),
            FilledButton(
              style: FilledButton.styleFrom(backgroundColor: NavTheme.accent),
              onPressed: _add,
              child: const Text('Add'),
            ),
          ],
        ),
        const SizedBox(height: 14),
        if (safety.contacts.isEmpty)
          const Text('No contacts yet. SOS cannot send without one.',
              style: TextStyle(color: NavTheme.denied, fontSize: 12))
        else
          ...safety.contacts.map((c) => ListTile(
                contentPadding: EdgeInsets.zero,
                dense: true,
                leading: const Icon(Icons.person, color: NavTheme.accent),
                title: Text(c,
                    style: const TextStyle(color: NavTheme.label, fontSize: 14)),
                trailing: IconButton(
                  icon: const Icon(Icons.close, color: NavTheme.tertiaryLabel),
                  onPressed: () {
                    widget.nav.removeSafetyContact(c);
                    setState(() {});
                  },
                ),
              )),
      ],
    );
  }

  InputDecoration _dec(String hint) => InputDecoration(
        hintText: hint,
        hintStyle:
            const TextStyle(color: NavTheme.tertiaryLabel, fontSize: 13),
        filled: true,
        fillColor: NavTheme.sheetElevated,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
      );
}

/// Sheet panel: what would be sent right now, who it goes to, and what is still queued.
class SafetyPanel extends StatelessWidget {
  final NavigationStateProvider nav;
  const SafetyPanel({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final r = nav.report;
    final safety = nav.safety;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(r.isShareable ? Icons.place : Icons.location_disabled,
                      color: r.isShareable ? NavTheme.good : NavTheme.denied,
                      size: 18),
                  const SizedBox(width: 8),
                  Text(
                    r.isShareable ? 'Your position' : 'No GPS fix yet',
                    style: const TextStyle(
                        color: NavTheme.label,
                        fontSize: 15,
                        fontWeight: FontWeight.w700),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              if (r.isShareable) ...[
                SelectableText(
                  '${r.lat.toStringAsFixed(6)}, ${r.lon.toStringAsFixed(6)}',
                  style: const TextStyle(
                      color: NavTheme.label,
                      fontSize: 16,
                      fontFamily: 'monospace'),
                ),
                const SizedBox(height: 6),
                _line('Accurate to',
                    '±${r.uncertaintyM.toStringAsFixed(0)} m (${r.qualityWord})'),
                _line('Last real GPS', _age(r.fixAgeSeconds)),
                _line('Since then, by sensors',
                    '${r.distanceSinceFixM.toStringAsFixed(0)} m'),
              ] else
                Text(
                  'The track is relative to where the session started. Once a single '
                  'GPS fix arrives, everything walked since is converted to a real '
                  'latitude and longitude — including the '
                  '${r.distanceSinceFixM.toStringAsFixed(0)} m covered so far.',
                  style: const TextStyle(
                      color: NavTheme.secondaryLabel, fontSize: 12.5, height: 1.4),
                ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        _card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Expanded(
                    child: Text('Emergency contacts',
                        style: TextStyle(
                            color: NavTheme.label,
                            fontSize: 15,
                            fontWeight: FontWeight.w700)),
                  ),
                  TextButton(
                    onPressed: () => showSafetyContactsSheet(context, nav),
                    child: const Text('Edit'),
                  ),
                ],
              ),
              Text(
                safety.contacts.isEmpty
                    ? 'None set — add one before the demo.'
                    : safety.contacts.join(', '),
                style: TextStyle(
                    color: safety.contacts.isEmpty
                        ? NavTheme.denied
                        : NavTheme.secondaryLabel,
                    fontSize: 12.5),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                      backgroundColor: NavTheme.bad,
                      padding: const EdgeInsets.symmetric(vertical: 14)),
                  icon: const Icon(Icons.sos_rounded),
                  label: const Text('SEND MY LOCATION',
                      style: TextStyle(fontWeight: FontWeight.w800)),
                  onPressed: () => showSosConfirm(context, nav),
                ),
              ),
              if (safety.pendingCount > 0) ...[
                const SizedBox(height: 10),
                Text(
                  '${safety.pendingCount} message(s) waiting for signal — they go out '
                  'automatically when it returns.',
                  style: const TextStyle(color: NavTheme.denied, fontSize: 11.5),
                ),
              ],
            ],
          ),
        ),
        if (safety.log.isNotEmpty) ...[
          const SizedBox(height: 12),
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Activity',
                    style: TextStyle(
                        color: NavTheme.label,
                        fontSize: 15,
                        fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                ...safety.log.take(8).map((l) => Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Text(l,
                          style: const TextStyle(
                              color: NavTheme.secondaryLabel,
                              fontSize: 11.5,
                              fontFamily: 'monospace')),
                    )),
              ],
            ),
          ),
        ],
      ],
    );
  }

  static String _age(double s) {
    if (!s.isFinite) return 'never';
    if (s < 60) return '${s.ceil()} s ago';
    return '${(s / 60).ceil()} min ago';
  }

  static Widget _card({required Widget child}) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: NavTheme.sheetElevated,
          borderRadius: BorderRadius.circular(14),
        ),
        child: child,
      );

  static Widget _line(String k, String v) => Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(k,
                style: const TextStyle(
                    color: NavTheme.secondaryLabel, fontSize: 12.5)),
            Text(v,
                style: const TextStyle(
                    color: NavTheme.label,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600)),
          ],
        ),
      );
}
