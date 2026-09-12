import 'dart:async';

import 'package:flutter/services.dart';

import 'position_report.dart';

enum ShareOutcome {
  sent,          // handed to the radio
  composed,      // messaging app opened, prefilled - the user still presses send
  queued,        // no signal; held for retry
  refused,       // nothing to share (no GNSS anchor, or no recipients)
  failed,
}

class ShareResult {
  final ShareOutcome outcome;
  final String detail;
  const ShareResult(this.outcome, this.detail);
  bool get reachedSomeone => outcome == ShareOutcome.sent;
}

/// Gets a position to another human when there is no internet.
///
/// SMS, because it is the only channel that survives the conditions this system is built
/// for. A data connection is the first thing to go in a basement, a tunnel, a rural road or
/// a crowd, and it is precisely then that someone needs to be told where you are. SMS rides
/// the control channel and gets through on a single bar.
///
/// Two delivery paths, in order:
///
///   1. Direct send via the platform (SmsManager). One tap, no further interaction - which
///      matters when the user may not be free to look at the screen.
///   2. If that is unavailable or not permitted, the messaging app is opened prefilled.
///      Slower and needs a second tap, but it needs no dangerous permission and always
///      works, so it is never left with nothing.
///
/// Anything that cannot go out right now is QUEUED rather than dropped, and flushed when
/// signal returns. A safety message that silently evaporated is worse than no feature.
class SafetyService {
  static const MethodChannel _channel = MethodChannel('navpulse/safety');

  /// Recipients. Kept in memory here; persisting them is the same one-line swap as the
  /// calibration store.
  final List<String> contacts = [];
  String? ownerName;

  /// Reports that could not be delivered yet, oldest first.
  final List<PositionReport> pending = [];
  static const int maxPending = 50;

  /// Every attempt, for the UI and for after-the-fact review. A safety feature that cannot
  /// show what it did is not auditable.
  final List<String> log = [];

  bool get hasContacts => contacts.isNotEmpty;
  int get pendingCount => pending.length;

  void addContact(String number) {
    final n = number.trim();
    if (n.isEmpty || contacts.contains(n)) return;
    contacts.add(n);
  }

  void removeContact(String number) => contacts.remove(number);

  void _note(String msg, DateTime now) {
    log.insert(0,
        '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}  $msg');
    while (log.length > 40) {
      log.removeLast();
    }
  }

  /// Share [report] with every contact.
  ///
  /// [now] is injected so this is testable without a wall clock.
  Future<ShareResult> share(
    PositionReport report, {
    bool emergency = true,
    DateTime? now,
  }) async {
    final t = now ?? report.at;

    if (contacts.isEmpty) {
      _note('refused: no emergency contacts set', t);
      return const ShareResult(
          ShareOutcome.refused, 'Add at least one contact first.');
    }
    if (!report.isShareable) {
      // Deliberately not sending a fabricated coordinate. Sending (0,0) or the session
      // origin would be worse than sending nothing - it would send help somewhere wrong.
      pending.add(report);
      _trimPending();
      _note('no GNSS anchor yet - report queued, not sent', t);
      return const ShareResult(ShareOutcome.queued,
          'No GPS fix yet, so there is no position to send. Held until one arrives.');
    }

    final body = report.smsBody(name: ownerName, emergency: emergency);

    try {
      final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'sendSms',
        {'recipients': contacts, 'body': body},
      );
      final status = (result?['status'] ?? 'failed').toString();
      if (status == 'sent') {
        _note('sent to ${contacts.length} contact(s)', t);
        return ShareResult(
            ShareOutcome.sent, 'Sent to ${contacts.length} contact(s).');
      }
      if (status == 'composed') {
        _note('messaging app opened, awaiting send', t);
        return const ShareResult(ShareOutcome.composed,
            'Messaging app opened with the message ready - press send.');
      }
      pending.add(report);
      _trimPending();
      _note('send failed ($status) - queued', t);
      return ShareResult(
          ShareOutcome.queued, 'No signal. Queued and will retry ($status).');
    } on MissingPluginException {
      // Running in a test or on a platform with no native side.
      _note('no platform channel - queued', t);
      pending.add(report);
      _trimPending();
      return const ShareResult(
          ShareOutcome.queued, 'Messaging unavailable here. Queued.');
    } catch (e) {
      pending.add(report);
      _trimPending();
      _note('send error - queued', t);
      return ShareResult(ShareOutcome.queued, 'Queued after an error: $e');
    }
  }

  /// Retry everything held. Called when a GNSS fix or signal returns.
  ///
  /// Only the newest unanchored report is worth keeping - a queue full of "position
  /// unknown" is noise - but every anchored one is a distinct place the person was, and
  /// together they are a trail, so all of those are sent.
  Future<int> flush({DateTime? now}) async {
    if (pending.isEmpty || contacts.isEmpty) return 0;
    final batch = List<PositionReport>.from(pending);
    pending.clear();
    var sent = 0;
    for (final r in batch) {
      if (!r.isShareable) continue; // stale "unknown position" entries are dropped
      final res = await share(r, now: now);
      if (res.outcome == ShareOutcome.sent ||
          res.outcome == ShareOutcome.composed) {
        sent++;
      }
    }
    return sent;
  }

  void _trimPending() {
    while (pending.length > maxPending) {
      pending.removeAt(0);
    }
  }

  void clear() {
    pending.clear();
    log.clear();
  }
}
