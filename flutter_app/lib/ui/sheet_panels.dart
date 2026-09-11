import 'package:flutter/material.dart';

import '../localization/speed_model.dart';
import '../models/sensor_quality.dart';
import '../sensors/sensor_calibrator.dart';
import '../state/navigation_state_provider.dart';
import 'theme.dart';

/// Section heading in the Apple grouped-list idiom: small, uppercase, low contrast.
class _Section extends StatelessWidget {
  final String title;
  final List<Widget> children;
  const _Section(this.title, this.children);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 18, 4, 8),
          child: Text(title.toUpperCase(),
              style: const TextStyle(
                  fontSize: 11,
                  letterSpacing: 0.9,
                  fontWeight: FontWeight.w600,
                  color: NavTheme.tertiaryLabel)),
        ),
        Container(
          decoration: NavTheme.groupedCard,
          child: Column(children: children),
        ),
      ],
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColour;
  final bool last;
  const _Row(this.label, this.value, {this.valueColour, this.last = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        border: last
            ? null
            : const Border(
                bottom: BorderSide(color: NavTheme.separator, width: 0.5)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(label,
                style: const TextStyle(fontSize: 14, color: NavTheme.label)),
          ),
          const SizedBox(width: 12),
          Flexible(
              child: Text(value,
                  textAlign: TextAlign.right,
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: valueColour ?? NavTheme.secondaryLabel))),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class NavigationPanel extends StatelessWidget {
  final NavigationStateProvider nav;
  const NavigationPanel({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Indoor walking / GPS off'),
          subtitle: Text(nav.isRunning
              ? 'Stop to change mode'
              : 'Works from a local start point; no GPS permission needed'),
          value: nav.indoorMode,
          onChanged: nav.isRunning ? null : nav.setIndoorMode,
        ),
        if (nav.lastError != null)
          Padding(
              padding: const EdgeInsets.all(8),
              child: Text(nav.lastError!,
                  style: const TextStyle(color: NavTheme.bad))),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: NavTheme.groupedCard,
          child: Row(
            children: [
              Icon(
                  nav.motionState.name == 'phoneHandled'
                      ? Icons.pan_tool
                      : Icons.directions_car,
                  color: NavTheme.secondaryLabel,
                  size: 20),
              const SizedBox(width: 12),
              Expanded(
                child: Text(nav.motionReason,
                    style: const TextStyle(
                        fontSize: 13, color: NavTheme.secondaryLabel)),
              ),
            ],
          ),
        ),
        if (nav.indoorMode) ...[
          _Section('Walking setup', [
            _Row('Detected steps', '${nav.pedestrian.steps}'),
            _Row('Estimated distance',
                '${nav.pedestrian.distanceM.toStringAsFixed(2)} m'),
            _Row('Heading source', nav.headingSource, last: true),
          ]),
          Text(
              'Step length: ${nav.pedestrian.stepLengthM.toStringAsFixed(2)} m'),
          Slider(
              value: nav.pedestrian.stepLengthM,
              min: 0.35,
              max: 0.95,
              divisions: 12,
              onChanged: nav.isRunning ? null : nav.setStepLength),
          Text(
              'Step threshold: ${nav.pedestrian.threshold.toStringAsFixed(1)} (lower = more sensitive)'),
          Slider(
              value: nav.pedestrian.threshold,
              min: 0.4,
              max: 1.6,
              divisions: 12,
              onChanged: nav.isRunning ? null : nav.setStepThreshold),
          const Text(
              'Hold screen-up, top edge forward. Turn your body and phone together. '
              'Use normal steps. This estimates a relative path, not a room map. '
              'Distance uses your step length; drift is an unvalidated allowance.',
              style: TextStyle(fontSize: 12, color: NavTheme.secondaryLabel)),
        ],
        _Section('Position estimate', [
          _Row(nav.indoorMode ? 'Right of start (X)' : 'East',
              '${nav.ekf.posEast.toStringAsFixed(1)} m'),
          _Row(nav.indoorMode ? 'Forward of start (Y)' : 'North',
              '${nav.ekf.posNorth.toStringAsFixed(1)} m'),
          _Row('Forward speed', '${nav.ekf.speed.toStringAsFixed(2)} m/s'),
          _Row('Lateral speed',
              '${nav.ekf.lateralSpeed.toStringAsFixed(3)} m/s'),
          _Row('Heading', '${nav.headingDeg.toStringAsFixed(1)}°'),
          _Row('Gyro bias', '${nav.ekf.gyroBias.toStringAsFixed(5)} rad/s',
              last: true),
        ]),
        _Section('GNSS', [
          _Row(
              'Mode',
              nav.indoorMode
                  ? 'OFF / local coordinates'
                  : nav.ekf.gnssState.name),
          _Row(
              'Fix',
              nav.indoorMode
                  ? 'not requested'
                  : nav.hasFix
                      ? 'acquired'
                      : 'searching',
              valueColour: nav.hasFix ? NavTheme.good : NavTheme.bad),
          _Row(
              'Accuracy',
              nav.gnssAccuracy > 0
                  ? '±${nav.gnssAccuracy.toStringAsFixed(1)} m'
                  : '—'),
          _Row('Reported speed', '${nav.gnssSpeed.toStringAsFixed(2)} m/s'),
          _Row('Blackout elapsed',
              '${nav.ekf.blackoutDurationS.toStringAsFixed(1)} s',
              last: true),
        ]),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class DiagnosticsPanel extends StatelessWidget {
  final NavigationStateProvider nav;
  const DiagnosticsPanel({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final r = nav.qualityReport;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (r != null) ...[
          _Section('Sensor streams', [
            _streamRow(r.accel),
            _streamRow(r.gyro),
            _streamRow(r.mag, last: true),
          ]),
          _Section('Health', [
            _Row('Overall score', '${(r.overallScore * 100).round()}%',
                valueColour: r.overallScore > 0.8
                    ? NavTheme.good
                    : (r.overallScore > 0.5 ? NavTheme.denied : NavTheme.bad)),
            _Row('Active faults',
                r.activeFaults.isEmpty ? 'none' : r.activeFaults.join(', '),
                valueColour:
                    r.activeFaults.isEmpty ? NavTheme.good : NavTheme.bad,
                last: true),
          ]),
          if (r.recentAnomalies.isNotEmpty)
            _Section('Recent anomalies', [
              for (var i = 0; i < r.recentAnomalies.length.clamp(0, 5); i++)
                _Row(r.recentAnomalies[i].code, r.recentAnomalies[i].detail,
                    valueColour: NavTheme.denied,
                    last: i == r.recentAnomalies.length.clamp(0, 5) - 1),
            ]),
        ] else
          const Padding(
            padding: EdgeInsets.all(20),
            child: Text('Start a session to see sensor diagnostics.',
                style: TextStyle(color: NavTheme.secondaryLabel)),
          ),
        _CalibrationCard(nav: nav),
      ],
    );
  }

  Widget _streamRow(SensorDiagnostic d, {bool last = false}) {
    final colour = switch (d.freshnessStatus) {
      SensorFreshnessStatus.live => NavTheme.good,
      SensorFreshnessStatus.stale => NavTheme.denied,
      SensorFreshnessStatus.unavailable => NavTheme.bad,
    };
    return _Row(
      d.sensorName,
      '${d.actualHz.toStringAsFixed(1)} Hz · ±${d.jitterMs.toStringAsFixed(1)} ms',
      valueColour: colour,
      last: last,
    );
  }
}

class _CalibrationCard extends StatelessWidget {
  final NavigationStateProvider nav;
  const _CalibrationCard({required this.nav});

  @override
  Widget build(BuildContext context) {
    final c = nav.calibrator;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Section('Calibration', [
          _Row('Status', c.status.name,
              valueColour: switch (c.status) {
                CalibrationStatus.completed => NavTheme.good,
                CalibrationStatus.failed => NavTheme.bad,
                _ => NavTheme.secondaryLabel,
              }),
          if (c.currentCalibration != null)
            _Row('Gyro bias |b|',
                '${c.currentCalibration!.gyroBiasMagnitude.toStringAsFixed(5)} rad/s'),
          _Row('Detail', c.message, last: true),
        ]),
        const SizedBox(height: 12),
        // Why this matters, stated where the user is deciding whether to bother.
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            'A stationary gyro still reports a small non-zero rate. Left uncorrected, '
            '0.03 rad/s integrates to about 155° of heading error over a 90-second '
            'blackout. Park, hold still, and tap below.',
            style: TextStyle(
                fontSize: 12, color: NavTheme.tertiaryLabel, height: 1.4),
          ),
        ),
        const SizedBox(height: 12),
        FilledButton(
          onPressed: nav.isRunning ? nav.startCalibration : null,
          style: FilledButton.styleFrom(
              backgroundColor: NavTheme.accent,
              minimumSize: const Size.fromHeight(46)),
          child: Text(c.status == CalibrationStatus.calibrating
              ? 'Calibrating… ${(c.progress * 100).round()}%'
              : 'Calibrate sensors'),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class AnalyticsPanel extends StatelessWidget {
  final NavigationStateProvider nav;
  const AnalyticsPanel({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final ekf = nav.ekf;
    if (nav.indoorMode) {
      return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        _Section('Pedestrian pipeline', [
          _Row('Accelerometer',
              '${nav.qualityReport?.accel.actualHz.toStringAsFixed(1) ?? "0"} Hz'),
          _Row('Step signal', nav.pedestrian.signal.toStringAsFixed(2)),
          _Row('Detector', nav.pedestrian.ready ? 'ready' : 'hold still'),
          _Row('Steps', '${nav.pedestrian.steps}'),
          _Row('Heading', nav.headingSource),
          _Row('GNSS / network', 'not used'),
          _Row('Vehicle speed model', 'not used', last: true),
        ]),
        const Text(
            'Live accelerometer cycles + relative phone rotation + configured step length. '
            'Walking speed and position are estimates; repeated hand shaking can cause false steps.'),
      ]);
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Section('Speed sources', [
          _Row('Model (on-device)', '${nav.mlSpeed.toStringAsFixed(2)} m/s',
              valueColour: NavTheme.accent),
          _Row('GNSS', '${nav.gnssSpeed.toStringAsFixed(2)} m/s'),
          _Row('Fused (EKF)', '${ekf.speed.toStringAsFixed(2)} m/s',
              valueColour: NavTheme.good),
          _Row('Model σ (measured)',
              '±${nav.mlUncertainty.toStringAsFixed(2)} m/s',
              last: true),
        ]),
        _Section('Pipeline', [
          _stage('IMU ingest', ekf.isImuFused),
          _stage('Frame alignment', nav.frame.forwardConfidence >= 0),
          _stage('Motion gate', true),
          _stage('Speed model', nav.modelLoaded && ekf.isMlFused),
          _stage('NHC constraint', ekf.isNhcFused),
          _stage('ZUPT', ekf.isZuptFused),
          _stage('GNSS update', ekf.isGnssFused, last: true),
        ]),
        // What the app has learned about THIS phone in THIS car while GPS was up. Shown
        // with its evidence, because a calibration number without provenance is not worth
        // trusting - and because watching it converge is the demo.
        _Section('Learned from this drive', [
          _Row('Gyro bias',
              '${(nav.calibration.yawBias * 180 / 3.14159265).toStringAsFixed(3)}°/s',
              valueColour: nav.calibration.yawBiasConfidence > 0.5
                  ? NavTheme.good
                  : NavTheme.secondaryLabel),
          _Row('  from stationary time',
              '${nav.calibration.stationarySeconds.toStringAsFixed(0)} s · '
              '${(nav.calibration.yawBiasConfidence * 100).round()}% confident'),
          _Row('Gyro scale', '${nav.calibration.yawScale.toStringAsFixed(4)}×',
              valueColour: nav.calibration.yawScaleConfidence > 0.5
                  ? NavTheme.good
                  : NavTheme.secondaryLabel),
          _Row('  from GPS-scored turns',
              '${nav.calibration.headingWindows} windows · '
              '${(nav.calibration.yawScaleConfidence * 100).round()}% confident'),
          _Row('Speed correction',
              '×${nav.calibration.speedGain.toStringAsFixed(2)}'
              '${nav.calibration.speedOffset >= 0 ? '+' : ''}'
              '${nav.calibration.speedOffset.toStringAsFixed(2)}',
              valueColour: nav.calibration.speedConfidence > 0.5
                  ? NavTheme.good
                  : NavTheme.secondaryLabel),
          _Row('  from GPS-scored samples',
              '${nav.calibration.speedSamples} · '
              '${(nav.calibration.speedConfidence * 100).round()}% confident'),
          _Row('Speed error it now expects',
              '±${nav.calibration.speedSigma.toStringAsFixed(2)} m/s',
              last: true),
        ]),
        const SizedBox(height: 10),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            nav.calibration.hasLearnedAnything
                ? 'Learned while satellites were visible and applied the moment they are '
                    'not. Nothing here was learned from our own estimate — only from GPS '
                    'and from standing still — so it cannot drift into believing itself.'
                : 'Drive with a GPS fix and this fills in. Until it does, every correction '
                    'is the identity, so an uncalibrated phone behaves exactly as before.',
            style: const TextStyle(
                fontSize: 11, color: NavTheme.tertiaryLabel, height: 1.4),
          ),
        ),
        _Section('Model', [
          _Row('Loaded', nav.modelLoaded ? 'yes' : 'no',
              valueColour: nav.modelLoaded ? NavTheme.good : NavTheme.bad),
          _Row('Features', '${SpeedModel.featureNames.length}'),
          _Row('Forward-axis confidence',
              '${(nav.frame.forwardConfidence * 100).round()}%',
              last: true),
        ]),
        const SizedBox(height: 10),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            nav.modelProvenance.isEmpty
                ? 'Model provenance unavailable.'
                : nav.modelProvenance,
            style: const TextStyle(
                fontSize: 11, color: NavTheme.tertiaryLabel, height: 1.4),
          ),
        ),
      ],
    );
  }

  Widget _stage(String name, bool active, {bool last = false}) => _Row(
        name,
        active ? 'active' : 'idle',
        valueColour: active ? NavTheme.good : NavTheme.tertiaryLabel,
        last: last,
      );
}

// ─────────────────────────────────────────────────────────────────────────────

class SessionsPanel extends StatelessWidget {
  final NavigationStateProvider nav;
  const SessionsPanel({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    final rec = nav.recorder;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Section('Current session', [
          _Row('Recording', rec.isRecording ? 'yes' : 'no',
              valueColour:
                  rec.isRecording ? NavTheme.good : NavTheme.secondaryLabel),
          _Row('Samples', '${rec.sampleCount}'),
          _Row('Track points', '${nav.track.length}', last: true),
        ]),
        if (rec.eventLog.isNotEmpty)
          _Section('Event log', [
            for (var i = rec.eventLog.length - 1;
                i >= 0 && i > rec.eventLog.length - 7;
                i--)
              _Row('${rec.eventLog[i].timestampS.toStringAsFixed(1)} s',
                  rec.eventLog[i].label,
                  last: i == rec.eventLog.length - 6 || i == 0),
          ]),
        if (rec.completedSessions.isNotEmpty)
          _Section('Completed', [
            for (var i = 0; i < rec.completedSessions.length; i++)
              _Row(
                rec.completedSessions[i].sessionName,
                '${rec.completedSessions[i].totalSamples} samples · '
                '${rec.completedSessions[i].gnssDeniedPct.toStringAsFixed(0)}% denied',
                last: i == rec.completedSessions.length - 1,
              ),
          ]),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class SettingsPanel extends StatelessWidget {
  final NavigationStateProvider nav;
  const SettingsPanel({super.key, required this.nav});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Section('Architecture', [
          _Row('Network use', 'none'),
          _Row('Map tiles', 'rendered locally'),
          _Row('Model', 'bundled asset, 43 KB'),
          _Row('Inference', 'on-device', last: true),
        ]),
        const SizedBox(height: 10),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            'Everything here runs offline by construction. There is no tile server, no '
            'API key and no upload — a system whose claim is that it works where GNSS '
            'does not should not quietly require a network to demonstrate it.',
            style: TextStyle(
                fontSize: 12, color: NavTheme.tertiaryLabel, height: 1.4),
          ),
        ),
        _Section('Honest limits', [
          _Row('Blackout drift (real data)', '51% median'),
          _Row('Target', '10%'),
          _Row('Drives meeting target', '0 of 6'),
          _Row('Model R² (held-out)', 'negative on 2 of 3', last: true),
        ]),
        const SizedBox(height: 10),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            'Measured on real IO-VNBD drives, reported as exit error ÷ distance travelled '
            'with GNSS off. The system does not yet meet the target and the numbers above '
            'say so rather than hiding it.',
            style: TextStyle(
                fontSize: 12, color: NavTheme.tertiaryLabel, height: 1.4),
          ),
        ),
      ],
    );
  }
}
