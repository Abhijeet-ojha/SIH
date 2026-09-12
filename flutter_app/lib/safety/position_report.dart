import 'dart:math' as math;

/// An absolute, shareable position with an honest error bound.
///
/// Everything else in this app works in a local ENU frame measured from wherever the
/// session started. That is fine for drawing a track and useless for safety: somebody
/// coming to help needs a latitude and longitude, and needs to know how much to trust it.
///
/// So a report carries three things, not one:
///
///   lat/lon          the anchor (last real GNSS fix) plus the dead-reckoned offset
///   uncertaintyM     how far out that could be, growing with distance since the fix
///   fixAgeSeconds    how long ago the last real fix was
///
/// The last two are not decoration. This system's measured vehicle drift is around half
/// the distance travelled during a blackout, and the pedestrian drift coefficient below is
/// an allowance rather than a measurement. Reporting a bare coordinate from that would be
/// telling a responder to search the wrong building with total confidence. A circle and a
/// staleness are more useful to them AND more honest.
class PositionReport {
  final double lat;
  final double lon;

  /// Radius in metres within which the person is expected to be. GNSS accuracy at the
  /// anchor, plus accumulated dead-reckoning drift since.
  final double uncertaintyM;

  /// Seconds since the last real GNSS fix. 0 means this IS a fix.
  final double fixAgeSeconds;

  /// Distance dead-reckoned since the anchor.
  final double distanceSinceFixM;

  /// False when no GNSS fix has ever been obtained, in which case lat/lon are meaningless
  /// and only the relative track exists. Callers must not share an unanchored report as a
  /// location - see [isShareable].
  final bool gpsAnchored;

  final String mode; // 'walking' or 'vehicle'
  final DateTime at;

  const PositionReport({
    required this.lat,
    required this.lon,
    required this.uncertaintyM,
    required this.fixAgeSeconds,
    required this.distanceSinceFixM,
    required this.gpsAnchored,
    required this.mode,
    required this.at,
  });

  static const double earthRadiusM = 6371000.0;

  /// Fraction of distance travelled that pedestrian dead reckoning is assumed to lose.
  ///
  /// UNVALIDATED. Published step-length methods land around 5-15% and this is deliberately
  /// set above that range, because for a safety feature an over-large circle costs a
  /// searcher a little time while an over-small one sends them to the wrong place. Replace
  /// it with a measured figure before claiming any accuracy number.
  static const double pedestrianDriftFraction = 0.20;

  /// Local east/north metres from an anchor to a geographic point, and back.
  ///
  /// Equirectangular, which is exact enough here by a wide margin: over the few kilometres
  /// a dead-reckoning session covers, the error against a proper geodesic is centimetres,
  /// far below the metres of drift the estimate already carries. Both directions live here
  /// so the map, the taps on it and the shared coordinate can never disagree.
  static (double, double) latLonFromEnu(
      double anchorLat, double anchorLon, double eastM, double northM) {
    final lat = anchorLat + (northM / earthRadiusM) * 180.0 / math.pi;
    // cos(lat) shrinks the metres-per-degree of longitude away from the equator. Guarded
    // so a position near the poles cannot divide by zero.
    final cosLat = math.max(math.cos(anchorLat * math.pi / 180.0).abs(), 1e-6);
    final lon = anchorLon + (eastM / (earthRadiusM * cosLat)) * 180.0 / math.pi;
    return (lat, lon);
  }

  static (double, double) enuFromLatLon(
      double anchorLat, double anchorLon, double lat, double lon) {
    final cosLat = math.max(math.cos(anchorLat * math.pi / 180.0).abs(), 1e-6);
    final north = (lat - anchorLat) * math.pi / 180.0 * earthRadiusM;
    final east = (lon - anchorLon) * math.pi / 180.0 * earthRadiusM * cosLat;
    return (east, north);
  }

  /// Build a report from an anchor fix and a local ENU offset accumulated since.
  factory PositionReport.fromAnchor({
    required double anchorLat,
    required double anchorLon,
    required double anchorAccuracyM,
    required double eastM,
    required double northM,
    required double fixAgeSeconds,
    required double distanceSinceFixM,
    required double drUncertaintyM,
    required bool gpsAnchored,
    required String mode,
    required DateTime at,
  }) {
    final (lat, lon) = latLonFromEnu(anchorLat, anchorLon, eastM, northM);

    return PositionReport(
      lat: lat,
      lon: lon,
      // Errors from independent sources add in quadrature rather than linearly - adding
      // them straight would overstate the circle after a long walk.
      uncertaintyM: math.sqrt(anchorAccuracyM * anchorAccuracyM +
          drUncertaintyM * drUncertaintyM),
      fixAgeSeconds: fixAgeSeconds,
      distanceSinceFixM: distanceSinceFixM,
      gpsAnchored: gpsAnchored,
      mode: mode,
      at: at,
    );
  }

  /// Dead-reckoning uncertainty for a walked distance. Kept separate so it can be tested
  /// and replaced independently of the geodesy.
  static double pedestrianDrift(double distanceM) =>
      0.3 + distanceM * pedestrianDriftFraction;

  /// A report with no GNSS anchor is not a location and must never be sent as one.
  bool get isShareable => gpsAnchored && lat.isFinite && lon.isFinite;

  /// Rough confidence wording for a non-technical recipient.
  String get qualityWord {
    if (!gpsAnchored) return 'NO FIX';
    if (uncertaintyM <= 15) return 'good';
    if (uncertaintyM <= 50) return 'approximate';
    return 'rough';
  }

  String get mapsLink =>
      'https://maps.google.com/?q=${lat.toStringAsFixed(6)},${lon.toStringAsFixed(6)}';

  /// The SMS body.
  ///
  /// SMS is the transport because it needs no data connection - it survives exactly the
  /// conditions this system exists for, and on rural networks it is often the only thing
  /// that works. That means the message must be short and readable by a human on a basic
  /// handset, with the machine-readable link last.
  ///
  /// The staleness line is mandatory. A coordinate that is 4 minutes old during a walk is
  /// a very different instruction from a live one, and the recipient cannot tell without
  /// being told.
  String smsBody({String? name, bool emergency = true}) {
    final who = (name == null || name.isEmpty) ? '' : ' for $name';
    final head = emergency ? 'SOS$who' : 'Location$who';
    final t = '${at.hour.toString().padLeft(2, '0')}:'
        '${at.minute.toString().padLeft(2, '0')}';

    if (!isShareable) {
      return '$head at $t\n'
          'NO GPS FIX YET - position unknown.\n'
          'Moved ${distanceSinceFixM.toStringAsFixed(0)} m by sensors since start.';
    }

    // Rounded UP: a stated staleness must never be shorter than the real one, or the
    // recipient searches a smaller circle than the person could have walked out of.
    final age = fixAgeSeconds < 60
        ? '${fixAgeSeconds.ceil()}s'
        : '${(fixAgeSeconds / 60).ceil()}min';

    return '$head at $t\n'
        '${lat.toStringAsFixed(6)},${lon.toStringAsFixed(6)}\n'
        'within ${uncertaintyM.toStringAsFixed(0)}m ($qualityWord)\n'
        'last GPS $age ago, ${distanceSinceFixM.toStringAsFixed(0)}m by sensors since\n'
        '$mapsLink';
  }

  Map<String, dynamic> toJson() => {
        'lat': lat,
        'lon': lon,
        'uncertainty_m': uncertaintyM,
        'fix_age_s': fixAgeSeconds,
        'distance_since_fix_m': distanceSinceFixM,
        'gps_anchored': gpsAnchored,
        'mode': mode,
        'at': at.toIso8601String(),
      };

  static PositionReport fromJson(Map<String, dynamic> j) => PositionReport(
        lat: (j['lat'] as num).toDouble(),
        lon: (j['lon'] as num).toDouble(),
        uncertaintyM: (j['uncertainty_m'] as num).toDouble(),
        fixAgeSeconds: (j['fix_age_s'] as num).toDouble(),
        distanceSinceFixM: (j['distance_since_fix_m'] as num).toDouble(),
        gpsAnchored: j['gps_anchored'] == true,
        mode: (j['mode'] ?? 'walking').toString(),
        at: DateTime.parse(j['at'] as String),
      );
}
