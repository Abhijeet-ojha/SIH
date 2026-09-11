import 'package:flutter/material.dart';

/// Apple Maps palette. Muted map ground, one saturated accent for the user's own track,
/// and amber reserved exclusively for "GNSS denied" so that colour carries exactly one
/// meaning throughout the app.
class NavTheme {
  static const Color mapLand = Color(0xFF1C1C1E);
  static const Color mapRoad = Color(0xFF2C2C2E);
  static const Color mapRoadCasing = Color(0xFF3A3A3C);
  static const Color mapWater = Color(0xFF13293D);

  static const Color sheet = Color(0xFF1C1C1E);
  static const Color sheetElevated = Color(0xFF2C2C2E);
  static const Color separator = Color(0x33FFFFFF);

  static const Color label = Color(0xFFFFFFFF);
  static const Color secondaryLabel = Color(0x99EBEBF5);
  static const Color tertiaryLabel = Color(0x66EBEBF5);

  static const Color accent =
      Color(0xFF0A84FF); // live position / GNSS-tracked route
  static const Color denied = Color(0xFFFF9F0A); // dead-reckoned segment
  static const Color good = Color(0xFF30D158);
  static const Color bad = Color(0xFFFF453A);

  static ThemeData dark() {
    final base = ThemeData.dark(useMaterial3: true);
    return base.copyWith(
      scaffoldBackgroundColor: mapLand,
      colorScheme: base.colorScheme.copyWith(
        primary: accent,
        surface: sheet,
      ),
      textTheme: base.textTheme.apply(
        bodyColor: label,
        displayColor: label,
      ),
    );
  }

  /// Apple's grouped-list row look: rounded container, hairline separators inside.
  static BoxDecoration get groupedCard => BoxDecoration(
        color: sheetElevated,
        borderRadius: BorderRadius.circular(12),
      );
}
