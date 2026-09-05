# The Flutter app source is not in this repository

## What happened

`flutter_app/` on branch `feature/offline-localization-mobile-mvp` contains the pubspec,
the Android build config, the Gradle wrapper, and **eight unit tests** — but not the
application itself. `flutter_app/lib/` is empty in git.

The cause is one line in the root `.gitignore`:

```
lib/          <- unanchored: matches at ANY depth
```

That rule is there for Python virtualenvs, which only ever appear at the repo root. Because
it was not anchored, git applied it to `flutter_app/lib/` as well and silently refused to
stage the entire Dart source tree. Confirmed with:

```
$ git check-ignore -v flutter_app/lib/main.dart
.gitignore:14:lib/    flutter_app/lib/main.dart

$ git log --all -- flutter_app/lib
(no output — never committed, on any branch)
```

**The APK that exists was built from source that lives on exactly one laptop.** If that
machine is lost, so is the app. Nobody else can build, review, or modify it.

## The fix, already applied on this branch

The rules are now anchored to the repository root:

```
/lib/
/lib64/
```

Python virtualenvs are still ignored. `flutter_app/lib/` is not.

## What Abhijeet needs to do — tonight, before the presentation

From the machine that has the working app:

```bash
git checkout feature/offline-localization-mobile-mvp
git merge fix/frame-invariant-dr        # or cherry-pick the .gitignore fix
git status                              # flutter_app/lib/ should now appear as untracked
git add flutter_app/lib
git commit -m "fix: commit Flutter app source, previously swallowed by unanchored lib/ ignore"
git push
```

Then verify from a clean clone that it actually builds:

```bash
git clone <repo> /tmp/verify && cd /tmp/verify/flutter_app
flutter pub get && flutter test && flutter build apk --debug
```

If `flutter test` passes, the eight committed tests can finally run against real code —
right now they reference files git has never seen.

## Files the committed tests expect

From the `import 'package:navpulse_localizer/...'` lines in `flutter_app/test/`:

| Path under `flutter_app/lib/` |
| :--- |
| `main.dart` |
| `localization/ekf_fusion_engine.dart` |
| `sensors/sensor_calibrator.dart` |
| `sensors/sensor_quality_engine.dart` |
| `models/sensor_quality.dart` |
| `models/session_data.dart` |
| `analytics/session_recorder.dart` |
| `state/navigation_state_provider.dart` |

Use this as a checklist that nothing else is missing after the commit.

## Which app to demo

There are now two:

- **`flutter_app/`** — Abhijeet's, a 5-tab HUD with calibration wizard and session
  analytics. Has a working APK. Source not in git.
- **`android_logger/`** — native Kotlin, rebuilt on this branch as a live HUD with the
  blackout-simulation button. Source is in git. **Never compiled.**

For a presentation tomorrow, demo the one with a working APK. Commit the source either way
so the repository is not a single point of failure.
